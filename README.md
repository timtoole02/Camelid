# Camelid

[![CI][ci-badge]][ci-workflow]

**Camelid is trust-first local AI infrastructure for teams that need local models they can actually adopt.**

Most local-model stacks are easy to demo and hard to trust. Camelid is a Rust-native local inference backend for GGUF language models built for the point where local AI becomes real infrastructure: products, internal platforms, regulated environments, and customer-facing workflows.

Camelid does not treat “probably works” as “supported.” Support moves on evidence.

## Why organizations adopt Camelid

Organizations adopt Camelid because local AI needs more than raw inference. They need to know what works, what does not, and whether the runtime, API, and UI all tell the same truth.

Camelid gives teams:

- **clear support contracts** instead of hand-wavy compatibility
- **OpenAI-compatible APIs** that fit existing tools and agent stacks
- **fail-closed readiness** so unsupported models do not masquerade as production-ready
- **evidence-backed support** tied directly to runtime reality
- **one consistent product contract** across backend, API, WebUI, docs, and validation

If you have ever wanted to use local AI without guessing what is actually ready, Camelid is built for that gap.

## Why Camelid matters

The hard problem in local AI is no longer just getting a model to run. The hard problem is operational trust.

Most local inference stacks optimize for breadth first and clarity later. Camelid takes the opposite approach: make support explicit, make readiness visible, and make the product fail closed when the evidence is not there.

That discipline is not just a docs preference. It is the wedge.

## Why people want to try it

Camelid already ships a serious local inference product surface:

- a Rust runtime for GGUF model loading and inference
- OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
- exact-row capability reporting through `/api/capabilities`
- a React/Vite WebUI with fail-closed readiness gating
- parity and validation harnesses that verify behavior against llama.cpp before support language moves

Today, four exact Q8_0 rows are public and evidence-backed:

- **TinyLlama 1.1B Chat Q8_0** — verified support
- **Llama 3.2 1B Instruct Q8_0** — verified support within checked 512/1024/2048/4096/8192 bounded exact-row contexts
- **Llama 3.2 3B Instruct Q8_0** — verified support within checked 512/1024/2048 bounded exact-row contexts
- **Llama 3 8B Instruct Q8_0** — exact-row smoke support within checked 512/1024/2048 bounded contexts

Mixtral has one-token backend MoE runtime evidence but is not yet promoted to API/WebUI/frontend readiness. `Mistral-7B-Instruct-v0.3.Q8_0.gguf` is the active next exact-row bring-up lane.

## Why this becomes a company

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

Camelid achieves 1:1 parity with llama.cpp only for the supported exact GGUF rows with published validation evidence: TinyLlama at the current validated gate; Llama 3.2 1B Q8_0 with verified bounded exact-row support at checked 512/1024/2048/4096/8192-context packs; Llama 3.2 3B Q8_0 with verified bounded exact-row support at checked 512/1024/2048-context packs; and Llama 3 8B Q8_0 through exact-row smoke plus checked 512/1024/2048-context packs tied to cited source/runtime-head bundles.

`Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` has one-token backend MoE runtime evidence, but Gate 9A later-generation evidence diverges and a longer-continuation backend HTTP hang remains unresolved, so Mixtral API/WebUI/frontend readiness and broad Mixtral support are not claimed. `Mistral-7B-Instruct-v0.3.Q8_0.gguf` remains an active exact-row bring-up lane, not a supported row.

## Current execution tracks

Camelid is advancing on two tracks:

- **Supported-row hardening:** preserve TinyLlama as the full current gate, keep verified Llama template/Jinja and throughput wording tied to green exact-row evidence, and continue portability/broader-context work without blurring support scope.
- **Active next-model bring-up:** Mistral 7B Instruct is the lead exact-row bring-up lane; Qwen 2.5 7B Instruct and Gemma 2 9B Instruct remain planned exact-row candidates.

For deeper row-by-row promotion rules and blocker detail, see [`COMPATIBILITY.md`](COMPATIBILITY.md#locked-next-family-readiness-language) and [`STATUS.md`](STATUS.md).

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
| Llama 3.2 1B Instruct Q8_0 | **Verified support (bounded exact-row)** | Load, completions, chat completions, WebUI validation, compact/broader parity, exact-row metadata-Jinja row-template checks, template-shape checks, unique-chat perf/RSS sampling, and checked 512/1024/2048/4096/8192-context packs. |
| Llama 3.2 3B Instruct Q8_0 | **Verified support (bounded exact-row)** | Load, completions, chat completions, WebUI validation, metadata-Jinja exact-row template execution, compact/broader 50-token parity, template-shape checks, unique-chat perf/RSS, checked 512/1024/2048-context packs, and an opt-in parallel Q8 first-token direction probe. |
| Llama 3 8B Instruct Q8_0 | **Verified support (bounded exact-row)** | Load, completions, chat completions, WebUI validation, compact parity, three-prompt 50-token parity, checked 512/1024/2048-context packs, compact chat-template-shapes pack, memory evidence, structured RSS/Q8 file-read counters, and lazy-Q8 hot-path measurements. |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | **In active validation; not supported yet** | Source/SHA, exact tokenizer/template references, 1-token generation parity, broader five-prompt/50-token parity, checked 512/1024/2048 bring-up, checked 4096/8192 context validation, and fail-closed API/WebUI/RSS evidence are green; latest context bundle: `qa/evidence-bundles/mistral-7b-v0.3-q8-context-4096-8192-ubuntu-20260509T005229Z-head-9e3c64f2cfab/manifest.json`. |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | **Active validation; partial backend runtime only** | One-token backend MoE runtime evidence exists, with tokenizer/template and sparse MoE metadata proof. Blocker anchors: `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-gate9a-50tok-20260511/`, `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-longgen-continuation-20260511/`, and `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/`. |
| Qwen2.5-7B-Instruct-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. |
| gemma-2-9b-it-Q8_0.gguf | **Planned exact-row candidate** | Candidate row selected for acquisition/tokenizer planning only. |

### Latest model checks

The maintainer matrix now includes four exact Q8_0 supported rows with checked row-specific evidence. Mistral remains an active validation row, and Mixtral remains partial backend runtime evidence only, not a supported row. These are exact-row checks, not universal model claims.

| Exact row | Latest checked bucket | Result | Output checked |
| --- | --- | --- | --- |
| TinyLlama 1.1B Chat Q8_0 | direct chat validation | PASS | `Certainly! Here` |
| Llama 3.2 1B Instruct Q8_0 | 8192-context recall pack | PASS on cited source/runtime head `aaf9207d1669` | `CMLD-819` |
| Llama 3.2 3B Instruct Q8_0 | 2048-context recall pack | PASS | `CMLD-204` |
| Llama 3 8B Instruct Q8_0 | 2048-context recall pack | PASS on cited source/runtime head `8e26be0a73c0` | `CMLD-204` |
| Mistral-7B-Instruct-v0.3.Q8_0.gguf | Active validation: Ubuntu bring-up plus exact tokenizer/template, 1-token, broader five-prompt/50-token parity, checked 4096/8192 context evidence, and fail-closed API/WebUI/RSS evidence | PASS for validation lane; still unsupported | Explicit contract promotion and synchronized support surfaces still required before any support claim |
| Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf | Active validation: one-token backend MoE runtime evidence plus Gate 9A/continuation blocker bundles | BLOCKED by later-generation divergence and backend HTTP hang | No Mixtral API/WebUI/frontend readiness, neighboring-row, long-context, arbitrary-template, production-throughput, portability, or broader/full claim |

### Read this boundary carefully

- Support does **not** inherit across model sizes, variants, quantizations, tokenizer lanes, or nearby GGUFs.
- Support language currently means only the exact supported rows above; Mistral has no support claim yet and may only be discussed as the exact active validation row above. The Mixtral row may only be described as one-token backend MoE runtime evidence with later-generation/API/WebUI/frontend readiness blocked.
- Checked context packs do **not** imply model-native or broader context support.
- Template and bounded perf/RSS evidence for current supported rows is exact-row scoped; it does **not** imply broad arbitrary/Jinja-template behavior, production throughput, neighboring GGUFs, portability, or broader context support.
- The next exact-row 8B, `Mistral-7B-Instruct-v0.3.Q8_0.gguf`, and `Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf` gaps are broader context, portability, repeated durability evidence, and normalized support bundles; Mistral also still needs explicit contract promotion after fail-closed API/WebUI/RSS readiness, and Mixtral must first fix later-generation divergence and the continuation hang before any API/WebUI/frontend support claim.

Authoritative details live in [`COMPATIBILITY.md`](COMPATIBILITY.md). The current evidence snapshot lives in [`STATUS.md`](STATUS.md).

## Start here

- [`COMPATIBILITY.md`](COMPATIBILITY.md) — authoritative support matrix and promotion rules
- [`STATUS.md`](STATUS.md) — current evidence boundary, recent moves, and blockers
- [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) — shortest safe local path
- [`docs/VALIDATION_MATRIX.md`](docs/VALIDATION_MATRIX.md) — expected checks by change type
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution expectations and PR guidance
- [`DOCS.md`](DOCS.md) — full documentation map

## Quickstart

This quickstart verifies that Camelid builds cleanly, starts the backend, and returns a live API response. It is intentionally simple and honest: the repository does not bundle supported GGUF model files, so full local chat requires one additional setup step after the server is running.

### 1) Build and run the server

```bash
git checkout main
git pull --ff-only
cargo build --release --bin camelid
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
- the model path wired into a load request you control
- any extra contributor setup described in [`docs/CONTRIBUTOR_QUICKSTART.md`](docs/CONTRIBUTOR_QUICKSTART.md) and [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)

For a first supported local run, TinyLlama is the clearest path. It is **not bundled** in this repository, but once you have a supported GGUF locally and a load request wired in, Camelid is designed to make the readiness boundary explicit instead of guessing.

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
