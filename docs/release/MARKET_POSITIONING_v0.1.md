# Camelid v0.1 Market Positioning

Date: 2026-05-31

Release checkout: branch-agnostic working notes; record the exact branch and SHA when cutting rc1

## Positioning Summary

Camelid v0.1 should be positioned as an evidence-gated local GGUF inference project for exact-row compatibility work. The strongest public story is not speed bravado; it is clarity about what is supported, what is measured, and what is still blocked.

Recommended short description:

> Camelid is a Rust-native local GGUF inference backend with exact-row support claims backed by committed parity, API/WebUI, context, memory, and benchmark evidence.

## Audience

Primary v0.1 readers:

- local-inference engineers who want auditable support boundaries
- Rust systems developers evaluating the runtime architecture
- contributors who need a clear validation contract before changing support-sensitive code
- reviewers comparing Camelid evidence against existing local inference runtimes

This release is not aimed at casual "download any model and chat" users yet.

## Differentiators To Use

Use these claims when they are paired with the supporting docs:

- Exact-row support, not family-wide guesswork.
- Compatibility language synchronized across docs, API capability reporting, and frontend readiness state.
- Committed evidence bundles for parity, context, API/WebUI, memory, and bounded benchmark lanes.
- Explicit unsupported and active-validation states for partial rows such as Mixtral.
- Rust-native implementation with clear release gates.

## Claims To Avoid

Do not use these claim types for v0.1:

- broad performance superlatives
- broad readiness labels without a release gate
- hardware-saturation language without a measured release artifact
- all-GGUF support
- broad Llama, Mistral, Mixtral, Qwen, or Gemma support
- comparisons to specific hosted assistant UIs
- distributed inference readiness
- comparator speed wins over llama.cpp, MLX, or Ollama
- arbitrary Jinja template support
- model-native long-context support beyond checked packs

Several of these may become true later in narrow forms, but v0.1 should not spend credibility early.

## Comparator Language

Use:

- "Camelid checks parity against llama.cpp for cited exact rows."
- "Camelid publishes bounded benchmark and memory snapshots where committed artifacts exist."
- "A complete same-host throughput table versus comparator runtimes is still pending."

Avoid:

- uncited speed-win claims over llama.cpp
- uncited speed-win claims over MLX
- replacement claims against Ollama

## Frontend Language

Use:

- "local React/Vite chat surface"
- "truthful readiness state"
- "chat unlocks only for recognized supported rows"

Avoid:

- comparisons to specific hosted assistant UIs
- clone-language for hosted assistant products
- universal model chat
- production chat UI

## Distributed Language

Distributed code and experiments may be mentioned as implementation areas, but not as v0.1 release readiness. Public v0.1 docs should avoid presenting distributed inference as a supported user workflow unless a release-captain-approved evidence bundle and gate result are added.

## One-Sentence Public Pitch

Camelid v0.1 is a Rust GGUF inference release candidate that favors precise, evidence-backed compatibility over broad unsupported claims.

## Release Narrative

The v0.1 story should read like this:

1. Camelid can run a defined set of exact GGUF Q8_0 rows.
2. Each supported row has a bounded evidence trail.
3. Partial rows are named honestly and kept blocked where evidence fails.
4. Benchmarks are useful but narrow.
5. The next release work is fresh v0.1 evidence bundles and comparator baselines, not louder copy.
