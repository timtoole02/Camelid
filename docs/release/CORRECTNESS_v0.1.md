# Camelid v0.1 Correctness Boundary

Status date: 2026-05-31

This file defines the correctness language for Camelid v0.1. It supports the release-candidate support matrix in `SUPPORT_MATRIX_v0.1.md`.

## Correctness Definition

For v0.1, correctness means bounded exact-row parity against the cited known-good reference for the stated prompt pack and context bucket:

- prompt token IDs match
- generated token IDs match
- generated text matches
- the model file, quantization, tokenizer/template path, context window, token budget, and source/runtime head match the cited evidence

Correctness does not imply broad model-family support, neighboring-row support, production throughput, portability, arbitrary templates, or model-native/larger contexts.

## Correctness by Supported Row

### TinyLlama 1.1B Chat Q8_0

Public correctness claim:

Camelid v0.1 has a full current-gate correctness proof for the exact TinyLlama Q8_0 row: broader five-prompt 50-token parity, marker-template shape parity, bounded 512-context parity, API/WebUI smoke, and RSS/perf evidence.

Evidence:

- `qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/manifest.json`

Boundary:

This proves only the exact TinyLlama Q8_0 row and checked packs. It does not prove other TinyLlama variants, other quants, arbitrary templates, broader LLaMA-family rows, or contexts beyond the cited evidence.

### Llama 3.2 1B Instruct Q8_0

Public correctness claim:

Camelid v0.1 has bounded correctness proof for the exact Llama 3.2 1B Instruct Q8_0 row through the checked 512/1024/2048/4096/8192 context packs, plus API/WebUI smoke, compact template-shape evidence, row-scoped metadata-Jinja evidence, and bounded unique-chat RSS/perf evidence.

Evidence:

- `qa/evidence-bundles/full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-1024-20260505T081001Z-head-156ded6fc76b/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-2048-rope-factors-20260506T0105Z-head-62f8cbc/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-4096-current-head-20260513T163426Z-head-470388f/manifest.json`
- `qa/evidence-bundles/llama32-1b-context-8192-current-head-20260513T183501Z-head-aaf9207d1669/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`

Boundary:

4096 and 8192 are compact-template bounded packs tied to their cited source/runtime heads. The claim does not extend to model-native/larger contexts beyond checked packs, arbitrary templates beyond the row-scoped renderer, production throughput, portability, neighboring rows, or broad Llama support.

### Llama 3.2 3B Instruct Q8_0

Public correctness claim:

Camelid v0.1 has exact-row smoke correctness proof for the exact Llama 3.2 3B Instruct Q8_0 row: canonical API/WebUI support-gate evidence, compact/broader parity, checked 512/1024/2048 context packs, compact template-shape evidence, bounded unique-chat RSS/perf evidence, and an opt-in parallel Q8 first-token direction probe.

Evidence:

- `qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json`
- `qa/evidence-bundles/full-support-normalized-wp1-20260505T032406Z-head-bcf9e647d6fd/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama32-3b-context-1024-20260505T094258Z-head-c14e5e7b5692/manifest.json`
- `qa/evidence-bundles/llama32-3b-context-2048-20260505T105742Z-head-36ec8e492d65/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-chat-template-shapes-20260505T060036Z-head-e9f28572e090/manifest.json`
- `qa/evidence-bundles/llama32-1b-3b-unique-chat-perf-rss-20260505T061644Z-head-e9f28572e090/manifest.json`
- `qa/evidence-bundles/llama32-3b-parallel-q8-first-token-20260505T140400Z-head-ffc22b85214f/manifest.json`

Boundary:

This is exact-row smoke support, not full support. It does not prove model-native/larger context beyond 2048, arbitrary/Jinja template coverage beyond row-scoped evidence, production throughput beyond bounded RSS/perf and the first-token direction probe, portability, neighboring rows, or broad Llama support.

### Llama 3 8B Instruct Q8_0

Public correctness claim:

Camelid v0.1 has bounded exact-row correctness proof for the exact Llama 3 8B Instruct Q8_0 row within the checked smoke/parity envelope: compact and broader 50-token parity, API/WebUI/RSS smoke, 512/1024/2048 context packs, compact template-shape coverage, and measurement-only lazy-Q8 hot-path evidence.

Evidence:

- `qa/evidence-bundles/full-support-normalized-wp2-8b-watchdog-20260505T041404Z-head-83c21f0cbf5a/manifest.json`
- `qa/evidence-bundles/8b-checkmark-current-main-20260505T084931Z-head-15bfc41d15d5/manifest.json`
- `qa/evidence-bundles/llama3-8b-broader-50tok-20260505T005031Z-head-d13541ad8d7e/manifest.json`
- `qa/evidence-bundles/four-row-context-512-20260505T051510Z-head-b403884/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-512-20260504T234625Z-head-58acf592345c/manifest.json`
- `qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json`
- `qa/evidence-bundles/llama3-8b-chat-template-shapes-20260505T003821Z-head-d13541ad8d7e/manifest.json`
- `qa/evidence-bundles/llama3-8b-api-webui-rss-clean-20260505T015843Z-head-aee469b9c13a/manifest.json`
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-20260505T021411Z-head-723a665/manifest.json`
- `qa/evidence-bundles/llama3-8b-lazy-q8-hotpath-helper-validated-20260505T0350Z-head-e22307f2f90b/manifest.json`

Boundary:

The 1024/2048 packs are tied to source/runtime head `8e26be0a73c0`. Lazy-Q8 hot-path artifacts are measurement evidence, not speed/support expansion. The claim does not extend to model-native/larger contexts beyond 2048, arbitrary templates, production throughput, portability, neighboring rows, or broad 8B/Llama support.

## Unsupported Correctness Claims

### Mistral-7B-Instruct-v0.3 Q8_0

Correctness status:

Evidence-only. There is strong row-specific parity/context bring-up evidence, but v0.1 must not call this row supported because the API/WebUI support-surface manifest is fail-closed.

Evidence:

- `qa/evidence-bundles/mistral-7b-v0.3-q8-1tok-parity-20260508T1847Z-head-fa7efc086c0e/manifest.json`
- `qa/evidence-bundles/mistral-7b-v0.3-q8-broader-50tok-ubuntu-20260509T000633Z-head-d330e97ae992/manifest.json`
- `qa/evidence-bundles/mistral-7b-v0.3-q8-context-512-1024-2048-ubuntu-20260508T203513Z-head-86ad5390d265/manifest.json`
- `qa/evidence-bundles/mistral-7b-v0.3-q8-context-4096-8192-ubuntu-20260509T005229Z-head-9e3c64f2cfab/manifest.json`
- `qa/evidence-bundles/mistral-7b-v0.3-q8-api-webui-rss-current-head-20260513T1935Z-head-9a296ea/manifest.json`

Reason support is rejected:

The API/WebUI/RSS manifest records `support_claim` as none, expected `compatibility_status=active_validation_unsupported`, `contract_supported=false`, and WebUI chat blocked. That directly conflicts with later prose that says Mistral is supported. v0.1 should resolve the conflict conservatively.

### Mixtral-8x7B-Instruct-v0.1 Q8_0

Correctness status:

Unsupported beyond bounded one-token backend MoE runtime evidence.

Evidence:

- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-backend-parity-refresh-20260511/manifest.json`
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-gate9a-50tok-20260511/summary.json`
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-longgen-continuation-20260511/summary.json`
- `qa/evidence-bundles/mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/manifest.json`
- `qa/evidence-bundles/sev1-mixtral-8x7b-q8-nontoy-current-head-20260514T114731Z-head-7ac14462bbe7/summary.json`

Reason support is rejected:

Gate 9A diverges at generated token index 9 before 50 tokens, and the continuation lane records a backend HTTP hang. The later non-toy API probe is runtime-only and explicitly does not remove those blockers.

### Planned Rows

`Qwen2.5-7B-Instruct-Q8_0.gguf` and `gemma-2-9b-it-Q8_0.gguf` remain planned candidates only. `COMPATIBILITY.md` and `STATUS.md` mention them as planned rows, but no row-specific support manifest was found under `qa/evidence-bundles`.

## First-Divergence Procedure

Use this procedure before changing runtime code or support wording.

1. Freeze the exact row: model filename, SHA256, quantization, source/runtime head, prompt pack, context window, max tokens, render mode, API/frontend support state, and environment flags.
2. Reproduce with deterministic settings and the row's existing harness. Use the row-specific script when available: `scripts/chat-parity-tinyllama.mjs`, `scripts/chat-parity-llama3.mjs`, `scripts/chat-parity-mistral.mjs`, or `scripts/run-llama3-prompt-pack.mjs`.
3. Record prompt token IDs for Camelid and the known-good reference. If prompt tokens differ, stop at tokenizer/template/BOS/EOS/control-token handling. Do not inspect dense runtime until prompt parity is restored.
4. If prompt tokens match, compare generated token IDs and generated text. Record `first_generated_token_diff_index` and `first_generated_text_diff_index`.
5. Dump logits/top-k around the first differing generated token when available. Record the selected token, known-good token, rank, and logit deltas.
6. Add ordered forward diagnostics only after prompt parity is proven. Compare stages in order: embedding, layer input, Q/K/V projections, RoPE-applied Q/K, attention scores/output, post-attention residual, pre-MLP RMSNorm, gate/up/down projection, SwiGLU activation, post-MLP residual, final norm, and logits.
7. Use the existing diagnostic tools where applicable: `scripts/extract-forward-trace.mjs`, `scripts/compare-forward-traces.mjs`, `scripts/compare-tensor-dumps.mjs`, `scripts/compare-attention-checkpoints.mjs`, `scripts/check-forward-trace-invariants.mjs`, and `scripts/check-output-projection-layout.mjs`.
8. Stop at the first proven divergence. Do not change multiple suspected components at once.
9. Preserve known guardrails, especially token-major `output.weight` behavior and exact Q8_0 support boundaries.
10. Add or update a row-specific regression artifact, then rerun public evidence checks before changing any support language.

## Evidence Publication Checklist

Before promoting or changing correctness language, the durable bundle must include:

- `manifest.json` with exact row id, model file, SHA256, source/runtime head, passed state, and explicit claim boundary
- row results showing prompt-token, generated-token, and generated-text parity
- API/frontend support contract fields if support is user-visible
- context window and max-token fields for context claims
- RSS/timing fields if memory or speed is mentioned
- `SHA256SUMS` when available
- privacy/public-scrub validation

Current public-evidence validation command:

```sh
node scripts/check-public-evidence-claims.mjs --root qa/evidence-bundles
```

That command passed on this checkout before these v0.1 docs were written.
