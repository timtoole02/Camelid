# Qwen3 4B speculative decoding and Metal experiment

Measured on mini2, Apple M4 with 16 GB unified memory, September 13–14, 2026. Builds, tests, and model runs were serialized on that machine. No inference or builds ran on mini1.

## Scope and result

The 100 tokens/second **decode** target has been reached for prompt-copying, not ordinary chat. Full 368-token copying responses measured **148.92 and 149.00 tok/s** in independent runs, and **148.86 tok/s through actual SSE streaming**, with the revised suffix drafter and opt-in verifier batching. Speculative token IDs matched their same-kernel plain references; streaming text and usage matched too. Ordinary streaming prose/code measured **29.82–29.85 tok/s** with the existing experimental V3 single-token path, versus roughly **18 tok/s** with the default kernels. Earlier short non-streaming V3 probes ranged around 30–32 tok/s.

V3 and V4 remain experimental, explicit environment options. They use different arithmetic from the default kernels. In particular, V4 rounds some Q6 operands to half precision. Matching a speculative result to its own plain-kernel reference does **not** qualify that kernel against the default model or establish general answer quality. No arithmetic option is enabled by this change.

## Implementation

- Disable target encode-ahead for speculative API requests, both during preparation and at cooperative streaming boundaries. An unused target command buffer could wait on an unsignaled event while preventing the smaller draft model from running on the same serial Metal queue. The standalone speculative benchmark already used this exclusion. The original API experiment stalled after 29 tokens; the fixed non-streaming experiment completed all three probes.
- Give the linear suffix drafter its own frequency-based chain construction. The previous code flattened the deepest leaf of a breadth-first tree, which could select a less frequent sibling and spend the proposal budget on branches. The chain now searches contexts up to 32 tokens in a bounded 8,192-token history. Existing tree-drafter policy is unchanged.
- Add opt-in `CAMELID_SPEC_VERIFY_BATCH_GLUE=1` for linear verification with an F16 primary cache and more than one row. Reuse existing per-head normalization, batched RoPE, and batched F16 cache-write kernels, reducing up to five dispatches per row to five per layer. Tree, width-one, F32-primary, and Q8-primary paths retain their row-wise implementation. `CAMELID_METAL_ATTN_BATCH_K=1` is the separate existing batched-attention option.
- Add an isolated API benchmark harness with same-kernel parity checks, full requests/responses, model and binary hashes, non-streaming token IDs, and actual SSE timing/usage. A parity failure records the result and fails the run. Raw receipts live in ignored `target/qwen-speed/` directories.

## Initial comparison

All rows use the same Qwen3-4B-Q4_K_M GGUF, greedy temperature zero, no tools, disabled thinking, and a 128-token cap. Each row has a fresh process and temporary state. These short code outputs are truncated and are not code-quality tests.

| Binary / configuration | Prose tok/s | Code tok/s | Copy tok/s |
| --- | ---: | ---: | ---: |
| Original, default kernels | 17.94 | 17.95 | 17.61 |
| Original, V2 | 22.58 | 22.59 | 22.01 |
| Original, V3 | 30.33 | 31.13 | 29.54 |
| Original, V4 | 17.43 | 17.41 | 17.03 |
| Original, V4 + old suffix, 15 drafts | 16.38 | 17.73 | 48.18 |
| Prototype, V3 plain | 32.50 | 29.94 | 28.73 |
| Prototype, V3 + Qwen3 0.6B Q8 draft, 3 drafts | 20.33 | 25.17 | 17.47 |
| Prototype, V4 + batched attention, plain | 17.39 | 17.38 | 16.98 |
| Prototype, V4 + batched attention + revised suffix, 15 drafts | 15.63 | 16.89 | 124.66 |

All completed speculative prototype outputs matched their same-kernel plain references exactly. The smaller-model drafter now completes but its overhead outweighs the saved target work in these probes. Suffix drafting pays off when the answer repeats material already in context; it can slow down novel prose. Batched verification alone cannot overcome low draft acceptance.

The copying result includes 1.484 seconds of prompt processing followed by 1.019 seconds of decode, 2.519 seconds request wall time. The headline rate excludes prompt processing; it is not an end-to-end 125-token/second rate.

## Final GPU-batching comparison

The final binary uses the same arithmetic on both sides of this comparison. All speculative output token IDs match the corresponding plain references. Prose ends naturally at 148 tokens; the other outputs reach the 256-token cap.

| Configuration | Prose tok/s | Code tok/s | Copy tok/s | Copy after long background tok/s |
| --- | ---: | ---: | ---: | ---: |
| V4 + batched attention, plain | 17.40 | 17.37 | 17.13 | 14.04 |
| Same + revised suffix, 15 drafts | 15.48 | 16.05 | 138.68 | 50.41 |
| V4 + attention/glue batching, plain | 17.35 | 17.29 | 17.06 | 14.12 |
| Same + revised suffix, 15 drafts | 15.55 | 16.11 | 144.98 | 51.52 |

The short copy A/B shows a 4.5% decode-rate increase from the new glue batching in this run. The long-context increase is smaller, about 2.2%. Both are preliminary timing observations; the exact-output checks are separate from the performance comparison.

The long prompt contains **4,184 tokens**, compared with 392 for the short copy prompt. Long-context prompt processing takes about **39.3 seconds**, and the complete speculative request about **44.3 seconds**. Speculation accelerates decoding after the prompt; it does not solve this prefill delay. General chat, long-context latency, and copy-heavy throughput must therefore remain separate performance targets.

Final tested binary SHA256: `a7f43698a6fdfa5567c70795be14739935d2ad73a4ef55907e5ef86e38cd3744`.

## Full responses and streaming

Two independent full-copy A/B runs used a 512-token cap and both stopped naturally at 368 tokens. Each arm used a fresh process, and every speculative result matched the same-kernel plain token IDs.

| Full copy | Earlier verifier tok/s | New glue batching tok/s |
| --- | ---: | ---: |
| Repeat 1 | 142.25 | 148.92 |
| Repeat 2 | 142.46 | 149.00 |

The new batching improves full-copy decode throughput about **4.6–4.7%** in these repeats. The full request takes about **4.02 seconds**, including approximately **1.52 seconds** of prompt processing. That is about 91.5 generated tokens per wall-clock second, distinct from the 149 tok/s decode rate.

Actual SSE checks:

| Profile / workload | Decode tok/s | First visible content | Output tokens | Outcome |
| --- | ---: | ---: | ---: | --- |
| V3 plain prose | 29.82 | 0.485 s | 150 | completed |
| V3 + 0.6B drafter, prose | 21.12 | 0.448 s | 150 | exact plain text and usage |
| V3 plain code | 29.85 | 0.455 s | 256 | token cap |
| V3 + 0.6B drafter, code | 24.84 | 0.443 s | 256 | exact plain text and usage |
| V4 + glue, plain copy | 17.08 | 1.657 s | 368 | completed |
| V4 + glue + suffix, copy | 148.86 | 1.653 s | 368 | exact plain text and usage |

The speculative copying stream emits 26 content chunks for 368 sampled tokens, including EOS. Chunks must not be counted as tokenizer tokens. The rate uses terminal usage and server timings; first-content latency is measured by the HTTP client on mini2.

## Validation and remaining work

- Seven suffix tests and 42 speculative-logic tests passed; two pre-existing tests in the latter filter remain ignored.
- The explicitly armed Metal proof passed with 14 actual batched encodes, exact logits, and exact intermediate captures. It did not skip.
- Release build, Rust formatting, benchmark Python compilation, SmolLM3 source-pin checks, and qualification-roster checks passed on mini2. API source pins were refreshed mechanically; qualification status was not promoted.
- The final default-kernel binary reproduced all 128 token IDs from the original default baseline for prose, code, and copying. Plain V4 outputs also matched with the glue flag on/off across all four contexts.
- [Compact receipt](receipts/qwen4b-spec-metal-20260913.json) retains 66 completed measurements, hashes, configuration, token counts, parity outcomes, and the original incomplete drafter experiment. Full private receipts retain raw requests/responses and logs.

**Ordinary chat remains below the 100 tok/s target.** The next work should address draft cost/acceptance and the long-context prefill path. The independent Qwen3 0.6B drafter is reliable after the queue fix but provides no speed gain on the tested prose/code. Broad answer-quality qualification is still required before adopting experimental arithmetic defaults.

Code mode uses `LiveDriver` with temperature zero against the same inference API, so eligible agent requests can use these improvements. Reusing code already in context is a plausible suffix-drafting benefit. These probes did not run agent tasks or qualify tool-call correctness, edit quality, planning, or end-to-end coding throughput; the 149 tok/s figure must not be advertised as an agentic-coding benchmark.

The existing preview was restored with its original binary and settings after all heavy tests completed. Saved preview data and project files were preserved; experimental settings remain opt-in.

## Reproduction

Use `scripts/bench-qwen4b-metal-spec.py` on the benchmark machine with other model workloads stopped. It refuses occupied preview/comparison ports and owns only its child server. Supply local model paths; model downloads and host configuration are deliberately outside the harness.

```sh
python3 scripts/bench-qwen4b-metal-spec.py \
  --binary /path/to/camelid --model /path/to/Qwen3-4B-Q4_K_M.gguf \
  --out target/qwen-speed/repeat-1 --max-tokens 256 \
  --arms v4batch,v4batch-suffix-15,v4glue,v4glue-suffix-15 \
  --cases prose,code,copy,deepcopy

python3 scripts/bench-qwen4b-metal-spec.py \
  --binary /path/to/camelid --model /path/to/Qwen3-4B-Q4_K_M.gguf \
  --draft-model /path/to/Qwen3-0.6B-Q8_0.gguf \
  --out target/qwen-speed/stream --stream \
  --arms v3,v3-draft-3 --cases prose,code
```

The reported non-streaming rate is `(completion_tokens - 1) * 1000 / (timings.generate - prompt_evaluation.prefill.forward_total - prompt_evaluation.first_token.forward_total)`. It includes drafting and verification overhead, not just the target's GPU counters. SSE records first visible content, terminal usage, completion, and server timing; SSE parity is text plus token counts because that endpoint does not expose output token IDs.

For the fail-closed GPU exactness proof, run in a fresh process:

```sh
CAMELID_METAL_KV_DTYPE=f16 CAMELID_METAL_F32Y=1 \
CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 \
CAMELID_METAL_ATTN2=1 CAMELID_METAL_ATTN_SPLITK=1 \
CAMELID_METAL_ATTN_BATCH_K=1 CAMELID_SPEC_VERIFY_BATCH_GLUE=1 \
cargo test --lib metal_spec_verify_batched_qk_glue_bit_identical \
  -- --ignored --nocapture --test-threads=1
```

This proof asserts actual dispatch, every logit's bits, argmax IDs, and intermediate captures. It uses weighted Q/K normalization, head dimension 128, GQA ratio four, split-K boundary positions 126/510, and base 4090 with widths 1, 2, 7, 8, 15, and 16. It fails if required gates or Metal are absent. It validates batching with synthetic weights; live Q4_K_M token comparisons separately validate the tested model runs.

Model SHA256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`. The GGUF metadata's “Awq” name is not the actual quantization.
