# Qwen3 4B direct coding: GPU prefill and reusable context

This experiment measures direct Python coding on mini2 (Apple M4, 16 GB), before resuming agent orchestration work. It uses the same Qwen3-4B-Q4_K_M artifact as the preceding speculative-decoding experiment. It does not grant the model file or command tools.

The new opt-in path substantially reduces long-prompt latency. It does not establish 100 tokens/s on general coding, and the harder scheduler edit still fails the behavioral fixture. Agentic work remains paused; these changes are an inference and measurement foundation.

## New path measurements

The first complete new-path pair measured:

| Long-context request | Prompt processing | Full response wall time | Scheduler test |
| --- | ---: | ---: | --- |
| Prior V3, cold | 25.94 s | 32.57 s | Failed |
| New V3 + Q/K matrix attention, cold | 8.62 s | 15.23 s | Failed |
| New matrix attention, extension with reuse off | 8.70 s | 15.35 s | Failed |
| New matrix attention, extension with reuse on | 0.44 s | 7.05 s | Failed |

The cached extension reused 3,283 of 3,312 prefill positions and processed 29 new positions; the API prompt count additionally includes the final first-token-forward position. Generated token IDs matched the cold matrix reference in all four new-path comparisons, including the reused extension. The API CPU-cache field remained false, as expected for this GPU parking route.

A second fresh-server long-context pair reproduced the result: cold matrix prompt processing took 8.61–8.62 seconds; the reused extension took 0.448 seconds for prompt processing and 7.044 seconds wall time, with the same 3,283 reused positions and exact reference token IDs. Both long-context pairs still failed the scheduler test. The [32-measurement receipt](receipts/qwen4b-direct-coding-20260913.json) includes every passing and failing response, hashes, validation outcomes, and the explicit GPU checks.

Both short coding tasks passed on the new paths: interval merging took about 6.24 seconds to tested code and binary search 3.52 seconds, with roughly 30 tokens/s decode. Against the prior V3 profile, most of this iteration's benefit is on longer context. The previous V3 kernel work accounts for most of the short-task gain over the default path.

## Baseline observations

Before the new prefill changes, the default path completed the passing interval and binary-search tasks in 9.89 and 5.29 seconds including validation. V3 reduced those to 6.42 and 3.70 seconds, at about 29 tokens/s. V4 suffix speculation completed the binary-search repair in 2.62 seconds at 51.98 tokens/s, but interval merging took 11.27 seconds. Speculation is workload-dependent even within this small coding pack.

The long scheduler prompt contains 3,293 tokens; its controlled extension contains 3,313. Baseline prompt processing takes about 26 seconds. The default, V3, and V4 answers fail scheduler behavior despite completing normally: the output mishandles zero-cost jobs after the budget is exhausted and/or stable priority ordering. These are retained as failures with no time-to-tested-code score. Faster generation alone does not establish successful coding.

## Changes under test

`CAMELID_METAL_QK_NORM_ATTN_MM=1` admits dense Q/K-normalized models to existing matrix attention during GPU prefill. Q/K normalization and activation storage remain f32. Attention stages half query, score, and probability panels; this is a different arithmetic path from row attention, so token identity against the old path is not promised. K-quant weights also require `CAMELID_METAL_KQUANT_ATTN_MM=1`.

`CAMELID_QWEN_PREFIX_REUSE=1` separately enables parked resident F16 KV reuse for Qwen3 K-quant sessions on that matrix path. Reuse requires matching model identity and geometry, a common token prefix of at least 256 positions, and a watermark that vouches for those rows. It also records which attention path actually produced the prefix: flags alone are insufficient because a scratch-cap fallback can use row attention. Unknown or incompatible provenance refuses reuse. The last prompt position is always recomputed. A declined continuation rebuilds a fresh engine for full prefill. Captured-layer requests opt out. The record currently covers prompt tokens, not the generated answer.

Before parking, the engine settles any committed encode-ahead graph, including an unchanged-watermark graph waiting for an event, and clears its sampling bookkeeping. Otherwise the next request's prefill or cache-growth copy can block behind the previous request's GPU work.

All new flags are off by default. Existing V3/V4 kernel experiments also remain opt-in.

## Benchmark method

The versioned pack contains three tasks: implement interval merging, repair binary search, and replace a scheduler function within existing module context. The long prompt is measured at its actual tokenizer length rather than described by character count. A controlled extension appends an instruction to the same source prompt and measures reuse of the unchanged prefix; it is not a full agent conversation.

The validator checks 159 hidden behavioral cases, exact output types, input preservation, and a logarithmic loop bound for binary search. It evaluates an explicitly restricted Python AST in a bounded child process; generated output is never compiled or executed as host Python. This is a narrow regression fixture, not a general Python conformance suite or a coding leaderboard.

Requests use greedy sampling and a 768-token cap. Failed and truncated answers remain failures. Time to tested code is request wall time plus validation time, reported only for passing answers. Prompt time includes prefill and first-token forward time. Decode throughput is `(completion_tokens - 1) / decode_seconds`, including speculation overhead where applicable. Nonstreaming results do not measure first visible SSE content. Model loading and the server's startup warm-up occur before timing; “cold” means no reused prompt context.

Profiles run sequentially in isolated servers with separate temporary databases. Normal preview service is stopped for the experiment. Same-arithmetic token comparisons pair V4 suffix with plain V4, and Qwen reuse with cold Qwen matrix attention. Different-arithmetic outputs are assessed by the same hidden behavioral tests.

## Correctness evidence

The explicit Metal proof checks exact KV bits for cold versus split prefill at total/split positions 192/64, 257/131, and 4137/4099. It includes Q/K normalization, GQA, two layers, and distinct hidden/query widths. All three passed on mini2; maximum matrix-versus-row absolute output error was 0.00003177 and relative L2 error at most 0.00001171 on the synthetic fixture. These numeric bounds are fixture observations, not model-wide guarantees.

The pending-graph proof uses a real event-gated GPU copy. Equal and rewound watermarks, signaled and unsignaled graphs, invalid claims, and idempotent cleanup passed. These proofs fail if their required Metal path is unavailable. The expanded matrix proof also checks refusal of a row-generated prefix, preservation through parking, and invalidation by generic watermark updates or host reseeding.

The two existing resident parking tests passed, as did all 33 validator self-checks and the mini2 release build. With the new flags off, the candidate V3 control matched all four prior V3 token sequences and retained comparable timing. API source and model artifacts were unchanged.

```sh
CAMELID_METAL_WIRE=1 CAMELID_METAL_MM=1 CAMELID_METAL_KV_DTYPE=f16 \
CAMELID_METAL_QK_NORM_ATTN_MM=1 \
cargo test --lib metal_qk_norm_matrix_prefill_continuation_proof \
  -- --ignored --nocapture --test-threads=1

CAMELID_METAL_KV_DTYPE=f16 \
cargo test --lib metal_prefix_reuse_releases_pending_graph_at_equal_watermark \
  -- --ignored --nocapture --test-threads=1
```

## Reproduction

Run only on the designated benchmark machine, with other model/build workloads stopped:

```sh
python3 scripts/lib/qwen_coding_fixtures.py --self-check
python3 scripts/bench-qwen-coding.py \
  --binary /path/to/camelid --model /path/to/Qwen3-4B-Q4_K_M.gguf \
  --out target/qwen-coding/run-1 \
  --profiles default,v3,v4,v4suffix,qwenmm,qwenreuse --warm-followup
```

The harness refuses occupied preview/comparison ports, records binary/model/fixture hashes, and terminates only the server it starts. Raw requests, responses, generated token IDs, validation outcomes, timing, and prefix-reuse traces are retained under the chosen output directory. `CAMELID_QWEN_PREFIX_TRACE=1` records actual reused/prefilled token counts; the API's separate CPU prompt-cache-hit field does not describe this parked GPU cache.

Model SHA256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.

The normal preview was restored with its original binary, settings, and saved data after all model tests finished. These experimental flags were not enabled on that preview.
