# Llama 3.2 3B EAGLE-3 benchmark

## 2026-08-29 wide-hybrid follow-up

This document preserves the original linear-head campaign and its promotion
boundary. The later suffix-first, dynamic-tree, lazy-head, row-batched
attention, and width-16 verifier campaign is recorded in
`qa/speculative/LLAMA32_3B_MINI2_WILD.md`.

That follow-up measured a stable **201.67 tok/s** mean on a 115-token exact
recurrence and **161.58 tok/s** at 1,819 prompt tokens, both at 100% suffix
acceptance. Those are deliberately labelled recurrence ceilings, not general
chat. The measured learned or mixed lanes reached **42.47-56.09 tok/s** and
remain benchmark-only. All claimed rows matched their paired target-authoritative
plain stream exactly. The implementation is still EAGLE-3 plus model-free
suffix drafting, not native MTP, and is not wired into serving.

## Status and claim boundary

This is a benchmark-only implementation of a learned EAGLE-3 sidecar for one
exact Llama 3.2 3B target. The original campaign below drafts a linear top-1
chain; the follow-up adds bounded dynamic trees and suffix-first chains. In all
cases the target model authoritatively verifies every proposed token. It is
**not** a native MTP
head in the target model, is not wired into serving, is not enabled by default,
and does not widen Camelid's supported-model claims.

The original gamma sweep below is retained as tuning history. The promoted
receipts were rerun from committed source on mini2 and pin the source commit,
binary, target, head, tokenizer metadata, effective prompt bytes, execution
plan, Metal device, acceleration flags, generated token ids, and draft token
ids. This is still benchmark evidence, not a release or support promotion.

## Exact artifacts

| Role | Artifact | Revision | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| Target | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | exact GGUF identified by its hash | 2,019,377,696 | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |
| Learned head | `thoughtworks/Llama-3.2-3B-Instruct-Eagle3/model.safetensors` | `02d343789b502a3edfe351bdd4537a44affb98cd` | 486,297,280 | `c0713251464a9b6b5fcf9fb229587bbe59b6fd1521027aef32101d11b9ebbdaf` |

`bench-eagle3` hashes both files and refuses any other target or head. The
head README declares the Meta Llama 3.2 Community License; it is not an MIT
artifact.

## Architecture and execution contract

The target admission gate requires this exact geometry:

| Field | Required value |
| --- | ---: |
| GGUF architecture | `llama` |
| Hidden width | 3,072 |
| Decoder blocks | 28 |
| FFN width | 8,192 |
| Attention heads / KV heads | 24 / 8 |
| Head dimension | 128 |
| Target vocabulary | 128,256 |

The head is a single 3,072-wide decoder layer with an 8,192-wide SwiGLU FFN,
24 attention heads, 8 KV heads, 128-wide heads, RoPE theta 500,000, and RMSNorm
epsilon `1e-5`. Its output projection covers a 32,000-row draft vocabulary,
which is mapped back into the 128,256-token target vocabulary.

The loader admits exactly 15 tensors: nine BF16 matrices, four BF16 RMSNorm
vectors, an I32 `d2t` vector, and a BOOL `t2d` mask. It validates every tensor
name, dtype, shape, byte range, and full payload coverage. The four norms are
decoded to f32 for execution. A draft row `i` maps to target token
`i + d2t[i]`; the mapping is range-checked, strictly increasing, and
cross-checked against `t2d`.

EAGLE-3 consumes target decoder layer-input taps `[2, 14, 25]`. Camelid
interleaves those three 3,072-wide captures as
`[low || middle || high]`, applies the head's 9,216-to-3,072 feature fusion,
and combines the fused target state with the target token embedding for the
one-layer recurrent head. The current resident lane keeps both the target and
learned head on Metal. Recursive draft rows are ephemeral; only target-verified
rows extend the stable head cache, and every round checks that the target and
head cache watermarks still agree.

The original benchmark mode is greedy and linear top-1. `gamma`
(`--draft-tokens`) is the maximum number of draft tokens proposed for one target
verify round; it is a runtime tuning knob, not part of the checkpoint
architecture. The later follow-up reuses the same greedy verification contract
with a bounded dynamic tree and an independent suffix-first lane.

## What the head README says about training

The matching head README records a training batch of `B=8`, sequence length
`T=1024`, and gradient accumulation 4. It also records TTT length 7: during
training the head autoregressively rolls out seven tokens and receives a
geometrically weighted multi-step loss.

Those are training settings. **Sequence length 1024 is a training-distribution
fact, and rollout length 7 is not a runtime gamma cap.** Neither the source
head's `config.json` nor its README declares a runtime context limit. In
particular, a `2048` limit from an older Llama 3.1 account or from converted
sidecar metadata must not be carried into this Llama 3.2 head contract.

## Measurement definitions

- `gamma` is the maximum linear draft-chain length offered to one verify round.
- Offered-draft acceptance is `accepted_drafts / drafted`.
- Emitted tokens per round is `(accepted_drafts + rounds) / rounds`; it includes
  the authoritative target token emitted by each round.
- `lossless=true` means the complete EAGLE token-id stream exactly matched the
  paired plain greedy target stream. A divergence fails the command.
- The receipt records 96 generated tokens. Throughput is decode-only and
  excludes the first target anchor from the timed token numerator.

These are single-host, single-prompt tuning measurements, not distributional
performance claims.

## Earlier CPU-target sweep

This prototype kept the learned head on Metal but deliberately used the generic
CPU target path for prompt processing and verification. All six rows used the
same exact target and head, a 38-token prose prompt, and a 96-token continuation.
All six produced the exact plain-target token stream.

The prototype binary SHA-256 was
`7dcd349abe54f3fb5f3b86e8211bc87e3c2414e70aad8aeae63aa8b5d0b5fdce`.
Its receipts report `commit=f43e16fbe7b3c7d892e722b9e9ce23b88bbc662f`, but that value was only the
dirty worktree's HEAD and does not identify the implemented source state.

| gamma | Plain tok/s | EAGLE-3 tok/s | Ratio | Offered acceptance | Emitted/round | Lossless |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 13.030 | 10.540 | 0.8089x | 54.10% | 1.532 | yes |
| 2 | 12.940 | 8.605 | 0.6650x | 41.35% | 1.827 | yes |
| 3 | 12.973 | 7.002 | 0.5398x | 30.20% | 1.900 | yes |
| 4 | 12.803 | 5.783 | 0.4517x | 23.71% | 1.939 | yes |
| 5 | 12.826 | 5.015 | 0.3910x | 19.92% | 1.979 | yes |
| 6 | 12.822 | 4.380 | 0.3416x | 16.73% | 1.979 | yes |

The narrow finding is that this matching head has real novel-prose acceptance,
but CPU target verification is unprofitable; gamma 1 is merely the least-bad
CPU arm. This sweep does not decide resident performance and is not an
acceptance A/B against the resident sweep, whose prompt has 46 tokens.

## Current resident gamma sweep

All rows below used a raw prose prompt encoded with BOS and without EOS, 46
prompt tokens, 96 generated tokens, the resident Metal target verifier, and the
resident Metal learned head. Every row has zero CPU verify rounds and is
lossless (`first_divergent_generated_token_index=-1`).

These are explicitly **tuning results from a dirty precommit binary**, SHA-256
`203b882536f0f357f758523a3328af90825e76f3210f3a1e1cde7d683b931c75`.
As with the CPU prototype, the embedded
`commit=f43e16fbe7b3c7d892e722b9e9ce23b88bbc662f` names only the dirty
worktree's HEAD. The source receipts are named
`llama32-3b-eagle3-resident-gamma{1..6}-prose96.jsonl` on the measurement host.

| gamma | Plain tok/s | EAGLE-3 tok/s | Ratio | Offered acceptance | Emitted/round | Resident / CPU verify rounds | Lossless |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 30.656 | 31.931 | 1.042x | 95.83% | 1.958 | 48 / 0 | yes |
| 2 | 31.136 | 25.543 | 0.820x | 57.47% | 2.136 | 44 / 0 | yes |
| **3** | **31.254** | **42.541** | **1.361x** | **87.18%** | **3.615** | **26 / 0** | **yes** |
| 4 | 31.207 | 39.767 | 1.274x | 65.38% | 3.615 | 26 / 0 | yes |
| 5 | 31.225 | 38.934 | 1.247x | 55.65% | 3.760 | 25 / 0 | yes |
| 6 | 31.229 | 37.059 | 1.187x | 46.62% | 3.760 | 25 / 0 | yes |

Gamma 3 was the tuning winner: `31.254 -> 42.541 tok/s`, or `1.361x`.
Its aggregate timing fields are 259.536 ms drafting, 1,743.795 ms verifying,
and 196.892 ms updating the head over 26 resident verify rounds; it also used
one resident normal step for the final no-successor case. The paired streams
were identical.

## Committed raw-prompt result and head-cache A/B

The final no-terminal-newline raw prompt was rerun from source commit
`9626148a149f2e3cace675df1b3ae145d897cf4b` with binary SHA-256
`dac2fb1cc83b2c63f2926b082ba8a6639c3da1804fa493523f26f024fd2be411`.
The exact input/rendered-prompt SHA-256 is
`ab0316e833f5891f3f9b7f537540c3879743350369be10cb2524d315a4599c08`.
It measured **30.681 -> 41.536 tok/s (`1.354x`)** at gamma 3, with 87.18%
offered-draft acceptance, 3.615 emitted tokens per round, 26 resident / 0 CPU
verify rounds, and an identical 96-token target stream.

The final benchmark also ran the authoritative-row optimization against its
full-compute control. Both modes produced exactly the same complete draft-token
array, EAGLE output array, and plain output array. The control measured 36.830
tok/s (`1.201x`); skipping unused Q/attention/FFN/lm-head work on intermediate
authoritative rows measured 41.536 tok/s (`1.354x`). This closes the functional
A/B for the K/V-only head-cache update, while leaving an external golden
head-logit oracle as follow-up.

One byte matters here. Reading the committed prose file literally includes its
terminal newline and changes the continuation: on that prompt variant the same
gamma-3 lane measured `30.723 -> 26.825 tok/s` (`0.873x`) with 43.09%
acceptance, still lossless. The harness therefore removes terminal newlines
before invoking the exact no-newline tuning prompt. The promoted prose/chat
receipts record effective prompt hashes, and the latest schema also records the
raw input hash. The 1.354x row is real and reproducible, but it is not a
distributional claim.

## Served-chat prompt sweep

`--chat` renders the supplied text through the same no-tools single-user path
used by `/v1/chat/completions`, then tokenizes the rendered control markers with
special-token parsing. The resulting prompt is 55 tokens. All six arms below
come from the final committed binary, use 96 generated tokens, run every verify
round on resident Metal, use zero CPU fallback, and exactly match plain greedy.

| gamma | Plain tok/s | EAGLE-3 tok/s | Ratio | Offered acceptance | Emitted/round | Resident / CPU rounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | **30.649** | **26.085** | **0.851x** | **59.32%** | **1.593** | **59 / 0** |
| 2 | 30.641 | 21.288 | 0.695x | 40.78% | 1.808 | 52 / 0 |
| 3 | 30.650 | 24.606 | 0.803x | 36.84% | 2.089 | 45 / 0 |
| 4 | 30.594 | 23.588 | 0.771x | 29.07% | 2.136 | 44 / 0 |
| 5 | 30.559 | 23.281 | 0.762x | 25.62% | 2.238 | 42 / 0 |
| 6 | 30.548 | 22.057 | 0.722x | 21.49% | 2.238 | 42 / 0 |

The head is functionally working on served chat and its 59.32% first-draft
acceptance is close to the head README's held-out first-position figure, but
the current linear top-1 runtime is not economically working there: gamma 1 is
the least-bad arm and remains a 15% regression. A representative chat speedup
needs a top-k/tree EAGLE implementation; the raw-prompt win must not be
presented as served-chat throughput. This conclusion is stronger than a generic
optimization hunch: the gamma-1 row spent 3,641.969 ms versus 3,099.581 ms
plain, of which 372.146 ms was head update and only 0.061 ms was recorded draft
submission. Even deleting both costs would reach only about 29.05 tok/s, still
below 30.65 plain. Admission can cap a loss, but only a wider accepted tree can
amortize the remaining verify cost enough to create a win.

## Context checks

The final context rows share source `758b565f`, binary SHA-256
`616b3094b242d3bdf4c2f55ca68312d86e96f88223433fc58b86e24523c728cc`,
the exact target, prompt bytes, host, generation length, and embedded execution
provenance. At 556 prompt tokens, raw gamma-3 EAGLE measured
`30.403 -> 26.649 tok/s` (`0.877x`) with 44.17% acceptance, while suffix gamma 7
measured `30.449 -> 27.873 tok/s` (`0.915x`) with 19.67% offered-draft
acceptance. Both exactly matched their paired 96-token plain streams and used
resident verification only. The older `7187a071` EAGLE row is retained as
history and measured the same acceptance and `0.875x`.

At the original 4,137-token agentic prompt, the same provenance build measured
gamma-3 EAGLE at `24.077 -> 23.014 tok/s` (`0.956x`), 55.14% acceptance, 2.639
emitted tokens per round, and 36 resident / 0 CPU rounds. Earlier binaries put
the same deterministic-acceptance workload at `1.015x` and `1.049x`, so the
defensible conclusion is break-even within run-level throughput variation, not
a stable speedup. This is outside the head's `T=1024` training distribution
(not outside a declared runtime cap) and is labelled as extrapolation. At that
depth, context-matching suffix drafting is substantially stronger.

## Reproduction

`qa/speculative/harness/run_eagle3.sh` runs the pinned resident sweep and fails
if the token streams diverge, resident verification does not engage, or any CPU
fallback occurs. `CHAT=1` selects served-chat rendering; the default is the
exact raw tuning prompt shape. The exact target/head paths and output directory
can be overridden with `MODEL`, `EAGLE3`, and `OUT_DIR`.

## Suffix comparison at depth

The canonical suffix row is the same `758b565f` provenance build and exact 4k
prompt as the EAGLE row above. At gamma 7 it measured
**`22.015 -> 43.910 tok/s` (`1.995x`)**, accepted 66 of 78 offered drafts
(`84.615%`), emitted 6.50 tokens per round, used 12 resident and 0 CPU verify
rounds, and exactly matched all 96 plain token ids. Its enriched receipt embeds
the binary/model/prompt/tokenizer hashes, prompt format, full plain/spec token
arrays, Metal device, host ISA, effective environment, planner updates, and
execution plan.

The suffix and EAGLE commands have separate paired plain arms, so their small
plain-rate difference is run noise; compare their speculative throughput and
within-command ratios. Older suffix observations agreed on the absolute depth
result but are superseded for provenance. No depth row is directly comparable
to the 38- or 46-token prose sweeps.

## Conclusion

This conclusion describes the original linear-head campaign. See the
wide-hybrid follow-up at the top of this document for the later dynamic-tree and
suffix-first results.

The exact learned EAGLE-3 head is loaded, resident, recurrent, and losslessly
verified. It exceeds plain decode on one exact raw completion (`1.354x`) and is
a run-variable near-break-even at the 4k extrapolation point (`0.956x` on the
canonical same-build row), and it regresses on the
served-chat sweep (`0.851x` best) and the 556-token code context (`0.877x`).
So the original implementation was functionally working, while broad MTP-style
acceleration was not finished. That result justified the bounded top-k/tree
EAGLE step now measured in the follow-up; cheaper head updates alone could not
recover the measured chat deficit. This
remains learned EAGLE-3 rather than a native target MTP head, and it is not
wired into serving.
