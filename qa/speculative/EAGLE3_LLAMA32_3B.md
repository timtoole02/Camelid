# Llama 3.2 3B EAGLE-3 benchmark

## Status and claim boundary

This is a benchmark-only implementation of a learned EAGLE-3 sidecar for one
exact Llama 3.2 3B target. It drafts a linear top-1 chain, then lets the target
model authoritatively verify every proposed token. It is **not** a native MTP
head in the target model, is not wired into serving, is not enabled by default,
and does not widen Camelid's supported-model claims.

The resident results below are tuning measurements from a dirty, precommit
binary. They establish that the lane can be lossless and faster on this one
prompt, but they are not a release or promotion receipt. **A final rerun from a
committed binary, with repository-owned receipts, is still pending.**

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

The benchmark is greedy and linear top-1 only. `gamma` (`--draft-tokens`) is the
maximum number of draft tokens proposed for one target verify round; it is a
runtime tuning knob, not part of the checkpoint architecture.

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

Gamma 3 is the current tuning winner: `31.254 -> 42.541 tok/s`, or `1.361x`.
Its aggregate timing fields are 259.536 ms drafting, 1,743.795 ms verifying,
and 196.892 ms updating the head over 26 resident verify rounds; it also used
one resident normal step for the final no-successor case. The paired streams
were identical. This row is promising, but the **final committed receipt is
pending**.

## Suffix baseline at depth

A legacy same-target suffix-drafter receipt used a 4,137-token agentic prompt,
96 generated tokens, and suffix gamma 7. It measured
`21.1072 -> 42.4001 tok/s` (`2.0088x`), accepted 66 of 78 offered drafts
(`84.615%`), emitted 6.50 tokens per round, used 12 resident and 0 CPU verify
rounds, and was lossless.

That result is useful as a depth baseline only. It is **not directly comparable**
to the 38- or 46-token learned-head prose sweeps: it uses a different drafter,
prompt, context depth, and measurement provenance. The legacy receipt records
`commit="unknown"` and no binary hash, so it cannot support a final same-binary
comparison or a claim that suffix and EAGLE-3 were fairly ranked here.

## Conclusion

The exact learned EAGLE-3 head can preserve the target's greedy output and, with
resident target verification, exceeded the paired plain target on this prompt.
The best current tuning point is gamma 3. The result remains bounded to the
exact artifacts and benchmark command above until a clean committed build is
rerun and its durable receipts are checked in; it is not native target MTP and
not a serving result.
