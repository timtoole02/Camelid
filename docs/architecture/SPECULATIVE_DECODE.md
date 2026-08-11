# Lossless Greedy Speculative Decoding

Default-off serving optimization (`camelid serve --spec-decode …`). It makes **no support
claim**: enabling it promotes no lane, changes no release-ledger row, and byte-parity for a
lane is asserted only by evidence (tests and parity receipts), never by resemblance.

## What it is

Decode is memory-bandwidth bound: every sequential token costs a full pass over the weights.
Speculation drafts k candidate tokens cheaply, then verifies them in **one batched forward**
through the target model (`LlamaInferenceSession::forward_greedy_verify_chunk`, which returns
the greedy argmax at every position of the batch). Accepted tokens are exactly the target's
own greedy choices; rejected drafts are discarded by KV rollback
(`rollback_to_position` — the KV cache is position-major, so rollback is a cursor reset) and
are never observable in any output.

Two drafters (`src/inference/speculative.rs`):

| Mode | Flag | Drafts come from | Cost of drafting |
|---|---|---|---|
| `ngram` | `--spec-decode ngram` | The most recent earlier occurrence of the current token suffix (prompt lookup) | ~zero |
| `draft` | `--spec-decode draft --spec-draft-model <gguf>` | A smaller model with the **identical token mapping** (enforced fail-closed at load) running greedy ahead | one small-model forward per drafted token |

`--spec-draft-tokens` overrides the per-round draft window (default 8 for ngram, 5 for
draft). Speculation engages only for plain greedy requests (temperature 0, no per-step
logit/dense diagnostics, non-streaming); everything else takes the unchanged vanilla path.
When the drafter proposes nothing (no n-gram match), the loop falls through to the plain
single-token decode step — a non-drafting round costs exactly what vanilla costs.

**A spec-enabled server runs the CPU execution plan** (the serve entry disables the
Metal-resident envs): speculative verification needs CPU-resident packed Q8 weights, and the
Metal-resident plan deliberately keeps CPU-side weights file-backed (the GPU owns the
resident copy), which makes verify rounds pay a file-speed weight pass each — the first
implementation measured 4 tok/s because of exactly this. Enabling speculation is therefore a
workload decision, not a free flag: it trades the Metal decode lane for batched CPU
verification.

## The losslessness invariant, and its proof

Every emitted token is the target model's own greedy argmax given the accepted prefix, so
speculative output must be byte-identical to vanilla greedy decode. Evidence (2026-06-05,
M4 16GB, commit of this doc):

- **Unit equivalence** (`api::tests`): ngram speculation and self-drafting model speculation
  both reproduce vanilla greedy token-for-token on synthetic weights, with multi-token
  acceptance exercised; KV rollback provably restores exact decode state.
- **Real-model byte-parity**: Llama 3.2 1B Q8_0, 63-token greedy generation — spec-on vs
  spec-off `generated_token_ids` identical. Llama 3.2 3B with a 1B draft model, 21 tokens —
  identical, with 19/19 drafted tokens accepted in 4 verify rounds (100% acceptance).
- **Receipts**: a parity receipt emitted from a speculation-enabled server (81 tokens
  generated at 76.9 tok/s on the repeat workload below) fully verified — `camelid
  verify-receipt` replays the request through the vanilla path in-process AND re-runs
  llama.cpp on the same GGUF, and both matched byte-for-byte
  (`first_divergent_token_index=-1`, `RECEIPT VERIFIED`). Receipts are the standing way to
  prove, per request, that speculation changed nothing.

Numerics caveat, stated honestly: byte-parity relies on the batched verify forward and the
sequential decode forward agreeing at every argmax. That holds in all evidence above but is
an empirical property of the kernels, not a theorem — which is exactly why the receipt
machinery, not this document, is the proof of any particular run.

## Measured performance envelope (why it is default-off)

Same host (M4 16GB), warm prompt cache, greedy, byte-parity confirmed for every cell.

**Llama 3.2 1B Q8_0, ngram mode (k=8, min n-gram 3):**

| Workload | Vanilla Metal (default) | Vanilla CPU | Speculative ngram |
|---|---:|---:|---:|
| Repeat-a-sentence ×8 (96 tok) | 70.6 | 44.3 | **76.9** (80% acceptance) |
| Number list (64 tok, no n-gram repeats) | 70.5 | 45.7 | 45.4 (fallback floor) |
| Freeform QA (64 tok, no matches) | 71.7 | 44.6 | 46.2 (0 spec rounds) |

ngram speculation **beats the best vanilla configuration on repetitive/structured output**
and floors at CPU-vanilla speed everywhere else (non-drafting rounds are plain decode
steps). The remaining gap to Metal on non-repetitive text is the CPU-plan trade described
above, not speculation overhead.

**Llama 3.2 3B Q8_0 target, 1B draft model (k=8):**

| Workload | Vanilla Metal | Vanilla CPU | Speculative draft |
|---|---:|---:|---:|
| Number list | 27.3 | 20.7 | 20.0 (98% acceptance) |
| Freeform QA | 27.1 | 20.7 | 17.4 (74% acceptance) |

Draft mode is correctness-complete but **not yet profitable**: even at 98% acceptance it
only reaches CPU-vanilla parity. Two scoped inefficiencies remain — the target's small-M
verify GEMM costs ~3× a single resident weight pass (the i8mm chunk kernels are tuned for
hundreds of rows), and the sequential 1B draft steps are not free. The draft session already
reuses accepted-draft KV entries across rounds (only the rejected tail rolls back and is
never re-fed).

History worth keeping: the first implementation measured 4.2 tok/s on the repeat workload
because the Metal-resident execution plan left CPU-side weights file-backed and every verify
round paid a file-speed weight pass. Auto-selecting the CPU repack plan for spec servers
recovered an 18× speedup with zero kernel changes.

## Path to widening the win

1. A small-M-efficient packed-Q8 GEMM for the verify chunk (weight-stationary across the
   batch; the current i8mm prefill kernels are tuned for ≥hundreds of rows), which is also
   what draft mode needs to clear CPU vanilla, or
2. a Metal batched-verify kernel (multi-token forward with per-position logits and GPU-side
   KV rollback) so speculation can ride the resident stack instead of trading it away.

Performance claims stay bounded to the measured rows above; the flag stays default-off.

## Support boundary

Enabling speculation never changes which rows are supported. A speculative generation on an
unsupported lane is still unsupported; a verified receipt from a speculative run proves that
one request only, exactly as receipts always do ([`RECEIPTS.md`](../reference/RECEIPTS.md)).
