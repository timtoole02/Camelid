# llama3_1_8b_instruct_q8_0 — parity/tolerance decision

The row sits at `runnable_exact_row_numerical_variance` with
`parity_audited: failed_exact_greedy_token_ids`, and its `next_step` requires
"a documented parity/tolerance decision before Verified or Supported promotion".
This is that decision, with the measurement it rests on.

## What fails

Of the three pinned probes in
`fixtures/phase2/llama3_1_8b_instruct_q8_0-tokenizer-b9632.json`, **two now match
the oracle exactly** and one does not:

| prompt | oracle | camelid | match |
| --- | --- | --- | --- |
| `The capital of France is` | ` a city of romance,` | identical | yes |
| `Once upon a time` | `, in a small village` | identical | yes |
| `Hello` | `, I am a ` | `, I am interested in` | **no** |

Divergence is at generated index 3, exactly where the 2026-08-11 capture put it.
(That capture tested one prompt and scored 0/1; this is 2/3.)

## What the divergence actually is

Teacher-forced on the oracle's own prefix `[128000, 9906, 11, 358, 1097]`, both
engines on the **same sha256-verified artifact** (`9da71c45…`) and the **same
oracle build** (llama.cpp `acd79d603` = b9632, self-reported `version: 9632`):

| token | oracle logprob | camelid logprob | delta |
| --- | --- | --- | --- |
| ` a` | -2.123934 | -2.173546 | +0.04961 |
| ` interested` | -2.200481 | -2.142938 | -0.05754 |
| ` looking` | -2.671263 | -2.659318 | -0.01194 |
| ` trying` | -3.038179 | -3.090789 | +0.05261 |
| ` an` | -3.725251 | -3.751602 | +0.02635 |
| ` writing` | -4.097141 | -4.167905 | +0.07076 |
| ` new` | -4.174589 | -4.210609 | +0.03602 |
| ` having` | -4.222599 | -4.252682 | +0.03008 |
| ` the` | -4.347295 | -4.418090 | +0.07079 |
| ` currently` | -4.371387 | -4.408416 | +0.03703 |

- **The top-10 token SET is identical**, and the ordering is identical except
  for the top-2 pair.
- **Max disagreement: 0.071 nats.**
- The separation being resolved is 0.077 nats (oracle) / 0.031 (camelid) — i.e.
  **the same order as the inter-engine noise**. Greedy argmax on a margin
  smaller than the numerical difference between two independent implementations
  is a coin-flip, not a correctness property.

## Why this is a tolerance case and not a bug

This codebase has a worked example of the opposite. In the qwen35 investigation
a genuine defect put the oracle's token at **rank 70, 8.4-19.3 nats out** — a
disagreement about *what the model believes*. Here the engines agree on what the
model believes to within 0.071 nats across the whole top-10 and disagree only on
which of two near-equal tokens wins. That is 100-270x smaller, at rank 2.

Ruled out along the way: Llama-3.1 delivers its rope scaling through the
`rope_freqs.weight` tensor (present, `[64]`) rather than `llama.rope.scaling.*`
metadata (absent — which is why the runtime reports `rope_scaling_type: none`
and looks alarming). The llama loader does find and apply that tensor.

## Decision

**Exact greedy token-ID equality is not an appropriate promotion gate for this
row.** The defensible gate is distributional: identical top-k token set and
agreement within a stated epsilon, with tiebreak differences permitted where the
top-2 margin is below that epsilon.

Proposed: **top-10 set equality and max |delta| <= 0.15 nats**, ~2x the observed
0.071 so it is a bound rather than a fitted value.

Under that gate this row passes on all three pinned probes.

## What this does NOT cover

- Only three prompts, all short. `tested_context` remains
  `short_prompt_oracle_pack_only` until the bounded-context packs run.
- Says nothing about Metal-lane parity; captured on the CPU reference lane
  (`CAMELID_LAZY_Q8_0_LINEAR=1`, needed to stay under the 6.44 GB
  materialization cap and to match the fixture's `no_repack` backend).
- Says nothing about K-quant 3.1, which has no ledger row at all.
