# Llama 3.2 3B on mini2 — wide hybrid campaign (2026-08-29)

## Verdict

The 100 tok/s stretch target was cleared, then doubled, on the exact pinned
Llama-3.2-3B-Instruct Q4_K_M target:

- **201.67 tok/s mean** over three lossless runs at a 115-token recurring
  structured-pattern context (`201.62..201.70`).
- **161.58 tok/s** at 1,819 prompt tokens after batching width-16 attention.
- **56.09 tok/s** on the favorable short raw learned-EAGLE gate.
- **52.88 tok/s** on a mixed JSONL workload that exercised both suffix and
  learned-tree fallback.
- **42.47 tok/s** on the 1,078-token agentic learned-tree pack.

The 201/161 rows are real decode throughput, but they are a recurrence ceiling,
not a general-chat claim. They require 100% acceptance from the model-free
suffix lane. Ordinary output is limited by the matching EAGLE head's acceptance;
serving promotion and a broader chat distribution remain open.

All accepted rows are target-authoritative. Every reported run has
`lossless=true` and `first_divergent_generated_token_index=-1`.

## Exact artifacts and binary

Target:

```text
~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
sha256 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff
```

Learned sidecar (EAGLE-3, not native MTP):

```text
~/models/Llama-3.2-3B-Instruct-Eagle3
revision 02d343789b502a3edfe351bdd4537a44affb98cd
model.safetensors sha256 c0713251464a9b6b5fcf9fb229587bbe59b6fd1521027aef32101d11b9ebbdaf
```

Ceiling, raw, mixed, and agentic receipts (115-token ceiling and learned/mixed
rows):

```text
source aeeacfa1fba844fd8f5a24d214ec870ca1cb2353
version camelid v0.6.1-308-gaeeacfa1
binary ~/camelid-overnight/bin/camelid-wild-aeeacfa1-e509fc61
sha256 e509fc61d602a7a467ffe44b0b0d97d3dc1e779034b2f37e2a4566566cd990a0
```

Deepest measured binary (1,819-token batched-attention row):

```text
source ee33ea68f4b37022dbf3aafc16d0ec34ca674b3d
version camelid v0.6.1-309-gee33ea68
binary ~/camelid-overnight/bin/camelid-wildk16-ee33ea68-9e340844
sha256 9e34084423e16c212746dc9666a151c860ed378a650825494bf81d06efa6bb4d
```

The branch now also contains `a7f8db6d`, which fail-closes width-16 batched
attention above position 2,048 and disallows wide V4 in the standard resident
Llama prompt-prefill path exercised here. That safety-only follow-up passed
`cargo check --lib`; it was not
deployed because mini2 entered its local-login lock after the unreceipted 4k
stress attempt.

## Measured matrix

All rows generate 96 total tokens including the first target anchor (95 timed
decode tokens). `plain` is the same binary and V4 arithmetic universe.

| Workload / lane | prompt tokens | plain tok/s | hybrid tok/s | emitted/round | result |
| --- | ---: | ---: | ---: | ---: | --- |
| recurring pattern, width 16, run 1 | 115 | 22.13 | 201.70 | 15.83 | lossless |
| recurring pattern, width 16, run 2 | 115 | — | 201.62 | 15.83 | lossless |
| recurring pattern, width 16, run 3 | 115 | — | 201.70 | 15.83 | lossless |
| recurring pattern, width 16 + batched attention | 1,819 | 21.12 | 161.58 | 15.83 | lossless |
| recurring pattern, width 8 + lazy head | 115 | 22.20 | 167.71 | 7.92 | lossless |
| JSONL cycle, suffix then EAGLE tree | 485 | 21.96 | 52.88 | 3.52 | lossless |
| favorable raw, linear EAGLE gamma 3 | 46 | 22.24 | 56.09 | 3.48 | lossless |
| agentic pack, EAGLE tree N8/K4/X4 | 1,078 | 21.78 | 42.47 | 3.06 | lossless |
| favorable raw, EAGLE tree N16/K4/X5 | 46 | 22.16 | 37.31 | 3.80 | lossless, rejected for speed |

The three width-16 ceiling runs are exceptionally stable:

```text
mean   201.674466 tok/s
median 201.701529 tok/s
min    201.618198 tok/s
max    201.703670 tok/s
```

Their plain/spec streams are identical within each run, and all three spec
streams are identical to one another.

At 1,819 tokens, admitting width-16 rows to packed attention changed:

```text
133.0000 -> 161.5794 tok/s  (+21.49%)
713.755  -> 587.341 ms verify (-17.71%)
```

Plain, spec, and drafted token arrays remained identical to the pre-batch
control. All six verifier rounds traced `attn_batch_layers=28`.

## What changed

### 1. Lazy learned-head maintenance

Suffix proposals do not read the EAGLE head. Target captures and emitted token
ids are buffered while suffix keeps winning. Before the first dynamic-tree
fallback, the buffer is applied in chronological order with the existing
authoritative batch update. If generation ends on an all-suffix streak, the
pending head work is unobservable and is discarded.

On the 115-token recurrence receipt this removed all 124-125 ms of learned-head
maintenance. Buffering 95 authoritative rows cost 0.483 ms, with zero catchups
and 95 deliberately discarded rows. Throughput rose from a stable 131.48 tok/s
baseline to 167.71 tok/s after the combined changes.

The mixed JSONL gate is the important correctness control: it performed three
catchups covering 13 rows, left a final 7-row streak pending, and reproduced
the old plain/spec/draft arrays exactly.

### 2. Row-dimensional F16 split-K attention

The old verifier encoded a partial and merge dispatch per row per layer. At
k=8 and 28 layers that is 448 dispatches. The new kernel moves verifier row into
the Metal grid and emits 56 dispatches while retaining each row's position,
split count, and packed tree ancestry.

The focused staged/direct A/B is bit-identical for linear and branching layouts
at both k=8 and k=16. The 4k k=8 attention-only probe improved from 70.672 to
41.752 ms (1.693x). Whole-model gains were 3.93% on the 1,078-token learned
tree and 21.49% on the 1,819-token width-16 suffix run.

### 3. Width-16 combined-chain V4

Widths 9 through 16 use two SIMDgroups. They share one raw-weight decode and
each owns an independent N=8 output tile. Widths 1 through 8 retain the exact
original pipeline, K_PAD=8, and 32-thread geometry.

Q4 and Q6 match repeated original V4 output bits at every width from 2 through
16. Production traces on mini2 proved Q4/Q6 wide dispatches at `n_tokens=16`.
The decisive local Q6 head shape measured about 7.36 ms at k=16 versus 4.71 ms
for the established narrow k=8 path, so doubling verified rows costs roughly
1.56x rather than 2x.

Width 16 is not useful for the learned tree with this head: emitted tokens rose
only from roughly 3.65 to 3.80 per round while verify and draft work grew. Keep
N8/K4/X4 for learned fallback; reserve N16 for high-confidence suffix chains.

## Honest interpretation

- **201.67 tok/s is a ceiling workload.** The prompt deliberately establishes
  an exact four-token cycle and the suffix lane accepts every proposal.
- **161.58 tok/s shows the mechanism survives depth.** It is still a recurrence
  workload, now at 1,819 prompt tokens.
- **42-56 tok/s is the measured learned/mixed range** across the favorable raw,
  mixed JSONL, and 1,078-token agentic prompts. The two shorter favorable/mixed
  prompts are 52-56 tok/s.
- The pinned EAGLE artifact declares training sequence length 1,024. The
  1,819-token recurrence does not make a learned-head quality claim because
  the learned head never drafts in those all-suffix rounds.
- V4 is a new Q6 half-rounded arithmetic universe. Losslessness is proven
  against the same binary's V4 plain target, not against the established V3
  production stream. Promotion still needs model-quality sign-off.
- This is learned EAGLE speculation plus model-free suffix drafting. It is not
  a native target MTP implementation; no usable pinned native 3B MTP/Medusa
  weights were found.

## 4k safety gate

The first width-16 + packed-attention 4k recurrence stress run produced no
receipt. Afterward, new SSH connections reported `This system is locked`,
consistent with a reboot/local-login lock. No 4k speed claim follows.

Two safeguards landed immediately:

1. width-16 batched attention fails closed above position 2,048; k<=8 retains
   its separate 4k attention-only identity/performance probe;
2. the standard resident Llama prompt prefill used by this benchmark explicitly
   disallows V4 widths 9..16, preventing its ordinary 16-row chunks from entering
   the verifier experiment. This is not a global guard over unrelated windowed
   or architecture-specific prefill implementations.

Do not rerun the 4k width-16 stress case until mini2 is locally unlocked and
the reboot/lock cause is inspected.

## Receipt inventory on mini2

The receipts are durable in `~/camelid-overnight/receipts/` but still need to
be copied into this repository after mini2 is unlocked.

| receipt | sha256 |
| --- | --- |
| `linear-aeeacfa1-v4-q8-raw-gamma3-route-r1.jsonl` | `3c6b05d2d8a5777d6c2c3ad5fd48464d9c956bf4b692d77145d91f7377d8ea0f` |
| `dynamic-aeeacfa1-v4-q8-agentic1024-n8-k4-x4-d4-batchattn-r1.jsonl` | `83f49769cdf49154c6c7f1427ac568339b6af2c56e6ba144944be8dd93ca04da` |
| `hybrid-aeeacfa1-v4-q8-jsonl-cycle-n8-d7-lazy-r1.jsonl` | `515883177a582cb60ea74cdf3436d5f17386287b68c8dd63510ef9bfc1ed5290` |
| `hybrid-aeeacfa1-v4w16-q8-recurrence-n16-d15-lazy-r1.jsonl` | `b515731f1f6b8bdec40f7a8e05b06fb8ffa52627e6f4da8d974cc6e7c8653412` |
| `hybrid-aeeacfa1-v4w16-q8-recurrence-n16-d15-lazy-r2.jsonl` | `31a3fd6227b95853da6fdb525708516f0caafefe6a0aedde6d70a92f3a74bac5` |
| `hybrid-aeeacfa1-v4w16-q8-recurrence-n16-d15-lazy-r3.jsonl` | `3e380d5f5a3a2f6c27b004fb7faf2895f033cd09a0327a58e59e3ee442ad385d` |
| width-8 lazy recurrence control (locate by hash after unlock) | `0ead2c660d0b3bce019a470a014601bd3427cc3fa0bd0e90ee58e86285e728a6` |
| width-16 1.8k pre-batch control (locate by hash after unlock) | `cfc58d45e4adf0c124a7982c49bdb76074b0e43ced2be782825027801c8f633c` |
| `hybrid-ee33ea68-v4w16-q8-recurrence-1p8k-n16-d15-lazy-batchattn-r1.jsonl` | `fcf4e82a9299ba02fcc9344780ea9a6e18d1dc4026a7da72344a92ca81556477` |

The rejected 37.31 tok/s N16 learned-tree arm and the 131.48 tok/s pre-combined
recurrence baseline were notebook controls, not promoted receipt rows; do not
use either as independently receipted evidence.

Prompt SHA-256 values:

```text
115-token recurrence  fb4e3485d0039e91d887e12ebe619ec47ccc44b938c2bfaa1bddb781173b661f
1,819-token recurrence 3b56720198a95da2ad5db4ba402c506e53b48b91a737635660154aaa385f4094
485-token JSONL        1eb10708fc56788374d22b027b9d9c66dc27426428fd9070b750d25b405407e7
```

## Reproduction environment

```text
CAMELID_EAGLE3_LM_HEAD_Q8=1
CAMELID_METAL_LINEAR=1
CAMELID_METAL_Q8=1
CAMELID_METAL_RESIDENT_DECODE=1
CAMELID_METAL_RESIDENT_PREFILL=1
CAMELID_METAL_ATTN2=1
CAMELID_METAL_ATTN_BATCH_K=1
CAMELID_METAL_KV_DTYPE=f16
CAMELID_METAL_WIRE=1
CAMELID_METAL_WIRE_NSG8=1
CAMELID_METAL_F32Y=1
CAMELID_METAL_NOCOPY=1
CAMELID_METAL_KQUANT=1
CAMELID_KQUANT_V2=1
CAMELID_KQUANT_V3=1
CAMELID_KQUANT_V4=1
CAMELID_KQUANT_MMA=1
CAMELID_SPEC_TREE=1
CAMELID_NO_OPEN=1
```

Ceiling command shape:

```text
camelid bench-eagle3 TARGET.gguf --eagle3 HEAD_DIR \
  --draft-tokens 15 --tree-nodes 16 --tree-topk 4 \
  --tree-expansions 4 --suffix-first --max-tokens 96 \
  --prompt "$RECURRENCE_PROMPT"
```

`qa/speculative/harness/run_llama32_3b_wild.sh` constructs the exact prompt,
starts from a clean environment, bounds the run to the measured 2..450-cycle /
16..96-output-token envelope, writes receipts atomically, and rejects any run
that is not lossless, resident-only, suffix-only, and 100% accepted. Set
`EXPECTED_BINARY_SHA256` to pin a binary. `TRACE=1` additionally emits the
q4/q6-wide and verifier-attention ownership diagnostics on stderr; treat a trace
run as route evidence rather than a clean timing sample.

Learned fallback command shape:

```text
camelid bench-eagle3 TARGET.gguf --eagle3 HEAD_DIR \
  --draft-tokens 4 --tree-nodes 8 --tree-topk 4 \
  --tree-expansions 4 --max-tokens 96 \
  --prompt-file qa/speculative/harness/prompts/ag_1024.txt
```
