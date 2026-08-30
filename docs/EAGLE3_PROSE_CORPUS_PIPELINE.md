# EAGLE-3 prose warm-start corpus contract

## Decision

Use a deterministic, target-generated prompt corpus rather than tuning on the Pitch canary or
copying a large conversational dataset. The first standard receipt is fixed at:

| Split | Technical / instructional | Code / system design | General | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,080 | 600 | 720 | 2,400 |
| Locked eval | 180 | 100 | 120 | 400 |
| Total | 1,260 (45%) | 700 (25%) | 840 (30%) | 2,800 |

The 50-record pilot has 40 train jobs and 10 eval jobs with a 46% / 24% / 30% integer-rounded
mix. It exists to validate rendering, exact-Q4 generation, feature alignment, storage, and MLX
training before the standard corpus consumes hours.

## Why the split happens before expansion

Random record-level splitting would leak the same template and near-identical matrix siblings
into evaluation. Here every catalog family owns a unique logical source and template, and the
catalog binds all three to one split before combinations are expanded. The audit fails if a
source ID, family ID, or template ID appears in both train and eval.

The local matrices are original MIT-licensed prompts. This keeps acquisition small and license
clear. It also avoids downloading a broad public corpus whose own train/test contamination and
license composition would require a separate provenance review. Public sources can be added in a
later schema version only with immutable revision, license, upstream record ID, and family-level
split metadata.

## Leak boundary

The protected Pitch prompt, its answer, links, distinctive requirements, and paraphrases are not
inputs to generation or model selection. The committed policy has a narrow lexical denylist and a
URL ban. The production command must additionally pass the exact protected file and expected
SHA-256. The audit compares normalized exact spans plus local content-word windows and records
only the protected hash.

This is a lexical and deterministic audit, not a semantic-proof claim. Human review of the
catalog remains required before the full run, and the Pitch prompt remains a final canary only.

## Exporter boundary observed during implementation

The in-progress exact-Q4 exporter schema accepts already-tokenized JSONL records containing only
`id`, `input_ids`, and `loss_mask`. Its feature store then writes target IDs, greedy argmax IDs,
auxiliary layer-input taps `[2,14,25]`, post-final-normalization hidden state, input embeddings,
and next-token embeddings.

The corpus pipeline does not invent missing tokenizer or generation behavior. It emits
`camelid-eagle3-corpus-job-v1`, whose two messages and exact greedy policy must be materialized by
the target lane. The raw exporter-input invariant is:

```text
raw_loss_mask[Q] = 1 iff input_ids[Q+1] belongs to the exact target-generated assistant continuation
```

This includes the last prompt row when it predicts the first assistant token. It excludes system
and user labels and the final input row. The finalized exporter shifts the mask exactly once:

```text
base_loss_mask[P] = raw_loss_mask[P+1]
labels[P] = input_ids[P+2]
```

That makes the mask describe the same teacher token as the EAGLE base row. The exporter zeros the
final base row; it now captures from an empty resident session and has no unavailable bootstrap
rows.

## Receipt chain

A training-worthy shard needs all of the following identifiers:

- corpus profile, job ID, source family, template, and JSONL payload SHA-256;
- exact protected-prompt SHA-256 used by the corpus audit;
- target GGUF SHA-256, chat-template identity, rendered prompt SHA-256, prompt IDs, generated IDs,
  and stop reason;
- exact exporter binary/source SHA-256 and feature-store manifest/payload hashes;
- warm-start head and fixed `d2t`/`t2d` mapping hashes;
- trainer source/config/state/output hashes and ordered locked-eval metrics.

Training loss is not the success criterion. Checkpoint selection uses the locked non-Pitch eval
suite, then the existing runtime harness must prove ordered target-token equality, no round-latency
regression, materially higher emitted tokens per round on diverse technical prose, and finally the
separate Pitch canary.

Implementation and commands live in `tools/eagle3_corpus/README.md`.
