# Deterministic EAGLE-3 prose corpus

This tool builds a small, reproducible job corpus for exact-Q4 EAGLE-3 warm-start training. The
standard profile contains 2,400 training conversations and a separately assigned, locked set of
400 evaluation conversations. Its exact mix is 45% technical/instructional prose, 25% code and
system design, and 30% general prose. The pilot profile is a stratified 50-record subset for the
first exporter/trainer smoke.

The committed catalog contains original prompt matrices under Camelid's MIT license. It embeds
no third-party dataset text, protected canary, answer, repository link, or network dependency.
Every logical source, template, and family is assigned to either `train` or `eval` in the catalog
before Cartesian expansion. The audit rejects overlap at all three levels.

## What the JSONL means

`train.jobs.jsonl` and `eval.jobs.jsonl` use `camelid-eagle3-corpus-job-v1`. A record has exactly
two messages (`system`, then `user`), a target-greedy generation policy, assistant-only
supervision intent, full source/license metadata, and a content hash. It deliberately has no
assistant answer.

This is a pre-materialization format. The exact-Q4 exporter currently under development accepts
only:

```json
{"id":"...","input_ids":[128000,128006],"loss_mask":[0,0]}
```

Materialize it with Camelid's target-authoritative bridge. The bridge performs these steps for
each job:

1. Render the two messages with the pinned Llama 3.2 Instruct chat template and its generation
   prompt. Preserve the rendered-byte hash and prompt token IDs.
2. Greedily generate the assistant continuation with the pinned exact-Q4 target, at temperature
   zero and the job's `max_new_tokens`. Preserve generated IDs and stop reason. Do not use a
   reference answer, another teacher, or a prepared continuation.
3. Concatenate the exact prompt and generated IDs. In the raw exporter input, set
   `raw_loss_mask[i]` to one exactly when `input_ids[i]` is target-generated assistant content;
   prompt tokens and assistant framing/control tokens remain zero. This mask is canonical and
   token-aligned; the materializer must not pre-shift it. A final content token produced at the
   max-token limit remains one.
4. Submit `{id,input_ids,loss_mask:raw_loss_mask}` to `export-eagle3-features`. The exporter
   alone maps the mask into its finalized base-row contract:
   `base_loss_mask[P]=raw_loss_mask[P+2]`, matching `labels[P]=input_ids[P+2]`, and supplies two
   trailing zero sentinels. If feature capture was not fused into generation, one teacher-forced
   replay is legitimate solely to capture the exact target activations.

The corpus-build manifest reports `required_not_performed` because building jobs alone still does
not generate an answer. A successful materializer run produces a separate sealed run manifest,
atomic shards, per-record audit evidence, and `COMPLETE.json`.

## Exact-Q4 materialization

The hidden engineering command reuses Camelid's production tokenizer, metadata-Jinja renderer,
`LlamaInferenceSession`, and greedy generation loop. It loads only the pinned Q4 target; there is
no reference answer, alternate teacher, response cache, or second inference implementation.

```bash
camelid materialize-eagle3-corpus /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --target-sha256 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff \
  --jobs target/eagle3-corpus/pilot/train.jobs.jsonl \
  --jobs-sha256 <exact-lowercase-jobs-jsonl-sha256> \
  --output target/eagle3-materialized/pilot-train \
  --shard-size 64 \
  --max-shards 1
```

Resume the exact run after exporting/training/deleting the consumed feature shard:

```bash
camelid materialize-eagle3-corpus /models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --target-sha256 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff \
  --jobs target/eagle3-corpus/pilot/train.jobs.jsonl \
  --jobs-sha256 <same-jobs-sha256> \
  --output target/eagle3-materialized/pilot-train \
  --shard-size 64 \
  --max-shards 1 \
  --resume
```

Resume fails closed if the job bytes, target/tokenizer/template, Camelid binary, shard size, job
ordering, or any completed shard differs. A shard is committed by atomically renaming a complete
directory containing `records.jsonl`, `audit.jsonl`, and `manifest.json`; a crash can leave only a
scoped `.shard-*.tmp` directory, which the matching resume safely discards and regenerates.

Each `records.jsonl` line has exactly the exporter shape:

```json
{"id":"...","input_ids":[128000,128006],"loss_mask":[0,0]}
```

The corresponding audit line seals the rendered-prompt bytes, prompt/generated/combined token
IDs, canonical raw mask, source content, stop reason, and exporter record. Prompt token IDs are
the prefix of `input_ids`; generated IDs are the suffix beginning at `assistant_generation_start`.
`assistant_content_start` and `assistant_content_end_exclusive` bound the actual trainable content.

## Build

Write only to a new or empty directory. The tool refuses to mix a run with stale output.

```bash
python3 tools/eagle3_corpus/build_corpus.py build \
  --profile pilot \
  --output target/eagle3-corpus/pilot

python3 tools/eagle3_corpus/build_corpus.py build \
  --profile standard \
  --output target/eagle3-corpus/standard
```

The output contains the two JSONL files, `manifest.json`, and `SHA256SUMS`. The manifest seals the
generator, catalog, leakage policy, schema, ordered record digest, payload hashes, counts,
license, family assignments, split method, and audit result. It contains no absolute source or
canary path.

On a storage-constrained validation host, seal a deterministic 100-record subset of the already
locked evaluation split before training. Allocation is proportional by category and then by
held-out family, with no randomness; the output manifest pins the full eval hash, source manifest
hash, exact quotas, selected-ID digest, and protected-reference hash:

```bash
python3 -m tools.eagle3_corpus.select_eval_subset \
  --corpus-dir target/eagle3-corpus/standard-protected \
  --output target/eagle3-corpus/standard-eval-100 \
  --count 100
```

## Protected-canary audit

The default policy rejects URLs and curated distinctive terms. A production build must also
audit the exact protected prompt supplied from outside the repository. Pin the expected file hash
so a renamed or edited prompt cannot silently weaken the gate:

```bash
python3 tools/eagle3_corpus/build_corpus.py build \
  --profile standard \
  --output target/eagle3-corpus/standard-protected \
  --forbidden-file /secure/path/to/protected-prompt.txt \
  --expected-forbidden-sha256 7c7319ab066e8e6f5bb83a813082db1ee117aacd88405a36616b7e54dd0de6a2
```

The audit rejects curated patterns, an exact normalized eight-word span, or high content-word
overlap with any similarly sized protected-prompt window. It records only the protected file's
SHA-256, never its path or text. Keep the protected prompt out of training, evaluation, tuning,
checkpoint selection, and intermediate corpus artifacts; use it only as a final runtime canary.

Re-audit an existing corpus, optionally against the protected file:

```bash
python3 tools/eagle3_corpus/build_corpus.py audit \
  --corpus-dir target/eagle3-corpus/standard-protected \
  --forbidden-file /secure/path/to/protected-prompt.txt \
  --expected-forbidden-sha256 7c7319ab066e8e6f5bb83a813082db1ee117aacd88405a36616b7e54dd0de6a2
```

An audit without `--forbidden-file` is honestly labeled `not_supplied_policy_only`; it is not a
full canary-leakage receipt.

## Streaming and locked evaluation

The JSONL is only about 3 MiB, but exact-Q4 activation records are large. Read one job (or one
small shard) at a time, exit the target process after export, train that shard, validate the new
state, and delete only the consumed feature shard. Do not buffer all target features.

Seal the 400-job evaluation JSONL before any training. Its source/template/family assignments and
payload hash must not change between checkpoint comparisons. Evaluation prompts can be captured
in bounded shards if disk is tight, but their job IDs, generation policy, target/model hash, and
ordered result aggregation must stay fixed.

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tools/eagle3_corpus/tests -v
python3 -m py_compile tools/eagle3_corpus/*.py tools/eagle3_corpus/tests/*.py
```

The tests cover exact profile counts and ratios, byte reproducibility, duplicate rejection,
source/template/family isolation, curated and exact protected-text leakage, length bounds,
payload tampering, the 50-record pilot, and the exporter bridge contract.
