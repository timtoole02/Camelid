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

It does not currently render a chat or generate the target completion. A target-authoritative
bridge must therefore perform these steps for each job:

1. Render the two messages with the pinned Llama 3.2 Instruct chat template and its generation
   prompt. Preserve the rendered-byte hash and prompt token IDs.
2. Greedily generate the assistant continuation with the pinned exact-Q4 target, at temperature
   zero and the job's `max_new_tokens`. Preserve generated IDs and stop reason. Do not use a
   reference answer, another teacher, or a prepared continuation.
3. Concatenate the exact prompt and generated IDs. In the raw exporter input, set
   `raw_loss_mask[Q]` to one exactly when `input_ids[Q+1]` is a generated assistant token; set
   every other row to zero. This includes the final prompt row that predicts the first assistant
   token.
4. Submit `{id,input_ids,loss_mask:raw_loss_mask}` to `export-eagle3-features`. The exporter
   shifts the mask exactly once into its finalized base-row contract:
   `base_loss_mask[P]=raw_loss_mask[P+1]`, matching `labels[P]=input_ids[P+2]`, and zeros the final
   base row. If feature capture was not fused into generation, one teacher-forced replay is
   legitimate solely to capture the exact target activations.

The bridge remains an explicit integration item. A corpus manifest reports
`required_not_performed`; it never implies that raw messages are already accepted by the current
exporter.

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
