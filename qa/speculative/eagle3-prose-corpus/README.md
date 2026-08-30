# EAGLE-3 prose corpus reference manifests

These are deterministic reference manifests for the `pilot` and `standard` corpus profiles. The
generated JSONL payloads are intentionally not committed; their exact byte lengths and SHA-256
values are sealed in each profile manifest and `SHA256SUMS`.

Regenerate into ignored scratch space:

```bash
python3 tools/eagle3_corpus/build_corpus.py build \
  --profile pilot \
  --output target/eagle3-corpus/pilot

python3 tools/eagle3_corpus/build_corpus.py build \
  --profile standard \
  --output target/eagle3-corpus/standard
```

The committed manifests have `forbidden_reference.status=not_supplied_policy_only`: their prompt
matrices pass the versioned lexical and URL gates, and the policy pins the protected prompt hash,
but these reference generations did not have the protected file on the local host. Before target
generation or training, rebuild with `--forbidden-file`; the command automatically rejects any
file whose SHA-256 differs from the versioned pin.

The reference manifests do not claim target completion generation, tokenization, feature export,
training, quality improvement, or runtime throughput.
