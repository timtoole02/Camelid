# Attention checkpoint bundles

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For current support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../../STATUS.md).

`camelid.attention-checkpoints.v1` is a narrow parity schema for comparing Camelid attention internals against a future independently instrumented known-good CPU trace. It intentionally stores compact samples and summary statistics, not full tensors.

## Capture from Camelid diagnostics

Start from a chat parity diagnostics JSON that includes `camelid.dense.layers[*]`, then extract the focused layers:

```bash
node scripts/extract-attention-checkpoints.mjs \
  --input target/attention-layer2-trace-captures/base.json \
  --layers 0,2 \
  --json-out target/attention-layer2-focused-checkpoints.json
```

Layer 0 is the sanity anchor. Layer 2 is the current first attention probe target from the zero-delta sweep. If a capture budget allows a wider target set, include layers `0,2,20,18,21` in that order: layer 0 anchors early residual behavior, while layers 2/20/18/21 are the attention zero-delta probes that moved the known-good token rank most.

## Validate a bundle

```bash
node scripts/check-attention-checkpoints.mjs \
  --input target/attention-layer2-focused-checkpoints.json
```

The checker validates the focused schema and sampled internal consistency:

- schema name and non-empty unique layer list
- prompt token IDs and first generated-token trace position count
- dense metadata dimensions (`embedding_length`, attention heads, KV heads)
- q/k/v/o checkpoint shapes, lengths, finite stats, and sampled windows
- attention trace head dimension/scale, grouped-query KV head mapping, sampled position order, sampled score/probability/key/value fields, exact sampled q·k score reconstruction with q×k product windows, probability-sum/max-probability summaries, and exact sampled context reconstruction from the full value/probability row. For prompts longer than eight positions, Camelid samples the first four positions and the last four/current-tail positions so trace bundles include both prompt-prefix anchors and the active decode position.

This is not a parity proof by itself. It is a guardrail that ensures a Camelid or future reference capture is normalized enough to compare honestly.

## Compare two bundles

Once an instrumented reference emits the same schema, compare it against the Camelid bundle directly:

```bash
node scripts/compare-attention-checkpoints.mjs \
  --left target/attention-layer2-focused-checkpoints.json \
  --right target/reference-attention-layer2-focused-checkpoints.json \
  --json-out target/attention-layer2-reference-compare.json
```

The comparator checks prompt-token alignment, core dense metadata, layer indexes, q/k/v/o compact stats, sampled checkpoint windows, sampled attention-trace scores/probabilities/key/value/context/query values, q·k score/product reconstruction fields, and attention context reconstruction fields. Exact structural fields must match; numeric values use `--atol` / `--rtol` tolerances so independent CPU captures can tolerate small floating-point drift while still reporting the first sampled divergence path.

## Expected tensor widths for TinyLlama Q8_0

For the current TinyLlama Q8_0 target (`embedding_length=2048`, `attention_head_count=32`, `attention_head_count_kv=4`):

- `q.output` and `q.rope_output`: `[1, 2048]`
- `k.output` and `k.rope_output`: `[1, 256]`
- `v.output`: `[1, 256]`
- attention context / `o.input`: `[1, 2048]`
- `o.output`: `[1, 2048]`
- trace `head_dim`: `64`
- trace `scale`: `1/sqrt(64) = 0.125`

## Known-good reference status

The local stock Docker `llama-server` binary has tokenizer/chat/logit-adjacent behavior but no discovered per-layer q/k/v/o checkpoint endpoint. No local source checkout or `llama-cli`/`llama-server` build was found outside Docker, and this host currently lacks `cmake`/`ninja`, so building a temporary instrumented reference is blocked without adding build tooling. A known-good bundle still requires an independently instrumented CPU reference or another non-copy source that can emit these same sampled fields.
