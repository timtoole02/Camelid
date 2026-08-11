# Local Inference Conformance Suite

Local inference runtimes are routinely compared on throughput and memory, and
almost never on whether they compute the same thing. Quantized models can
produce different outputs across runtimes, runtime versions, and execution
configurations — silently. This suite measures that, for any set of runtimes,
by one ruler.

## What it measures

For one GGUF model and a fixed set of **raw greedy completion prompts**
(deliberately template-free, so every engine receives the identical character
stream and divergence isolates tokenization + numerics):

1. **Determinism** — the same request repeated N times against the same
   server: are the outputs identical? (Token-level when the runtime reports
   token ids, character-level otherwise.)
2. **Cross-runtime agreement** — pairwise first-divergence depth between every
   pair of probed runtimes on the exact same model bytes. Reported as a
   matrix; **no runtime is treated as ground truth**.
3. **Tokenizer agreement** — prompt token ids from each runtime's `/tokenize`
   endpoint, compared pairwise where exposed.
4. **Provability** — can the runtime emit a sealed, independently verifiable
   record of what it computed? Camelid emits
   [parity receipts](../../docs/reference/RECEIPTS.md) (`camelid_receipt: true`); the
   probe seals one and runs `camelid verify-receipt` on it. Runtimes without
   such a mechanism are recorded as having none — that is a finding, not a
   judgment.

Servers run **sequentially** (never concurrently) so memory pressure cannot
contaminate results.

## Running

```sh
node tools/conformance/run.mjs \
  --model /path/to/model.Q8_0.gguf \
  --camelid-bin /path/to/camelid \
  --llama-server brew=/path/to/llama-server \
  --llama-server pinned=/path/to/other/llama-server \
  --ollama \
  --max-tokens 64 --rounds 3 --out conformance-out
```

- `--llama-server label=path` is repeatable: probe as many llama.cpp builds as
  you like (cross-version agreement is itself a useful measurement).
- `--ollama` imports the same GGUF into the local Ollama daemon through a
  generated Modelfile and probes `/api/generate` with `raw: true`.
- `--receipt-reference /path/to/llama-server` upgrades the provability probe
  from self-digest verification to the full chain (self-digest → lane identity
  → in-process replay → independent reference re-run).
- `--skip camelid` probes only the external runtimes.

Outputs land in `--out`: `results.json` (schema
`camelid.conformance/v1`, includes every raw round) and `SCOREBOARD.md`
(rendered tables).

## Reading the results

- `-1` means full agreement over the compared span; any other number is the
  index of the first diverging token/character.
- Disagreement between two runtimes does **not** say which one is wrong — it
  says at most one of them can be right, and that without a verifiable record
  there is no way to know which. That asymmetry is the point of the
  provability column.
- Greedy decoding is used everywhere because it is the only setting with a
  defined correct answer per engine state; sampled outputs are not comparable.

## Caveats

- Engines that cannot load the same GGUF bytes (different weight formats) can
  be probed for determinism but not byte-level agreement; keep such
  comparisons out of the agreement matrix.
- Near-tie logits mean two numerically-honest engines can legitimately
  diverge deep into a generation; shallow divergence (early tokens, or the
  tokenizer matrix) is the strong signal.
- Wall-clock and throughput are deliberately not measured here. This suite is
  about what was computed, not how fast.
