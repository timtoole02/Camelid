# Local Inference Conformance

Local inference engines are compared on speed and memory, almost never on
whether they compute the same thing. Quantized models can produce different
text across runtimes, runtime versions, and configurations — silently, with no
way for a user to notice or a developer to prove which output is faithful to
the model.

This page is about closing that gap. Two pieces:

- **Parity receipts** — a per-request, sealed, independently re-verifiable
  record of what was computed. See [`../RECEIPTS.md`](./reference/RECEIPTS.md).
- **The conformance suite** — `tools/conformance/run.mjs`, which measures any
  set of runtimes by one ruler. See
  [`../tools/conformance/README.md`](../tools/conformance/README.md).

The point is not to crown a winner. It is to make correctness measurable and
provable at all, on the same model bytes — and to be honest about where the
measurement finds nothing wrong.

## What the suite measures

For one GGUF model and a fixed set of template-free greedy completion prompts
(so every engine sees the identical character stream and divergence isolates
tokenization + numerics):

1. **Determinism** — the same request, repeated: identical output?
2. **Cross-runtime agreement** — pairwise first-divergence depth on the exact
   same model bytes. No runtime is treated as ground truth.
3. **Tokenizer agreement** — prompt token ids from each `/tokenize` endpoint.
4. **Provability** — can the runtime emit a sealed, independently verifiable
   record of what it computed?

## Findings to date

Reproducible with the commands below; raw `results.json` for each run carries
every round. These are same-machine snapshots (Apple M4, 16 GB), not durable
or general claims.

### TinyLlama 1.1B Chat Q8_0 (SentencePiece tokenizer)

Runtimes: camelid · llama.cpp (Homebrew b9430) · llama.cpp (pinned 5d56eff) ·
Ollama 0.30.6. All four were individually deterministic across three rounds.

- **A popular runtime diverges from the rest at character 2.** On the "why is
  the sky blue" prompt, Ollama's raw-mode completion forks from all three
  llama.cpp-family engines almost immediately — materially different content,
  same model bytes. Silent to every user of that path.
- **The two llama.cpp builds agree with each other** on generation and
  tokenization across all prompts.
- **camelid matches llama.cpp on generation but their `/tokenize` endpoints
  disagree** on prompts with no leading space: camelid's `/tokenize` prepends
  the SentencePiece leading-space prefix where llama.cpp's bare endpoint does
  not. With `add_special` on (the generation path) both prepend it, so
  generations agree; only the bare endpoints differ — it never reaches output.
  Notably this does **not** reproduce on the Mistral-7B board below (also
  SentencePiece, full `/tokenize` agreement), so it is a tokenizer-config
  difference in how the bare endpoint exposes the prefix, **not** a blanket
  SentencePiece-family property. Recorded as a finding — including against
  ourselves — rather than silently conformed to one side.
- **camelid emitted a sealed receipt that verified through the full chain**
  (self-digest → lane identity → in-process replay → independent llama.cpp
  reference re-run), `first_divergent_token_index = -1`. No other runtime emits
  a comparable record.

### Llama 3.2 3B Instruct Q8_0 (BPE tokenizer)

Runtimes probed: camelid · llama.cpp (Homebrew b9430). (Ollama and a second
llama.cpp build were recorded unavailable on this host — a near-full disk and a
memory-pressure restart respectively; the suite records such cases as findings
and still scores the rest.)

- **camelid and llama.cpp agree completely** — generation token ids and
  `/tokenize` output both identical (`-1`) across every prompt — byte-for-byte
  equivalent.
- **camelid's receipt verified through the full chain** against the independent
  reference.

### Mistral 7B Instruct v0.3 Q8_0 (SentencePiece tokenizer)

Runtimes probed: camelid · llama.cpp (Homebrew b9430) · llama.cpp (pinned
5d56eff). All three were individually deterministic.

- **The `/tokenize` endpoints agree fully** (`-1`, every prompt) — camelid and
  both llama.cpp builds. This is what refined the TinyLlama finding above: the
  bare-endpoint leading-space difference is tokenizer-config-specific, not a
  SentencePiece-family rule.
- **Generation diverges only on near-ties.** camelid matches llama.cpp exactly
  on the open-ended prompt, and forks on an adversarial repetition prompt at
  token 3 ("…fox **jump** over…" vs "…fox **jumps** over…") and on the very last
  token of the list prompt. camelid's Mistral row is independently proven
  GPU-decode ≡ CPU-decode greedy-exact, so these are the two *implementations'*
  tiny numeric differences flipping a near-tied argmax — the same class as
  documented deep near-ties, not a correctness defect in either.
- **camelid's 7B receipt verified through the full chain** on a 16 GB host
  (`first_divergent_token_index = -1`) — see the verifier note below; large
  receipts are why it had to learn to verify within one model's memory
  footprint.

The contrast across the three models is the useful result: the suite isolates
disagreement to a specific tokenizer config and a specific endpoint contract,
and confirms exact agreement where none of that applies — with a receipt
proving camelid's side either way.

## Reproducing

```sh
node tools/conformance/run.mjs \
  --model /path/to/model.Q8_0.gguf \
  --camelid-bin /path/to/camelid \
  --llama-server brew=/path/to/llama-server \
  --ollama \
  --receipt-reference /path/to/reference/llama-server \
  --max-tokens 48 --rounds 3 --out conformance-out
```

`--llama-server label=path` is repeatable; `--receipt-reference` upgrades the
provability probe from self-checks to the full independent-reference chain.
Outputs: `results.json` (schema `camelid.conformance/v1`) and `SCOREBOARD.md`.

## Verifying big receipts on small hardware

A receipt is only as useful as the hardware it can be checked on. The full
chain re-runs the model twice — Camelid's own replay and a spawned
`llama-server` — so a 7B Q8 receipt (~7.7 GB each) would ask a host to hold
~15 GB if both load at once, which OOM-kills one of them on a 16 GB Mac. The
Mistral-7B board surfaced exactly this against our own tooling.

`camelid verify-receipt` now runs Full mode as **two isolated subprocess
passes**, each loading exactly one model and fully exiting — so the OS reclaims
its entire footprint — before the next starts. At most one model is ever
resident. Three details make it fit on a 16 GB host:

- The **Camelid pass goes first**, while memory is cleanest, and loads its
  weights as **memory-mapped wire pages** (`CAMELID_METAL_NOCOPY`) — reclaimable,
  file-backed page cache rather than ~7.7 GB of anonymous heap that must fit.
- The **reference pass loads second**, mapping the same file (reusing the cache).
- The reference's context is **sized to the receipt's own length** (a receipt
  records one bounded generation), so its KV-cache working set stays tiny
  instead of using a fixed large default.

Verified end to end on a 16 GB Mac with no special flags: a Mistral-7B receipt
passes the full chain (`RECEIPT VERIFIED`, `first_divergent_token_index = -1`),
and the smaller lanes verify with no regression. Each model load stays within
one model's memory footprint, so the proof is checkable on the hardware people
actually own.

## What this is not

- Not a throughput benchmark. Speed lives in
  [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md).
- Not a support-matrix promotion. A receipt verifies one request; it does not
  change the release ledger.
- Not a claim that disagreement means a named runtime is "wrong" — only that at
  most one output can be faithful to the model, and that without a verifiable
  record there is no way to know which. Closing *that* gap is the entire point.
