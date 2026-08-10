# Parity Receipts

A parity receipt is a verifiable record of **one request**: the exact model file, the exact
input, and the exact tokens Camelid produced — packaged so that anyone, on their own machine,
can re-run it and check the result against llama.cpp.

## Two rules first

These rules govern everything below. If any copy, field, or log line ever seems to say more
than this, that is a bug.

1. **A receipt proves one request matched the reference — it is NOT a support promotion.**
   A green receipt for some prompt on some model does not make that model "supported."
   Support lives in the release ledger ([`README.md`](../../README.md),
   [`COMPATIBILITY.md`](../../COMPATIBILITY.md), [`STATUS.md`](./STATUS.md)) and is unchanged by
   receipts. A receipt's envelope is exactly one request on exactly one GGUF, byte-identified
   by hash. Support does not spread by resemblance.

2. **Receipts are only meaningful for deterministic runs.** Byte-for-byte reproducibility
   requires greedy decoding (`temperature: 0`, no top-p/top-k sampling). A receipt for a
   sampled run is stamped `"reproducible": false` and is never presented as verifiable; the
   verifier refuses to assert parity on it.

## What a receipt does and does not prove

A verified receipt proves, for that one request:

- the receipt body has not been altered (`receipt_id` is a SHA-256 over a canonical
  serialization of every other field);
- the GGUF you supplied is byte-identical to the one the receipt names (`lane.gguf_sha256`);
- a current Camelid build, replaying the exact request, reproduces the recorded prompt
  tokens, generated tokens, and text (Camelid is internally deterministic on this lane);
- llama.cpp (`llama-server`), given the same GGUF and the same request, produces the same
  tokens and text.

A receipt does **not** prove: anything about other prompts, other context lengths, other
quantizations, other models, performance, or general correctness. It does not change any row
of the support ledger.

### Optional: the execution-trace rollup

A receipt produced on the deterministic CPU lane (the server running with `--deterministic`)
also carries an **execution-trace rollup** — a single SHA-256 over the whole forward pass
(every layer's hidden state and the final logits, folded across every generated token). It
proves not just that the *output tokens* match, but that the *internal computation* re-runs
bit-for-bit: `verify-receipt` re-derives the digest from an independent run and checks it.
This is only meaningful because the deterministic lane is reduction-order-stable. The digest
is ISA-specific (the receipt records `host_isa`), so a verifier on a different CPU re-runs the
tokens but reports `SKIP execution-trace` rather than a false mismatch. Receipts emitted on the
default (non-deterministic, GPU) path carry no rollup and are unchanged.

## Verifying a receipt

```bash
camelid verify-receipt receipt.json \
    --gguf path/to/exact-model.Q8_0.gguf \
    --llama-server llama-server        # path or name in PATH
```

Steps run in order, each printed as a PASS/FAIL line:

1. **self-digest** — recompute `receipt_id` from the canonical body. Failure means the
   receipt was tampered with or is malformed; verification stops.
2. **reproducibility gate** — a `reproducible: false` receipt cannot be verified; only the
   digest and lane identity are checked, and the run exits with a distinct status (exit
   code 2), never printing "VERIFIED".
3. **lane identity** — SHA-256 of your `--gguf` must equal `lane.gguf_sha256`.
4. **Camelid re-run** — the request is replayed in-process through the same generation path
   the server uses; prompt tokens, generated tokens, and text must match the receipt.
5. **reference re-run** — a temporary `llama-server` is started on the same GGUF and fed the
   receipt's **exact prompt token ids** (so the comparison is pinned to the exact prompt the
   receipt claims); the continuation's tokens and text are compared with the same match
   semantics the parity harness uses (exact equality; first divergent index reported, `-1`
   meaning none). Cross-engine chat-template/tokenizer equivalence for the original messages
   is attested at emit time by the parity harness (`compared_against_reference: true`); the
   verifier does not re-derive it.

A receipt whose own parity block records a mismatch (`*_match: false`) is a **divergence
record**, not a parity claim — the verifier checks its digest and lane identity, says so,
and exits with a distinct status (exit code 3) instead of ever printing "VERIFIED".

The final line is `RECEIPT VERIFIED` (exit 0) only when steps 1, 3, 4, 5 all pass on a
reproducible receipt; otherwise `RECEIPT NOT VERIFIED` names the failing step (exit 1).

No llama.cpp installed? `--self-only` runs steps 1–4 and reports
`RECEIPT PARTIALLY VERIFIED`, which is exactly what it says: digest, identity, and Camelid's
own determinism — **not** full parity. `--reference-only` skips the Camelid re-run instead.

## Getting a receipt

Receipts are always an explicit opt-in; nothing attaches them silently.

- **Parity harness (primary):** `scripts/chat-parity-tinyllama.mjs --emit-receipt out.json`
  runs the live Camelid-vs-llama.cpp comparison and emits a receipt with the parity block
  filled (`compared_against_reference: true`). Sealing is delegated to
  `camelid seal-receipt` so canonical serialization lives in one implementation.
- **Server opt-in (convenience):** send `"camelid_receipt": true` on a non-streaming
  `POST /v1/chat/completions`. No reference runs there, so the receipt is emitted with
  `compared_against_reference: false` and `null` match fields — it is a *claim of output*
  for the verifier to check, never a fabricated comparison. Sampled requests get the honest
  `reproducible: false` stamp.
- **Frontend:** with the composer's "Receipt" toggle on, the next reply carries a receipt
  card (lane, reproducible badge, match fields, `receipt_id`) with download and
  copy-the-verify-command buttons.

## Schema (v1)

Schema identifier: `camelid.parity-receipt/v1`. Defined in `src/receipt/mod.rs`
(`ParityReceipt`); serialization round-trips losslessly.

```jsonc
{
  "schema": "camelid.parity-receipt/v1",
  "receipt_id": "<sha256 of the canonical body — every field below, key-sorted, compact>",
  "created_utc": "RFC 3339 timestamp",
  "lane": {
    "model_id": "tinyllama-q8",
    "gguf_sha256": "<sha256 of the exact GGUF file>",
    "gguf_filename": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    "quantization": "Q8_0",
    "architecture": "llama",
    "tokenizer_kind": "llama_spm",
    "tokenizer_sha256": "<sha256 of tokenizer.* GGUF metadata, or null>",
    "camelid_version": "<git describe, or crate version + commit>",
    "camelid_commit": "<git rev-parse HEAD at build time>"
  },
  "reference": {
    "tool": "llama.cpp",
    "binary": "llama-server",
    "version": "<reported build info, or null>",
    "commit": null
  },
  "request": {
    "endpoint": "/v1/chat/completions",
    "messages_or_prompt": [{ "role": "user", "content": "..." }],
    "max_tokens": 50,
    "temperature": 0.0,
    "top_p": null,
    "top_k": null,
    "seed": null,
    "stop": []
  },
  "reproducible": true,
  "result": {
    "prompt_token_ids": [1, 529, 29989],
    "generated_token_ids": [29907, 650],
    "generated_text": "...",
    "completion_tokens": 50,
    "finish_reason": "length"
  },
  "parity": {
    "compared_against_reference": true,
    "prompt_tokens_match": true,
    "generated_tokens_match": true,
    "generated_text_match": true,
    "first_divergent_token_index": -1
  }
}
```

Notes:

- All hashes are lowercase hex SHA-256.
- `parity` match fields are `null` (not fabricated) whenever no reference was live at emit
  time; the verifier fills the truth in by re-running.
- The canonical form for digesting is defined by Camelid's typed serialization (sorted keys,
  no insignificant whitespace, `receipt_id` excluded). Emitters in other languages must not
  digest their own serialization — they call `camelid seal-receipt`, which is the single
  sealing implementation.
- An optional `signature` block (detached signature over `receipt_id`) is reserved for a
  future decision and is simply absent in v1.
