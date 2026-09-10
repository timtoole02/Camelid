# CAMELID_METAL_KQUANT_ATTN_MM: what it buys, what it costs

`results.txt` has both halves; `host.txt` the machine. `preflight.sh` produces the
per-stage GPU-busy attribution, `identity.sh` the greedy-output comparison.

This flag had **no measurement on any model but 1B**, and appeared in zero documentation.
It is off by default. The obvious reading — a fast lane left opt-in, like
`CAMELID_METAL_KQUANT_MM` was before it was promoted — is wrong, and this bundle is the
evidence for why.

## What it buys

2.18x off total prefill on a 2351-token prompt with 3B-Q4_K_M, from 14.5x on the
attention stage alone. Every other stage matches within 1 ms, so the attribution is not
in doubt.

## What it costs

`kquant_mm_prefill_enabled` was opt-in because there was no second host to qualify it on;
once there was, it was promoted, and its doc records what the delay cost every K-quant
user. **This flag is not that.** It is opt-in because attention-as-matmul stages K/Q
scores as half, and on a K-quant model's flatter logits that moves greedy output.

The existing note in `metal.rs` measured that on 1B only. On 3B, across five prompt shapes
run twice: four token-identical, one divergent, reproducing byte-for-byte. So the concern
generalises — occasional and content-dependent, not absent, and deterministic rather than
flaky. Prose diverged where random filler did not; prose at 1171 prompt tokens did not
diverge where prose at 595 did, so it is not monotone in length either.

## Why this PR does not flip the default

Because the trade is real and belongs to whoever is running the model. Someone who armed
the K-quant MM lane for its lossless part should not silently receive a lane that
sometimes answers differently. What was actually wrong was that a 2.18x prefill lever was
undiscoverable and unmeasured, and that the predicate carried a stale comment claiming
work was still needed that had already shipped.

So: measure it, document it, correct the comment, leave the choice.

## Reproducing

    bash preflight.sh    # per-stage GPU-busy, both arms
    bash identity.sh     # greedy output comparison, both arms

Both start one server per arm because `kquant_attn_mm_prefill_enabled()` latches its env
read in a `OnceLock`. `identity.sh` uses a binary carrying PR #739 so a partial
prefix-cache hit cannot confound the comparison.
