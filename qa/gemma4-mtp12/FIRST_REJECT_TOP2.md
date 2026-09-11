# First-rejection top2 feasibility

The existing draft assistant's runner-up could recover 39.8% of first
rejections on the saved full-chat query set. Giving each such recovery one
additional verified token would increase tokens per round by at most 9.93%
before accounting for extra GPU work. This evidence does not support a 20%
throughput gain from a single extra alternative alone.

The study reconstructed the first rejected draft of every round from the
saved query headers and the target's actual emitted token IDs. All 233 round
boundaries, position advances and per-request accepted-token totals matched
the source receipts. It scored the 199 first-rejection queries with the full
canonical Q4_0 assistant embedding using the original production Metal GEMV.
All 199 recomputed argmax tokens matched the dump exactly. No BF16 head
approximation or shortlist was used for these ranks.

| Request | First rejections | Target was runner-up | Target in top4 | Target in top8 |
|---|---:|---:|---:|---:|
| Inference | 51 | 20 | 31 | 40 |
| Coding | 23 | 16 | 18 | 21 |
| Planning | 52 | 20 | 31 | 39 |
| Roadmap | 65 | 20 | 32 | 43 |
| Total excluding warmup | 191 | 76 | 112 | 143 |

The four measured chats emitted 765 decode tokens in 221 rounds (544
accepted drafts plus one target token per round). Hypothetically obtaining
one additional token on each of the 76 top2 hits raises 3.462 to 3.805 tokens
per round, a 9.93% gain. Accepted drafts alone rise 13.97%. The analogous
one-extra-token ceilings for top4 and top8 are 14.64% and 18.69%, with higher
verification cost. These calculations hold the original rounds fixed and
are feasibility estimates, not results of a tree verifier.

The median gap between the first draft logit and the correct target logit
was 1.791 on top2 hits and 5.300 on misses. The coding subset has the highest
runner-up hit rate (69.6%) but already has high acceptance, so its one-extra
ceiling is only 8.38%. Deeper alternative continuations might add more than
one token; the saved query set contains the original greedy draft path and
cannot estimate those continuations.

See `first_reject_top2_receipt.json` for identities and exact statistics.
The full per-query ranks and the bounded experiment scripts are retained in
the takeover task's `work/assistant/` folder; the independent pairing map is
in `work/decoder-audit/first-reject-map.json`. No model or source artifact
was modified and no mini2 compute was used for this study.
