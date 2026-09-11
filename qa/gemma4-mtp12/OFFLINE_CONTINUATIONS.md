# Offline assistant continuation diagnostics

Set `CAMELID_MTP12_DUMP_FINAL_KV=/absolute/output/directory` together with
`CAMELID_MTP12_DUMP_DRAFT_QUERIES=/absolute/queries.bin` for a diagnostic run.
Use fresh query and `.seeds` paths for every server run. These captures are not
throughput qualifications: per-round diagnostic writes affect decode timing.

At the end of each request, a uniquely named directory contains `snapshot.json`
and the final committed key/value prefixes for source layers46 and47. KV files
are little-endian f32 in `[kv_head][committed_position][head_dim]` order, without
capacity padding or rejected speculative rows. Metadata records dimensions,
file SHA256 values, exact prompt/generated token IDs, source model identities,
selectors, and the committed logical prefix. The state mutex remains held while
the buffers are copied. A missing source view, pending ticket, inconsistent
prefix or non-finite committed value fails the explicitly requested diagnostic.
With the selector absent, no snapshot buffers or files are allocated.

The existing 4,112-byte query format is unchanged. When both selectors are set,
`queries.bin.seeds` also receives one15,376-byte record per draft round:

- Four LEu32 fields: anchor token, query anchor position, fixed target KV length,
  and draft count.
- 3,840 LEf32 values of the initial recurrent hidden, before the existing
  embedding/recurrent gather applies BF16 rounding.

Request boundaries can be derived from snapshot round counts and prompt token
counts. Check every query header, seed header, next-round base increment, and
accepted count against the response receipt. Compare full generated token arrays
and request JSON with the frozen baseline; saved exactness flags alone are not
proof.

The recorded1,024-wide head query is the final-normalized assistant hidden. It
is exactly the input of `post_projection.weight`; its Q4 GEMV rounds the3,840
outputs to BF16. Thus an offline branch can regenerate its parent's recurrence
without replaying preceding assistant layers. For end-to-end assistant ablations,
the seed sidecar supplies each round's original initial hidden.

Committed target KV rows are causal and immutable after acceptance. Slicing the
final snapshot to an earlier round's prefix P reconstructs the target KV that
round borrowed. Prove this assumption with an exact replay of all captured main
draft tokens before interpreting alternate continuation results. Branches must
be chosen from pre-verification margins or another uniform policy; selecting
the future target mismatch is only an explicitly labeled oracle ceiling.

The3,840-wide post-projection recurrence approximates the teacher's
final-normalized hidden space. Applying the target head to it directly is an
optional reranking experiment; applying target output RMS again would be an
additional transform, not the official recurrence contract.
