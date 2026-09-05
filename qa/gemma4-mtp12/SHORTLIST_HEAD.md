# Overlapping assistant head shortlist

`CAMELID_GEMMA4_MTP12_SHORTLIST=/path/to/assistant.c4sl` enables an optional,
draft-only head shortlist. `CAMELID_GEMMA4_MTP12_SHORTLIST_TOP=192` selects the
number of the 2,048 raw-mean centroids to keep; admitted values are 1..=2048.
The default is 192. An absent sidecar setting preserves the full-head path;
a malformed configured sidecar is an error. The sidecar is tied to the exact
assistant SHA and adds 10,502,144 bytes of resident storage.

The GPU scores the centroids, selects the highest scores with stable ties,
and keeps a token if any of its three cluster IDs is selected. Kept rows use
the identical Q4 dot/reduction as the full head; omitted logits are -infinity.
The ordinary vocabulary argmax then chooses the draft. Only the resident
chain uses this path: K=1 full-logit APIs retain the full-head oracle.
The target verifier is unchanged and remains authoritative for emitted tokens.
A shortlist may alter draft proposals and acceptance; output parity and
acceptance must be measured on the actual target workload before enabling it.

The earlier single-assignment clustering experiment attained only 94.8%
recall at top128. The existing overlapping top3 sidecar changes that tradeoff:
on the saved 1,587 W8 draft queries, top128 recall is 98.74% and top192 is
99.18%. These are offline draft recall figures, not a generation acceptance
or throughput claim.

Local M4 synthetic Q4 head timing, 262,144 rows x 1,024 columns, nonzero
18-byte matrix offset, actual sidecar and first saved query, 7 sequential
heads per command, median of 8 warmed commands (2026-09-04):

| Head path | Selected rows | Median per draft |
|---|---:|---:|
| Full | 262,144 | 1,627 us |
| Top128 | 28,318 | 574 us |
| Top192 | 41,735 | 576 us |
| Top256 | 54,767 | 621 us |
| Top2048 | 262,144 | 1,788 us |

Shortlist timings include centroid scoring and selection. All kept row
logits matched the full head bit for bit; all omitted rows were -infinity.
Top2048 matched all full-head logits. GPU-selected centroid IDs matched CPU
stable ranking. These timings are isolated kernel measurements, not mini2
end-to-end results.

Validation:

```sh
cargo test --lib assistant_shortlist -- --nocapture
cargo test --lib gemma4_mtp12::shortlist -- --nocapture
```

The GPU fixture checks tied centroid scores, top1/17/128/192/256/2048,
8,193 nontrivial Q4 rows, a nonzero matrix offset, exact kept logits,
masked-out logits, and both vocabulary argmax kernels. Parser tests cover
identity, dimensions, nonfinite centroid values, cluster bounds/duplicates,
invalid padding, truncation, and trailing bytes.
