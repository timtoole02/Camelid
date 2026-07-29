# Indexed n-gram lookup receipt — 2026-07-28

Scope: synthetic lookup-only benchmark for project 5. This is not an
end-to-end generation tok/s claim.

Command:

```text
cargo test --release --lib indexed_ngram_microbench_against_reference_scan -- --ignored --nocapture
```

Workload:

- 16,384-token unique base history
- 512 incremental lookups
- match lengths 3 through 4
- miss-heavy suffixes (the historical implementation scans the history)
- release profile, Windows x86_64, same process

Result:

```text
indexed=2.2129ms
scan=59.8836ms
speedup=27.06x
```

Correctness gates run separately:

- `indexed_ngram_matches_reference_scanner`: PASS across 1,200 incremental
  histories and draft budgets 1, 3, and 7.
- `ngram_index_is_bounded_and_survives_rollback`: PASS with a 48-record
  ceiling and history rollback/re-append.
- Every hash candidate is compared token-for-token before drafting.
- A bounded eviction can suppress a proposal, but target-model verification
  remains authoritative and generated output cannot change.
