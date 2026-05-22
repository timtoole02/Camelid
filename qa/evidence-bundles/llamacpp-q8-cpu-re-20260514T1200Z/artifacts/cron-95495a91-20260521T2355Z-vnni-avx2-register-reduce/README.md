# cron-95495a91 VNNI AVX2 register reducer

## Scope

Default-off Ubuntu/Linux x86_64 Q8 FFN-down VNNI decode raw-pointer follow-on.

This slice replaces the AVX2 VNNI tile16 pair reduction's per-accumulator stack store plus scalar pair sums with an in-register `vphaddd`/lane-unpack reducer. The safe scalar/reference VNNI path and the default-off `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR` gate are unchanged.

No Mac, Mixtral, support, default-on, or broad throughput promotion is claimed.

## Changed files

- `src/inference.rs`
- `src/inference/tests.rs`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260521T2355Z-vnni-avx2-register-reduce/README.md`

## Local validation

Worktree: clean temp worktree from `origin/main` at `2d82619` with PR #49's merge commit already contained in main; after PR creation, the single commit was rebased onto current `origin/main` at `dc9752e`.

- `cargo fmt --check` passed.
- `cargo check --all-targets --all-features` passed locally.
- `cargo test q8_ffn_down_vnni_decode --lib` passed locally: 2 passed, 293 filtered.

## Ubuntu x86_64 validation

Host facts:

- `Linux 6.17.0-1013-aws x86_64 GNU/Linux`
- `rustc 1.95.0 (59807616e 2026-04-14)`
- `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`

Commands run in `/tmp/camelid-95495a91`:

- `cargo test q8_ffn_down_vnni_decode_rawptr_avx2 --lib -- --nocapture` passed: 1 passed, 307 filtered.
- `cargo test q8_vnni_tile16_avx2 --lib -- --nocapture` passed: 1 passed, 307 filtered.
- Before rebase: `cargo check --all-targets --all-features` passed, wall 7.64s after build cache.
- After rebase onto `dc9752e`: `cargo check --all-targets --all-features` passed, wall 0.72s after build cache.
- After rebase onto `dc9752e`: `cargo test q8_ffn_down_vnni_decode_rawptr_avx2 --lib -- --nocapture` passed: 1 passed, 307 filtered.
- After rebase onto `dc9752e`: `cargo test q8_vnni_tile16_avx2 --lib -- --nocapture` passed: 1 passed, 307 filtered.

Manual release microbench:

```text
cargo test --release q8_vnni_avx2_pair_reducer_benchmark --lib -- --ignored --nocapture
```

Repeated same-host results after release build cache:

```text
iterations=10000000 legacy_store_us=7531 register_hadd_us=7259 checksum=226447232
iterations=10000000 legacy_store_us=7481 register_hadd_us=7273 checksum=226447232
iterations=10000000 legacy_store_us=7518 register_hadd_us=7227 checksum=226447232
iterations=10000000 legacy_store_us=7562 register_hadd_us=7289 checksum=226447232
iterations=10000000 legacy_store_us=7607 register_hadd_us=7205 checksum=226447232
```

The repeated reducer-only wall-clock range moved from `7481-7607us` for the old store reducer to `7205-7289us` for the new register reducer. This narrows only the AVX2 tile16 reduction bottleneck inside the default-off VNNI raw-pointer experiment; it is not model-level or llama.cpp parity evidence.

Post-rebase release microbench smoke:

```text
iterations=10000000 legacy_store_us=7461 register_hadd_us=7309 checksum=226447232
```

## Retain/reject decision

Retain as a narrow hot-loop cleanup because it keeps parity and shows same-host reducer timing improvement under the Ubuntu x86_64 release microbench.

Reject any throughput/support/default-on promotion from this slice. Full model same-host Camelid vs llama.cpp timing was not run in this slice.
