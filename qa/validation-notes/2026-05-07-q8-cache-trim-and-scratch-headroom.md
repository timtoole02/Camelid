# 2026-05-07 — Q8 cache trim and batch scratch headroom

Scope: structural performance/RSS headroom only. This does not widen model support, API capabilities, frontend readiness, context support, or broader/full 8B claims.

Change:

- Tightened the bounded Q8_0 file-read cache eviction path for sequential adjacent streams. When a coalesced cache entry has grown to the configured capacity and the next adjacent chunk arrives, Camelid now trims the merged entry to the newest capacity-sized contiguous window instead of evicting the entire older coalesced entry. This preserves a useful tail window for repeated long-prefill Q8 reads while keeping the same cache byte cap.
- Added a multi-row lazy-Q8 reader scratch cap. Batch/prefill Q8 file-backed matmul now sizes output-column chunks by both encoded weight chunk bytes and reusable f32 output-scratch bytes (`CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES`, default 64 MiB). Single-row behavior stays unchanged; tall prefill batches get a controlled RSS-vs-read-reuse tradeoff instead of unbounded scratch growth.

Local gates:

- `./scripts/with-rustup-cargo.sh fmt --all -- --check`
- `./scripts/with-rustup-cargo.sh test -q q8_file_reader_batch_chunk_rows_respect_output_scratch_budget --lib`
- `./scripts/with-rustup-cargo.sh test -q q8_file_cache --lib`
- `./scripts/with-rustup-cargo.sh test -q q8_0_file_backed_batch_matmul_reuses_chunk_reads_across_input_rows --lib`
- `./scripts/with-rustup-cargo.sh test -q q8_0_file_backed_borrowed_batch_matmul_reuses_chunk_reads_across_input_rows --lib`
- `./scripts/with-rustup-cargo.sh test -q prefill --lib`
- `./scripts/with-rustup-cargo.sh clippy -q --all-targets -- -D warnings`

Remote note: remote validation was not attempted for this local slice; no duplicate 8B long-context validation run was launched.

Claim boundary: code-only structural headroom plus local guardrails. The existing exact-row bounded 8B 512/1024/2048 support boundary is unchanged; model-native/larger context, arbitrary templates, production throughput, portability, neighboring rows, and broad/full Llama/8B support remain gated by separate PASS artifacts and synchronized alignment.
