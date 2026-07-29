# Runtime foundations: projects 2, 4, 5, 6, and 7

This slice deliberately separates implemented foundations from production
promotion. Exact model support remains governed by the existing evidence
ledger.

## Delivered

- **2 — capability/config registry:** typed bounds for new runtime controls
  plus `config/runtime-capabilities.json`, exposed by `/api/capabilities`.
- **4 — CPU prefill:** centralized the Q4_K/Q6_K owner gate and reused Q6_K
  Rayon scratch per worker. The owner stays opt-in; bitwise Q4_K/Q6_K parity
  tests are the promotion floor.
- **5 — speculative decoding 2.0:** bounded incremental n-gram index with
  collision verification, rollback rebuild, exact longest/recent selection
  for retained candidates, stats, and a release microbenchmark.
- **6 — unified KV manager:** stable sequence IDs, active/idle lifecycle,
  global allocation budget, idle-only LRU eviction, and sequence-local
  checkpoints over the existing quantized `LlamaKvCache`.
- **7 — batching/slots:** real production-engine active-task snapshots power
  `/slots`; a slot-bounded fair token-step scheduler is implemented and tested
  as the integration foundation.

## Default behavior

- The production inference engine remains one exclusive owner.
- Continuous batching and the unified KV pool are not yet wired into default
  generation.
- Q4_K/Q6_K owner prefill remains default-off.
- N-gram indexing is used only when n-gram speculation itself is selected.

These defaults avoid throughput, memory, or output changes in ordinary serving
while the remaining integration and real-model benchmark gates are completed.
