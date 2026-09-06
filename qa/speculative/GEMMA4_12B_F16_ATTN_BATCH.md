# Gemma 4 12B strict F16 verifier attention

## Scope

This checkpoint prototypes the attention half of the Gemma 4 12B Metal verifier. It
starts from `e3ba4ce72c0462f8e38ca2af1ee271ee2dbbfb4f` and is deliberately not connected
to the resident model yet.

The implementation admits only the two exact 12B geometries:

| layer kind | Q heads | KV heads | head dimension | causal range |
|---|---:|---:|---:|---|
| sliding | 16 | 8 | 256 | last 1,024 positions |
| global V-less | 16 | 1 | 512 | full prefix |

Verifier widths are exactly `K=1,2,4,8`. Rows must be consecutive linear verifier
positions. The production entry point is additionally gated by
`CAMELID_GEMMA4_ATTN_BATCH_K=1`, and no current runtime calls it.

## Kernel contract

The new kernel family lives in `src/gemma4_attention_metal.rs`; it does not reuse the
Llama split-K shader whose head dimension is capped at 128.

- K/V storage is F16; query, online-softmax state, split partials, and output are F32.
- Metal fast math is disabled.
- Each verifier row carries its own absolute window start, position count, and split
  count in a `uint4` metadata record.
- The sliding cache is a ring and must have `1,024 + K` slots. The K slack slots are
  required because all candidates are scattered before the batched attention dispatch.
- Sliding attention stages one KV tile for the two query heads sharing a KV head.
- Global attention stages one KV tile for eight query heads at a time, so the 16:1 GQA
  group reads each tile twice rather than sixteen times.
- A V-less layer still binds a separate V cache. That cache contains the
  weightless-normalized raw K projection before K-norm/RoPE. Binding the roped K cache
  as V is incorrect and has a dedicated tripwire test.

The chosen exact oracle is the strict one-row Metal kernel. It uses the same split
boundaries, four-score online-softmax recurrence, SIMD dot reduction, and ordered split
merge as the row-dimensional kernel. This makes exactness a bit comparison rather than
a tolerance claim. A separate sequential CPU F16 calculation checks the window metadata
and V-cache interpretation within a `2e-3` envelope.

## Receipt

Run on an Apple Metal device:

```bash
cargo test --lib gemma4_attention_metal::tests -- --nocapture --test-threads=1
```

Result:

```text
running 3 tests
test gemma4_attention_metal::tests::global_vless_reads_pre_rope_value_cache_not_key_cache ... ok
test gemma4_attention_metal::tests::plan_admits_only_exact_widths_and_keeps_sliding_batch_slack ... ok
test gemma4_attention_metal::tests::strict_f16_batch_matches_row_oracle_at_1024_boundary ... ok

test result: ok. 3 passed; 0 failed
```

The exactness matrix covers both geometries at every admitted width. Its verifier rows
begin at position 1,023, so the matrix explicitly crosses positions 1,023, 1,024, and
1,025 where the sliding window changes from `[0,1024)` to `[1,1025)` and `[2,1026)`.

## Integration boundary

The dense verifier still needs to own F16 K/V rings, scatter all candidate K/V rows,
and retain the returned scratch buffers until its command buffer completes. The current
prototype intentionally omits tree-slot indirection and caps a global row at 4,096
positions (`64` splits of nominal span `64`). Those are fail-closed boundaries, not
claimed runtime support.
