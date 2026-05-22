# cron-1eeef0a5 VNNI decode group chunking

UTC: 2026-05-22T01:04Z

Source base: `origin/main` at `d388c036fbe8cf742c3571b8b5548b98891b5ec1`

Branch: `rust-vnni-decode-chunking-20260522T0104Z`

## llama.cpp Archaeology

The current Ubuntu x86_64 llama.cpp Q8 path is graph-level `GGML_OP_MUL_MAT` over `GGML_TYPE_Q8_0` weights. On CPU, `ggml/src/ggml-cpu/ggml-cpu.c` maps Q8_0 to `ggml_vec_dot_q8_0_q8_0` with Q8_0 input rows, tiles `MUL_MAT` work into chunks, and assigns chunks through `threadpool->current_chunk`.

Exact source anchors inspected on Ubuntu:

- `ggml/src/ggml-cpu/ggml-cpu.c:262`: Q8_0 CPU type trait uses `quantize_row_q8_0`, `ggml_vec_dot_q8_0_q8_0`, and Q8_0 as the vector-dot input type.
- `ggml/src/ggml-cpu/ggml-cpu.c:1313`: when `src1` is not already Q8_0, llama.cpp quantizes the RHS activation into the vector-dot type before compute.
- `ggml/src/ggml-cpu/ggml-cpu.c:1387`: `MUL_MAT` chooses a 16 or 64 element chunk size, then splits by result dimensions and falls back to per-thread chunking when chunks are too few for the thread count.
- `ggml/src/ggml-cpu/ggml-cpu.c:1417`: workers consume chunks by atomic `current_chunk`, avoiding one task per tiny output group.
- `ggml/src/ggml-cpu/arch/x86/quants.c:1170`: x86 Q8_0 dot uses AVX2 and multiplies per-block fp16 scales converted to f32 inside the dot loop.
- `ggml/src/llama-context.cpp:1701` and `src/llama-context.cpp:1752`: llama.cpp splits the request into ubatches; decode is the same `decode()` graph path with smaller ubatches, while prefill uses batched ubatches.
- `ggml/src/llama-context.cpp:2293`: graph compute selects `n_threads_batch`/`threadpool_batch` for batched prefill and `n_threads`/`threadpool` for one-row decode.

The local llama.cpp reference binary loaded the 3B Q8_0 file as CPU backend and reported separate `pp16` and `tg1` tests:

```text
llama 3B Q8_0 | 3.18 GiB | 3.21 B | CPU | threads=4 | pp16 | 121.51 t/s
llama 3B Q8_0 | 3.18 GiB | 3.21 B | CPU | threads=4 | tg1  | 13.72 t/s
build: 726704a
```

## Camelid Slice

Translated the bounded scheduling insight into a default-off Rust slice for the existing FFN-down VNNI raw-pointer decode helper:

- `src/inference.rs`: added `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUP_CHUNKING=on` plus `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUPS_PER_CHUNK`; when enabled, the AVX512 and AVX2 raw-pointer VNNI decode helpers process multiple 64-output groups per Rayon job instead of one Rayon job per 64-output group.
- `src/inference/tests.rs`: added Linux x86_64 parity coverage comparing chunked raw-pointer VNNI decode against the unchunked raw-pointer path.
- `src/execution_plan.rs`: added the new gate to managed default-off env handling so appliance planning clears stale values unless explicitly selected.
- `docs/CONFIGURATION.md` and `docs/performance/ubuntu-x86-q8.md`: documented the gate as a developer experiment only.

The fallback/reference path is unchanged. The new scheduling knob is default-off and only affects the already gated raw-pointer VNNI path.

## Ubuntu x86_64 Validation

Host proof:

```text
Linux x86_64
cargo 1.95.0 (f2d3ce0bd 2026-03-21)
```

Commands run on the canonical Ubuntu x86_64 validation host:

```bash
cargo fmt --all
cargo test q8_ffn_down_vnni_decode_rawptr_group_chunking_matches_unchunked_rawptr --lib -- --nocapture
cargo test q8_ffn_down_vnni_decode_rawptr_matches_rows4_decode_baseline --lib -- --nocapture
cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib -- --nocapture
cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib -- --nocapture
cargo fmt --all -- --check
cargo check
```

Results:

- `q8_ffn_down_vnni_decode_rawptr_group_chunking_matches_unchunked_rawptr`: passed.
- `q8_ffn_down_vnni_decode_rawptr_matches_rows4_decode_baseline`: passed.
- `planner_env_apply_clears_stale_x86_q8_decode_consumer_flags`: passed.
- `ubuntu_experimental_validated_gates_select_rust_avx2_q8_path`: passed.
- `cargo fmt --all -- --check`: passed.
- `cargo check`: passed.

## Boundary

Retain as parity/control-plane evidence for a default-off Ubuntu/Linux x86_64 scheduling slice only. No throughput, support, portability, RSS, default-on, Mac, Metal, Mixtral, or broader model-family claim is made from this run.
