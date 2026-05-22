# cron-1eeef0a5 current-main VNNI decode group chunking

UTC: 2026-05-22T08:09Z

Ubuntu host: `Linux x86_64` (`ip-172-31-19-175`)

Source base: Camelid `68fa3a6` (`perf(q8): gate x86 FFN decode chain (#82)`)

llama.cpp source base: `/home/ubuntu/work/llama.cpp-clean-20260517` at `4f0e43d`

Branch/worktree: `/home/ubuntu/work/camelid-cron-1eeef0a5-20260522T0809Z-arch-current`

## llama.cpp Archaeology

Inspected the current Ubuntu x86_64 llama.cpp CPU Q8 path for tensor type, matmul dispatch, scheduler/threading, prefill/decode split, and scale/dequant handling.

Source anchors:

- `ggml/src/ggml-common.h:241`: `block_q8_0` is a 32-value block with fp16 `d` scale plus 32 signed i8 quants; sizeof is 34 bytes.
- `ggml/src/ggml.c:721`: `GGML_TYPE_Q8_0` maps to block size `QK8_0`, `sizeof(block_q8_0)`, `dequantize_row_q8_0`, and `quantize_row_q8_0_ref`.
- `ggml/src/ggml-quants.c:234`: reference quantization stores `GGML_FP32_TO_FP16(d)`; `ggml/src/ggml-quants.c:491` dequantizes with `GGML_FP16_TO_FP32(x[i].d)`.
- `ggml/src/ggml-cpu/ggml-cpu.c:262`: CPU Q8_0 trait uses `quantize_row_q8_0`, `ggml_vec_dot_q8_0_q8_0`, and `GGML_TYPE_Q8_0` as the vector-dot RHS type.
- `ggml/src/ggml-cpu/arch/x86/quants.c:1170`: x86 Q8_0 dot asserts one row, uses AVX2 when available, multiplies fp16-derived block scales inside the dot loop, and falls back to scalar for remaining blocks.
- `ggml/src/ggml-cpu/ggml-cpu.c:1245`: `ggml_compute_forward_mul_mat` quantizes a non-Q8 RHS activation to the vector-dot type, then computes `MUL_MAT` with chunked work.
- `ggml/src/ggml-cpu/ggml-cpu.c:1387`: chunk size is 16 normally and 64 when either result dimension is one; poor chunk count falls back to per-thread chunks.
- `ggml/src/ggml-cpu/ggml-cpu.c:1417`: worker threads consume chunks through `threadpool->current_chunk` instead of launching one tiny task per output group.
- `src/llama-context.cpp:1715`: decode/prefill both process ubatches through `memory->init_batch`; prefill is larger ubatches while one-token generation is the small decode shape.
- `src/llama-context.cpp:2307`: `graph_compute` switches between `n_threads_batch/threadpool_batch` for batched prefill and `n_threads/threadpool` for decode.

Bounded benchmark sanity on the same Ubuntu host, CPU-only, Q8_0 3B:

```text
./build/bin/llama-bench -m /home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf -p 16 -n 1 -t 4 -ngl 0 -r 1 --no-warmup -o md
llama 3B Q8_0 | 3.18 GiB | 3.21 B | CPU | threads=4 | pp16 | 119.23 t/s
llama 3B Q8_0 | 3.18 GiB | 3.21 B | CPU | threads=4 | tg1  | 13.63 t/s
build: 726704a (1)
```

## Camelid Comparison

Current Camelid main already has backend-owned `PackedRows4` and VNNI sidecars. Key differences against llama.cpp for the current 3B Q8 gap:

- Camelid `src/tensor/mod.rs:84` keeps hot Q8 base blocks as f32 scale plus i8 quants in `Q8_0Block`, while llama.cpp canonical `block_q8_0` keeps fp16 scale bits in the source block. Camelid VNNI sidecar now carries both fp16 bits and f32 lanes.
- Camelid `src/inference.rs` has many default-off projection owners, but its decode raw-pointer VNNI helper previously let Rayon split work one 64-output group at a time. llama.cpp instead chunks result dimensions and uses atomic chunk consumption.
- Camelid prefill/decode are split through explicit high-level paths, while llama.cpp keeps both under graph/ubatch compute with separate threadpool choice.

## Retained Slice

Retained one bounded scheduling insight: coarse chunking for the existing default-off FFN-down VNNI raw-pointer decode helper. This does not replace the kernel, tensor format, or support contract.

Changes:

- `src/inference.rs`: added `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUP_CHUNKING=on` and `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_GROUPS_PER_CHUNK`; AVX512 and AVX2 raw-pointer VNNI decode helpers can process multiple 64-output groups per Rayon job.
- `src/execution_plan.rs`: added the gate to managed default-off x86 Q8 env handling.
- `src/inference/tests.rs`: added parity coverage comparing chunked raw-pointer VNNI decode against unchunked raw-pointer output.
- `docs/CONFIGURATION.md` and `docs/performance/ubuntu-x86-q8.md`: documented the gate as a developer experiment only.

Boundary: parity/control-plane only. No throughput, support, portability, RSS, default-on, Mac, Metal, Mixtral, or broader family claim.

## Ubuntu x86_64 Commands

Disk guard was run before cargo:

```text
2026-05-22T08:08:38Z CHECK avail_kb=142815688 use_pct=30 target_count=4
2026-05-22T08:08:40Z AFTER avail_kb=142815688 use_pct=30 target_count=4
```

Cargo target was externalized for all cargo commands:

```bash
export CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/archaeology-1eeef0a5
cargo fmt --check
cargo test q8_ffn_down_vnni_decode_rawptr_group_chunking_matches_unchunked_rawptr -- --nocapture
cargo test ubuntu_experimental_preserves_explicit_x86_q8_gate_opt_ins -- --nocapture
cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags -- --nocapture
cargo check --lib
```

Results:

- `cargo fmt --check`: passed.
- `q8_ffn_down_vnni_decode_rawptr_group_chunking_matches_unchunked_rawptr`: passed.
- `ubuntu_experimental_preserves_explicit_x86_q8_gate_opt_ins`: passed.
- `planner_env_apply_clears_stale_x86_q8_decode_consumer_flags`: passed.
- `cargo check --lib`: passed.
