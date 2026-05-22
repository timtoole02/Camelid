# cron-95495a91 x86 Q8 QKV decode group chunking

UTC: 2026-05-21T18:53:39Z

## Slice

Implemented a default-off scheduler slice for the existing x86 Q8 attention Q/K/V decode triplet path:

- Added `CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING` to the execution-plan managed x86 Q8 knobs.
- Kept the QKV decode consumer itself gated by `CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER`.
- Chunked parallel QKV decode output groups only when the new chunking flag is enabled and the existing decode parallel threshold is already met.
- Preserved the unchunked/fallback triplet path.

## Validation

Local validation ran on macOS arm64 only; Ubuntu x86 same-host timing/parity was not feasible in this run.

Commands run:

```bash
cargo fmt --check
cargo test q8_attention_qkv_decode_group_chunking_matches_unchunked_triplet_projection --lib
cargo test q8_attention_qkv_consumer_quantizes_once_for_runtime_packed_qkv --lib
cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib
cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib
```

Results:

- `q8_attention_qkv_decode_group_chunking_matches_unchunked_triplet_projection`: passed.
- `q8_attention_qkv_consumer_quantizes_once_for_runtime_packed_qkv`: passed.
- `ubuntu_experimental_validated_gates_select_rust_avx2_q8_path`: passed.
- `planner_env_apply_clears_stale_x86_q8_decode_consumer_flags`: passed.

An initial combined `cargo test` invocation was rejected by Cargo because multiple test-name filters were passed before `--`; the listed single-filter reruns passed.

## Timing

No retained wall-clock performance claim. This slice narrows a known decode scheduler bottleneck by reducing parallel task fan-out under an explicit x86 Q8 default-off gate. Ubuntu x86 same-host timing/profiling is still required before retaining any speed claim.
