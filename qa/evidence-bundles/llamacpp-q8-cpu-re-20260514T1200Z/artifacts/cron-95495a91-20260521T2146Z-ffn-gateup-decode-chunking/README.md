# FFN Gate/Up Decode Group Chunking

Cron: `95495a91-9153-4657-be7b-4b435be83268`

UTC: `2026-05-21T21:46Z`

Scope: local parity/control-plane evidence for a default-off Ubuntu x86 Q8 experiment. This slice adds `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING=on` for the existing FFN gate/up paired decode consumer. The goal is to reduce Rayon task fan-out across wide output groups while preserving the existing default-off gate and fallback/reference path.

Changed code:

- `src/inference.rs`
- `src/inference/tests.rs`
- `src/execution_plan.rs`
- `docs/performance/ubuntu-x86-q8.md`
- this evidence note

Validation run locally:

```bash
cargo fmt
cargo test q8_ffn_gate_up_decode_group_chunking_matches_unchunked_pair_projection -- --nocapture
cargo test resolved_runtime_plan_captures_q8_env_once -- --nocapture
cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path -- --nocapture
cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags -- --nocapture
```

Result:

- New paired FFN gate/up chunking parity test passed against the unchunked projection.
- Runtime-plan capture/default-off coverage passed.
- ExecutionPlan Ubuntu x86 experimental default-off gate coverage passed.
- Planner stale-env cleanup coverage passed.

Host/timing status:

- Local host for this cron turn was `Darwin ... arm64`, not Ubuntu x86_64.
- Same-host Ubuntu timing/profiling was not feasible in this run because no current canonical Ubuntu x86 validation command was provided or executed.
- No throughput, support, portability, default-on, Mac, Mixtral, or broad model promotion is claimed from this local evidence.
