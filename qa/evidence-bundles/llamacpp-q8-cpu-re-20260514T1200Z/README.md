# Ubuntu x86_64 Q8_0 CPU performance investigation — Llama 3.2 3B Instruct Q8_0

Generated: 2026-05-14 UTC
LANE: `UBUNTU_X86_Q8`
Scope: Ubuntu x86_64 dense Llama Q8_0 only.

Claim guardrail: this report is the current Q8 reference truth for the Ubuntu x86_64 experiment lane only. It is not Mac, Apple Silicon, Metal, Mixtral, portability, production-throughput, or support-contract evidence. All Camelid x86 Q8 runtime changes described here are default-off developer experiments unless explicitly promoted by separate support evidence.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T22:07Z

- Focused on clearing the previous Ubuntu validation blocker for the existing default-off multi-row `output.weight` PackedRows4 matmul consumer (`CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL`), which consumes backend-owned `Q8_0RuntimeStorage::PackedRows4` and adds no duplicate packed-copy sidecar.
- Refreshed llama.cpp grep evidence for Q8_0 vec-dot, repack/GEMV/GEMM hooks, x86 AVX2/AVX512/VNNI branches, `MUL_MAT` wiring, and OpenMP scheduling, plus Camelid grep evidence for the output packed-rows4 matmul flag, runtime-plan gating, backend-owned storage, and helper/test anchors.
- No fresh canonical Ubuntu validation was attempted in this documentation pass; this historical local-only artifact remains non-promotional without a current canonical Ubuntu timing/profiling bundle.
- No code change, support promotion, portability claim, Ubuntu throughput claim, retained measured effect, or default-on behavior was added; the output packed-rows4 matmul slice remains default-off and local-only until host validation succeeds.
- Artifact: `artifacts/cron-95495a91-20260517T2207Z-x86-output-packed-rows4-canonical-host-blocker.txt`.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T21:31Z

- Guarded the current Ubuntu x86_64 Q8 docs/report surface after the local-only packed-rows4 quantized-input scratch follow-on and output packed-rows4 local-only validation artifact without changing the project README support matrix, support contract, API/frontend readiness, platform labels, or public performance claims.
- Scrubbed raw local workspace paths, raw operator key path, and raw validation-host IP from `artifacts/cron-95495a91-20260517T2013Z-x86-output-packed-rows4-host-blocker.txt`, replacing them with `<llamacpp-workdir>`, `<camelid-workdir>`, `<operator-key>`, and `<validation-host>` placeholders.
- Re-ran `node scripts/audit-evidence-bundle-privacy.mjs --root qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z --strict`: `finding_count: 0`, `bundle_count_with_findings: 0`.
- Re-ran `node scripts/check-public-evidence-claims.mjs`: passed with `113 manifest(s), 51 summary file(s)`.
- Reviewed claim-sensitive lines across README/config/performance/report surfaces; the newest output/chunking/scratch follow-ons remain local-only/default-off where applicable and are not Ubuntu throughput, support, portability, retained-speedup, or default-on evidence.
- No Mac evidence, Apple Silicon label, Metal claim, Mixtral claim, production-throughput claim, portability claim, support-contract promotion, or default-on acceleration claim was added.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T2131Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 1eeef0a5, 2026-05-17T21:18Z

- Small local follow-on changed packed-rows4 matmul consumers to reuse the existing bounded thread-local quantized-input scratch buffer instead of allocating a fresh `Vec<Q8_0Block>` for every packed-rows4 matmul helper call.
- The slice stays inside existing default-off packed-runtime matmul gates, consumes backend-owned `Q8_0RuntimeStorage::PackedRows4` for weights, and adds no duplicate packed sidecar, new runtime flag, ISA expansion, tensor-family expansion, support claim, or default-on behavior.
- Covered consumers include the shared single-projection helper used by FFN-down, attention-output, and output packed-rows4 matmul plus the paired FFN gate/up and triplet attention Q/K/V packed-rows4 paths.
- Local validation passed: `cargo fmt --all -- --check`, focused packed-rows4 tests, `cargo clippy --all-targets --all-features -- -D warnings`, full `cargo test --lib` (`261 passed; 0 failed`), and `node scripts/check-public-evidence-claims.mjs`.
- Focused local timing smoke for the new scratch test passed with `/usr/bin/time -p cargo test q8_packed_rows4_matmul_quantized_input_scratch_matches_owned_rows --lib`: `real 1.02`, `user 0.04`, `sys 0.06`. This is allocation-shape/local timing evidence only, not a retained Ubuntu throughput result.
- No fresh canonical Ubuntu timing/profiling was attempted in this documentation pass. No Ubuntu throughput/support/default-on promotion or retained measured speedup is claimed.
- Artifact: `artifacts/cron-1eeef0a5-20260517T2118Z-x86-packed-rows4-input-scratch-local.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 1eeef0a5, 2026-05-17T20:01Z

- Local bounded follow-on changed packed-rows4 matmul projection helpers to chunk parallel output-group traversal for single, paired, and triplet multi-row projections rather than scheduling one Rayon task per 4-output group once the default-off matmul path is already selected.
- The slice stays inside backend-owned `Q8_0RuntimeStorage::PackedRows4` consumers, keeps matching I8 rows4/storage-shape guards, and does not add duplicate packed sidecars or any default-on behavior.
- Added `q8_packed_rows4_matmul_projection_chunked_prefill_matches_manual_output` to cover chunked multi-row semantics against a manual rows4 dot reconstruction; existing output/QKV/FFN-down/planner gates were also rerun locally.
- Local validation passed after fixing clippy feedback: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --lib` (`260 passed; 0 failed`). Only the known local incremental-cache hard-link warning appeared.
- No fresh canonical Ubuntu timing/profiling was attempted in this documentation pass. No Ubuntu throughput/support/default-on promotion or retained measured effect is claimed.
- Artifact: `artifacts/cron-1eeef0a5-20260517T2001Z-x86-packed-rows4-matmul-chunking-local.txt`.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T19:57Z

- Guarded the current Ubuntu x86_64 Q8 docs/report surface after the local-only multi-row `output.weight` packed-rows4 matmul slice without changing README support matrix, support contract, API/frontend readiness, platform labels, or public performance claims.
- Tightened `docs/CONFIGURATION.md` so `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL` explicitly says its current evidence is local parity/gate coverage only without a fresh canonical Ubuntu timing/profiling bundle, and that it must not be treated as Ubuntu throughput, support, portability, or default-on evidence.
- Tightened this report's checklist row so the newest local-only output matmul flag is separated from Ubuntu-validated slice results instead of being read as a blanket Ubuntu PASS.
- Scrubbed raw operator key/path, host IP/hostname, and local workspace paths from three newly-added artifacts in this bundle; the strict focused evidence privacy audit now reports zero findings for `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z`.
- Re-ran focused claim/private-detail greps over README/status/config/performance/report surfaces: no raw operator path/IP details found, README stayed unchanged, and local-only slices remain labeled as local-only with no retained measured effect.
- No Mac evidence, Apple Silicon label, Metal claim, Mixtral claim, production-throughput claim, portability claim, support-contract promotion, or default-on acceleration claim was added.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T1957Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 1eeef0a5, 2026-05-17T18:50Z

- Small bounded slice added `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL`, a default-off multi-row `output.weight` packed-runtime matmul consumer for the Ubuntu x86_64 Q8 lane.
- The path consumes only backend-owned `Q8_0RuntimeStorage::PackedRows4` for `output.weight`, runs before the existing one-row output decode-owner path, and fails closed unless the runtime plan, Q8_0 tensor type, exact tensor name, rank-2 multi-row input, Q8 block alignment, I8 interleave, row grouping, and packed shape guards match.
- `ExecutionPlan` now manages and pins `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL=off` with the other x86 Q8 consumer/matmul gates so stale owner experiments are cleared by appliance planning.
- Local validation passed: `cargo fmt --all -- --check`, `cargo test x86_q8_output_packed_rows4_matmul --lib`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --lib` (`259 passed; 0 failed`). The focused local timing smoke was `/usr/bin/time -p cargo test x86_q8_output_packed_rows4_matmul --lib`: `real 0.94`, `user 0.04`, `sys 0.06`.
- No fresh canonical Ubuntu timing/profiling was attempted in this documentation pass for this slice. No Ubuntu throughput/support/default-on promotion or retained measured effect is claimed.
- Artifact: `artifacts/cron-1eeef0a5-20260517T1850Z-x86-output-packed-rows4-matmul-local.txt`.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T18:20Z

- Guarded the current Ubuntu x86_64 Q8 docs/report surface after the latest local-only decode output-group helper and paired/triplet projection follow-ons without widening README support matrix, support-contract, API/frontend, platform, or performance claims.
- Tightened `docs/performance/ubuntu-x86-q8.md` so packed-rows4 matmul and decode-family slices are described by per-artifact evidence level rather than as a blanket Ubuntu validation/performance claim.
- Kept reproduction scoped to the current reference gates (`CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`); narrow consumer/matmul flags remain separate default-off developer experiments only when this report names the exact flag and validation target.
- Left local-only slices as local-only: no Ubuntu timing/profiling proof, no retained measured effect, no support promotion, no portability/default-on claim.
- No Mac evidence, Apple Silicon label, Metal claim, Mixtral claim, production-throughput claim, portability claim, support-contract promotion, or default-on acceleration claim was added.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T1820Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 1eeef0a5, 2026-05-17T17:44Z

- Small local follow-on slice added a shared one-row `Q8_0RuntimeStorage::PackedRows4` decode projection helper for the existing default-off x86 Q8 packed-runtime consumers, with closed guards for I8 rows4 storage, output width, and input block count.
- The helper parallelizes independent four-output-row groups when the output width is at least 1024 and Rayon has more than one worker, mirroring the llama.cpp/OpenMP direction at the output-group scheduling level while staying inside the existing default-off consumer gates.
- Reused the helper for one-row output, attention-output, attention Q/K/V projection, and FFN-down packed-runtime decode consumers; paired FFN gate/up and triplet attention Q/K/V decode helpers now also use parallel equal-width output-group traversal after quantizing the shared activation row once.
- The path still consumes only backend-owned `Q8_0RuntimeStorage::PackedRows4`, keeps Q8 block-alignment/I8 interleave/runtime-plan/fallback guards, and adds no duplicate packed sidecar.
- Local validation passed: `cargo fmt --all -- --check`, focused Q8 consumer/matmul test filters, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --lib` (`257 passed; 0 failed`) on the non-Ubuntu workstation. x86-only focused tests and x86 Linux cross-checking were not run locally, so they are not claimed as Ubuntu validation.
- No Ubuntu x86_64 timing/support/default-on promotion is claimed from this slice; measured effect retained here is none until canonical Ubuntu timing/profiling can run. Artifact: `artifacts/cron-1eeef0a5-20260517T1744Z-x86-packed-rows4-decode-output-parallel-local.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T16:55Z

- Small local follow-on slice tightened the existing default-off `CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER` path so one-row dense attention Q/K/V decode computes all three backend-owned packed-runtime projections through one triplet helper after quantizing the shared activation row once.
- The path still consumes only backend-owned `Q8_0RuntimeStorage::PackedRows4` for `blk.*.attn_{q,k,v}.weight`, keeps the same one-row decode, Q8 block-alignment, I8 interleave, width, runtime-plan, and safe fallback guards, and adds no duplicate packed sidecar.
- Added closed mismatch checks in the Q/K/V decode triplet helper for blocks-per-row, I8 interleave, and requested-vs-packed output width agreement.
- Non-Ubuntu local compile/check coverage passed: `cargo fmt --all -- --check`, `cargo test q8_attention_qkv_consumer --lib`, `cargo test q8_attention_qkv_packed_rows4_matmul --lib`, and `cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib`.
- No fresh canonical Ubuntu validation was attempted in this documentation pass for this slice; the local compile/check coverage remains non-promotional without a current canonical Ubuntu bundle.
- No Ubuntu x86_64 timing/support/default-on promotion is claimed from this slice. Artifact: `artifacts/cron-95495a91-20260517T1655Z-x86-attn-qkv-decode-triplet.txt`.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T16:50Z

- Guarded the current Ubuntu x86_64 Q8 docs/report surface after the latest local paired-projection follow-ons without widening README support matrix, support-contract, API/frontend, platform, or performance claims.
- Kept `docs/performance/ubuntu-x86-q8.md` reproduction scoped to the current reference gates (`CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`) and left narrow consumer/matmul gates as separate default-off developer experiments only when this report names the exact flag and validation target.
- Tightened the newest local-only report wording so local compile/check slices are not presented as Ubuntu x86_64 validation, timing, support, portability, or retained-performance evidence.
- No Apple Silicon label, Metal claim, Mixtral claim, production-throughput claim, portability claim, support-contract promotion, or default-on acceleration claim was added.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T1650Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 1eeef0a5, 2026-05-17T16:45Z

- Small local follow-on slice tightened the existing default-off `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER` path so one-row dense FFN gate/up decode computes both backend-owned packed-runtime projections through one paired output-group traversal after quantizing the shared activation row once.
- The path still consumes only backend-owned `Q8_0RuntimeStorage::PackedRows4` for `blk.*.ffn_{gate,up}.weight`, keeps the same one-row decode, Q8 block-alignment, I8 interleave, width, runtime-plan, and safe fallback guards, and adds no duplicate packed sidecar.
- Added closed mismatch checks in the paired decode helper for output width, interleave, packed row count, and blocks-per-row agreement.
- Non-Ubuntu local compile/check coverage passed: `cargo fmt --all -- --check`, `cargo test q8_ffn_gate_up_consumer --lib`, `cargo test q8_ffn_gate_up_packed_rows4_matmul --lib` (0 tests on non-x86 host), `cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib`, and `cargo clippy --all-targets --all-features -- -D warnings`.
- No Ubuntu x86_64 timing/support/default-on promotion is claimed from this slice. Canonical Ubuntu x86_64 validation and bounded timing/perf proof remain pending before retaining any measured effect for this decode paired-projection tweak.
- Artifact: `artifacts/cron-1eeef0a5-20260517T1645Z-x86-ffn-gate-up-decode-paired-local.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T15:22Z

- Small technical slice tightened the existing default-off `CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL` path so dense attention Q/K/V multi-row prefill computes the three packed-runtime projections through one triplet helper after quantizing the shared activation rows once.
- The path still consumes only backend-owned `Q8_0RuntimeStorage::PackedRows4` for `blk.*.attn_{q,k,v}.weight`, keeps the same rank-2/prefill, Q8 block-alignment, I8 interleave, dimension, row-grouping, runtime-plan, and safe fallback guards, and adds no duplicate packed sidecar.
- Added a common packed-rows4 output-group guard so future narrow packed-runtime matmul helpers fail closed on non-rows4 output widths instead of silently truncating.
- Canonical Ubuntu x86_64 validation passed from a synced temp checkout: `fmt --check`, `q8_attention_qkv_packed_rows4_matmul_matches_runtime_packed_baseline_for_prefill`, and the x86-only `q8_ffn_gate_up_packed_rows4_matmul_matches_runtime_packed_baseline_for_prefill` focused test.
- No throughput/support promotion is claimed from this slice. It is runtime/helper ownership evidence for the default-off Ubuntu x86_64 experiment lane only.
- Artifact: `artifacts/cron-95495a91-20260517T1522Z-x86-attn-qkv-triplet-packed-rows4.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 1eeef0a5, 2026-05-17T15:16Z

- Small follow-on slice tightened the existing default-off `CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL` path so FFN gate/up multi-row prefill computes both packed-runtime projections through one paired traversal after quantizing the shared activation rows once.
- The path still consumes only backend-owned `Q8_0RuntimeStorage::PackedRows4` for `blk.*.ffn_{gate,up}.weight`, keeps the same rank-2/prefill, Q8 block-alignment, I8 interleave, dimension, row-grouping, runtime-plan, and safe fallback guards, and adds no duplicate packed sidecar.
- This is non-Ubuntu local compile/check coverage only: targeted adjacent packed-rows4/planner tests, `cargo fmt --all -- --check`, and `cargo clippy --all-targets --all-features -- -D warnings` passed, but the x86-only FFN gate/up packed-rows4 test filter runs 0 tests on non-x86 hosts.
- Canonical Ubuntu x86_64 validation and bounded timing/perf evidence remain the next proof step before retaining any measured effect for this paired-projection tweak.
- Artifact: `artifacts/cron-1eeef0a5-20260517T1516Z-x86-ffn-gate-up-paired-projection-local.txt`.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T14:58Z

- Guarded the current docs/report surface after the FFN gate/up packed-rows4 matmul slice without widening README support matrix, support-contract, API/frontend, platform, or performance claims.
- Confirmed `docs/CONFIGURATION.md` names `CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL` only as a default-off Ubuntu x86_64 developer experiment for multi-row dense FFN gate/up packed-runtime matmul with exact fallback guards.
- Updated `docs/performance/ubuntu-x86-q8.md` only to include the FFN gate/up packed-rows4 matmul slice in the existing default-off planner/runtime-gate experiment list and evidence anchors.
- Kept reproduction scoped to the current reference gates (`CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`); narrow consumer/matmul flags remain separate opt-in developer experiments only when the current evidence report names the exact flag and validation target.
- No Mac evidence, Apple Silicon label, Metal claim, Mixtral claim, production-throughput claim, portability claim, support-contract promotion, or default-on acceleration claim was added.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T1458Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T13:59Z

- Small technical slice added `CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL`, a default-off Ubuntu x86_64 dense Llama Q8_0 multi-row FFN gate/up packed-runtime matmul consumer.
- The path uses backend-owned `Q8_0RuntimeStorage::PackedRows4` for `blk.*.ffn_{gate,up}.weight`, quantizes the shared activation rows once with `q8_0_quantized_matmul_input_rows`, reuses those input Q8_0 blocks for both projections through `q8_0_packed_rows4_matmul_projection_from_quantized`, then applies the existing gated activation order.
- Fallback stays closed unless the resolved runtime plan enables the gate, the input is rank-2 prefill, Q8_0 block aligned, both packed tensors are I8 `PackedRows4`, dimensions and row grouping agree, and both output widths match.
- No duplicate packed-copy sidecar was introduced; the slice continues the backend-owned runtime-storage design.
- Validation artifact: `artifacts/cron-95495a91-20260517T1359Z-x86-ffn-gate-up-packed-rows4-matmul.txt`.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T13:18Z

- Guarded the current docs/report surface after the attention Q/K/V shared-input quantization slice without widening README support matrix, support-contract, API/frontend, platform, or performance claims.
- Confirmed `docs/CONFIGURATION.md` names `CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL` only as a default-off Ubuntu x86_64 developer experiment for dense attention Q/K/V multi-row packed-runtime matmul with exact fallback guards.
- Confirmed `docs/performance/ubuntu-x86-q8.md` keeps reproduction scoped to the current reference flags (`CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`) and keeps matrix-level Q8 GEMM/MUL_MAT as evidence-gated default-off work only.
- Did not change the project README, support contract, API/frontend surfaces, Mac/Apple Silicon/Metal notes, Mixtral language, or public performance/support claims.
- No Mac evidence, Apple Silicon label, Metal claim, Mixtral claim, production-throughput claim, portability claim, support-contract promotion, or default-on acceleration claim was added.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T1318Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T11:48Z

- Small technical slice tightened the existing default-off dense attention Q/K/V multi-row packed-runtime matmul path, `CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL`, so `try_x86_q8_attention_qkv_packed_rows4_matmul_path` quantizes the shared multi-row activation input once and reuses those Q8_0 blocks for Q, K, and V projections.
- Added `q8_0_quantized_matmul_input_rows` plus `q8_0_packed_rows4_matmul_projection_from_quantized`; the existing `q8_0_packed_rows4_matmul_projection` remains the single-projection helper used by attention-output and FFN-down packed rows4 matmul slices.
- Added exact `blocks_per_row` equality guards across Q/K/V packed runtime storage before sharing quantized input. The path still requires rank-2 input with more than one row, Q8_0 `Q8_0RuntimeStorage::PackedRows4`, I8 interleave, input width divisible by 32, output rows divisible by four, and matching runtime-plan gate. If any guard fails or the flag is unset/off, the executor falls through to the existing safe attention projection paths.
- This continues to consume backend-owned packed runtime storage only; no duplicate packed-copy sidecar was added, and no throughput/support/default-on promotion is claimed from this unit/planner slice.
- llama.cpp/Camelid grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI in `artifacts/cron-95495a91-20260517T1148Z-x86-attn-qkv-shared-input-quant.txt`.
- Local validation passed: `cargo fmt --all`, `cargo fmt --all -- --check`, `cargo test q8_attention_qkv_packed_rows4_matmul --lib`, `cargo test q8_attention_output_packed_rows4_matmul --lib`, `cargo test q8_ffn_down_packed_rows4_matmul --lib`, `cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib`, and `cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib`.
- Canonical Ubuntu x86_64 validation passed in `<ubuntu-workdir>/camelid-q8-qkv-shared-input-20260517T1148Z` via `SSH to the canonical validation host` using Rust/Cargo 1.95.0: `cargo fmt --all -- --check`, `cargo test q8_attention_qkv_packed_rows4_matmul --lib`, `cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib`, and `cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib` passed.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T11:43Z

- Guarded the current docs/report surface after the attention Q/K/V packed-rows4 matmul slice without widening README support matrix, support-contract, API/frontend, platform, or performance claims.
- Confirmed `docs/CONFIGURATION.md` names `CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL` only as a default-off Ubuntu x86_64 developer experiment for dense attention Q/K/V multi-row packed-runtime matmul with exact fallback guards.
- Confirmed `docs/performance/ubuntu-x86-q8.md` keeps reproduction scoped to the current reference flags (`CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`) and treats matrix-level Q8 GEMM/MUL_MAT as evidence-gated default-off work only.
- Scrubbed raw operator SSH key path / host IP details in backend evidence artifacts to `<operator-key>`, `<validation-host>`, and `<ubuntu-workdir>` placeholders.
- No Mac evidence, Apple Silicon label, Metal claim, Mixtral claim, production-throughput claim, portability claim, or default-on acceleration claim was added.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T1143Z-doc-claim-guard.txt`.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T09:43Z

- Guarded the current public docs/report surface after the attention-output packed-rows4 matmul slice without widening support-contract/API/frontend/performance claims.
- Narrowed only the README Ubuntu x86 Q8 acceleration subsection so it describes selected default-off experiments, explicit `CAMELID_X86_Q8_KERNEL=avx2` packed-kernel work, concrete evidence-gated matrix-level slices, and no production-throughput/default-on/platform claim.
- Confirmed `docs/CONFIGURATION.md` names `CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL` only as a default-off Ubuntu x86_64 developer experiment scoped to dense attention-output multi-row packed-runtime matmul with fallback guards.
- Confirmed `docs/performance/ubuntu-x86-q8.md` keeps reproduction scoped to the current reference flags (`CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`) and keeps matrix-level Q8 GEMM/MUL_MAT evidence-gated.
- Did not change the support matrix rows, API/frontend readiness language, Mac/Apple Silicon/Metal notes, or Mixtral language.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T0943Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T08:25Z

- Small technical slice added a directly usable multi-row dense attention-output packed-runtime matmul consumer for backend-owned Q8_0 `PackedRows4` storage, gated by new default-off flag `CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL`.
- The path is intentionally narrow: dense Llama Q8_0 `blk.*.attn_output.weight`, rank-2 input with more than one row, `Q8_0RuntimeStorage::PackedRows4`, I8 interleave, input width divisible by 32, `rectangular_role == "linear"`, and exact packed projection shape guards. If any guard fails or the flag is unset/off, `linear_for_role_runtime_with_plan` falls through to existing guarded paths and the safe linear fallback.
- The executor reuses the backend-owned packed rows4 storage and shared dot path; it does not add a duplicate packed-copy sidecar.
- ExecutionPlan now manages and pins `CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL=off` in the validated Ubuntu x86 experimental plan, and stale planner env clearing covers this gate alongside the other x86 Q8 consumer/matmul flags.
- llama.cpp/Camelid grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI in `artifacts/cron-95495a91-20260517T0825Z-x86-attn-output-packed-rows4-matmul.txt`.
- Local validation passed: `cargo fmt --all`, `cargo test q8_attention_output_packed_rows4_matmul --lib`, `cargo test resolved_runtime_plan_captures_q8_env_once --lib`, `cargo test runtime_profile_defaults_keep_experimental_q8_gates_closed --lib`, `cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib`, and `cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib`.
- Canonical Ubuntu x86_64 validation passed in `<ubuntu-workdir>/camelid-q8-slice-20260517T0825Z` via `SSH to the canonical validation host` using Rust/Cargo 1.95.0: `cargo fmt --all -- --check`, `cargo test q8_attention_output_packed_rows4_matmul --lib`, `cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib`, and `cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib` passed.
- No throughput/support promotion is claimed from this slice. It is runtime-gate/backend-owned-storage evidence for the default-off Ubuntu x86_64 experiment lane only.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T07:33Z

- Guarded the current docs/report surface after the packed-rows4 FFN-down matmul planner slice without widening README/support-contract/API/frontend/performance claims.
- Confirmed `docs/CONFIGURATION.md` names `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL` only as a default-off Ubuntu x86_64 developer experiment scoped to dense FFN-down multi-row packed-runtime matmul with fallback guards.
- Confirmed `docs/performance/ubuntu-x86-q8.md` keeps reproduction scoped to the current reference flags (`CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`) and keeps matrix-level Q8 GEMM/MUL_MAT evidence-gated.
- Did not change the project README, support contract, API/frontend surfaces, Mac/Apple Silicon/Metal notes, or Mixtral language.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T0733Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T06:55Z

- Small technical slice completed planner/docs ownership for `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL`, the default-off dense FFN-down multi-row packed-runtime matmul experiment.
- The existing runtime path remains directly usable only for `ffn_down` with backend-owned `Q8_0RuntimeStorage::PackedRows4`, I8 interleave, Q8_0 weights, matching input/output dimensions, 32-wide input blocks, and output rows divisible by four; otherwise it falls back to the safe linear path.
- `ExecutionPlan` now manages stale-env clearing and pins `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL=off` in the validated Ubuntu x86 experimental plan, matching the default-off gate policy for the decode-consumer family.
- Refreshed llama.cpp evidence confirms dense Ubuntu x86_64 Q8_0 should keep moving toward directly usable backend-owned packed/runtime storage or x86-specific matmul: llama.cpp's CPU_REPACK Q8_0 selection has NEON/RISC-V branches but no AVX2/AVX512 x86 branch to mirror wholesale.
- Local validation passed: `cargo fmt --all`, `cargo test q8_ffn_down_packed_rows4_matmul --lib`, `cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib`, and `cargo test planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib`.
- Canonical Ubuntu x86_64 validation passed in `<ubuntu-workdir>/camelid-x86-q8-packed-rows4-matmul-20260517T0655Z` via `SSH to the canonical validation host` using Rust/Cargo 1.95.0: `cargo fmt --all -- --check` and the same three targeted test commands passed.
- Artifact: `artifacts/cron-95495a91-20260517T0655Z-x86-ffndown-packed-rows4-matmul-planner.txt`.
- No throughput/support promotion is claimed from this slice. It is planner/runtime-gate evidence for the default-off Ubuntu x86_64 experiment lane only.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T05:11Z

- Guarded the current docs/report surface after the attention-output consumer slice without widening README/support-contract/API/frontend/performance claims.
- Added the new `CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER` only as a default-off Ubuntu x86_64 developer experiment in `docs/CONFIGURATION.md`, backed by the 2026-05-17T05:03Z evidence entry.
- Tightened `docs/performance/ubuntu-x86-q8.md` wording to separate attention Q/K/V from attention-output decode consumers and keep all decode-consumer paths opt-in.
- Scrubbed the latest report entry to use `<ubuntu-workdir>`, `<operator-key>`, and `<validation-host>` placeholders instead of raw operator SSH/IP details.
- Removed a placeholder matrix-owner flag from the recommended future slice wording; matrix-level Q8 GEMM/MUL_MAT remains directional until a concrete default-off Ubuntu x86_64 path has fresh evidence.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T0511Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T05:03Z

- Small technical slice added a directly usable one-row dense attention-output decode consumer for backend-owned packed Q8_0 runtime storage, gated by new default-off flag `CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER`.
- The path is intentionally narrow: dense Llama Q8_0 `blk.*.attn_output.weight`, one input row, `Q8_0RuntimeStorage::PackedRows4`, I8 interleave, input width divisible by 32, and `rectangular_role == "linear"`. If any guard fails or the flag is unset/off, `linear_for_role_runtime_with_plan` falls back to the existing safe linear path.
- Runtime storage ownership was extended by packing `*.attn_output.weight` in directly consumable output-row order when `CAMELID_X86_Q8_REPACK=on`; no duplicate packed-copy sidecar was added.
- ExecutionPlan now manages and pins `CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER=off` in the validated Ubuntu x86 experimental plan, alongside the other default-off Q8 decode-consumer flags.
- llama.cpp/Camelid grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI in `artifacts/cron-95495a91-20260517T0503Z-x86-attn-output-consumer.txt`.
- Local validation passed: `cargo fmt --all`, `cargo test q8_attention_output_consumer --lib`, `cargo test q8_x86_repack --lib`, `cargo test resolved_runtime_plan_captures_q8_env_once --lib`, `cargo test runtime_profile_defaults_keep_experimental_q8_gates_closed --lib`, and `cargo test ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib`.
- Canonical Ubuntu x86_64 validation passed in `<ubuntu-workdir>/camelid-x86-q8-attn-output-consumer-20260517T0503Z` via `SSH to the canonical validation host` using Rust/Cargo 1.95.0: `cargo fmt --all -- --check` and the same five targeted test commands passed.
- No throughput/support promotion is claimed from this slice. It is runtime-gate/backend-owned-storage evidence for the default-off Ubuntu x86_64 experiment lane only.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T03:36Z

- Guarded `docs/performance/ubuntu-x86-q8.md` again against over-reading matrix-level Q8 GEMM/MUL_MAT as implemented owner/performance/support evidence.
- Kept the reproduction block scoped to only the current reference default-off Ubuntu x86_64 experiment gates: `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`.
- Clarified that matrix-level Q8 GEMM/MUL_MAT remains evidence-gated directional work until a fresh Ubuntu x86_64 slice proves a concrete default-off flag/path.
- Did not change the project README, support contract, API/frontend surfaces, Mac/Apple Silicon/Metal notes, or Mixtral language.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T0336Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T03:20Z

- Small technical slice completed ExecutionPlan ownership of the default-off Ubuntu x86 Q8 decode-consumer gate set by adding `CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER` to managed stale-env clearing and the validated experimental-plan off pins.
- The older `CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER` key remains cleared for compatibility, but the directly usable FFN-down packed runtime path reads the `CONSUMER` gate through `ResolvedRuntimePlan`; this closes the planner hygiene gap without adding a duplicate packed-copy sidecar.
- The current validated Ubuntu x86 experimental plan still requires explicit `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`, preserves safe fallback otherwise, and pins attention QKV/projection, FFN gate/up, FFN down, and output decode-owner experiments to `off` unless a fresh slice explicitly validates a narrower gate.
- llama.cpp/Camelid grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP/GOMP, AVX2, AVX512, and VNNI in `artifacts/cron-95495a91-20260517T0320Z-x86-ffndown-consumer-planner.txt`.
- Local validation passed: `cargo fmt --check`, `cargo test -q planner_env_apply_clears_stale_x86_q8_decode_consumer_flags --lib`, `cargo test -q ubuntu_experimental_validated_gates_select_rust_avx2_q8_path --lib`, `cargo test -q resolved_runtime_plan_captures_q8_env_once --lib`, and `cargo test -q q8_ffn_down_consumer --lib`.
- Canonical Ubuntu x86_64 validation passed in `<ubuntu-workdir>/camelid-x86-q8-planner-ffndown-consumer-20260517T0320Z` via `SSH to the canonical validation host` using Rust 1.90.0: same fmt and four targeted test commands passed.
- No throughput/support promotion is claimed from this slice. It is planner/runtime-gate evidence for the default-off Ubuntu x86_64 experiment lane only.

## CAMELID DOCS UBUNTU X86 Q8 — cron 5e4b0b83, 2026-05-17T00:43Z

- Guarded docs/report wording against unsupported Ubuntu x86 Q8 claims without changing the project README, support contract, API/frontend surfaces, Mac/Apple Silicon/Metal notes, or Mixtral language.
- Corrected `docs/performance/ubuntu-x86-q8.md` reproduction wording to use only the current reference default-off experiment gate shape: `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`.
- Removed stale/non-current sketch flags from that reproduction command (`CAMELID_X86_Q8_REPACK_TENSORS`, `CAMELID_X86_Q8_GEMM`, `CAMELID_X86_Q8_PACKED_TILE16`, `CAMELID_X86_Q8_PACKED_TILE16_SERIAL_OWNER`, and `CAMELID_X86_Q8_KERNEL=avx2_scaled_rowdot`) because they are not the current Q8 reference truth for this lane.
- Kept the language scoped to the Ubuntu x86_64 dense Llama Q8_0 experiment: default-off developer flags only, safe fallback preserved, no Mac evidence, no Apple Silicon labels, no Metal, no Mixtral, no production-throughput or support-contract promotion.
- Audit artifact: `artifacts/cron-5e4b0b83-20260517T0043Z-doc-claim-guard.txt`.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-17T00:35Z

- Small technical slice hardened ExecutionPlan ownership of the default-off Ubuntu x86 Q8 decode-consumer gates: `CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER`, `CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER`, and `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER` are now managed alongside the existing x86 repack/kernel/output/FFN-down gates.
- The current validated Ubuntu x86 experimental plan still requires explicit `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2`, preserves safe fallback otherwise, and now pins attention QKV/projection plus FFN gate/up decode consumers to `off` so stale opt-in owner experiments cannot leak into planned runs.
- This avoids the failed duplicate packed-copy sidecar direction: it does not add a new sidecar and leaves the directly usable x86 paths consuming backend-owned `Q8_0RuntimeStorage::PackedRows4` when separately opted in.
- llama.cpp/Camelid grep evidence was refreshed for `q8_0`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI in `artifacts/cron-95495a91-20260517T0035Z-x86-managed-decode-consumer-flags.txt`.
- Local validation passed: `cargo fmt --all -- --check`, targeted runtime/plan tests, and `./scripts/with-rustup-cargo.sh test execution_plan::tests --lib -- --nocapture` (`13 passed`).
- No throughput/support promotion is claimed from this slice. It is planner/runtime-gate evidence for the default-off Ubuntu x86_64 experiment lane only.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-16T01:36Z

- Small technical slice added a directly usable one-row dense FFN gate/up decode consumer for backend-owned packed Q8_0 runtime storage, gated by the new default-off x86 flag `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER`.
- The path is intentionally narrow: dense Llama Q8_0 `blk.*.ffn_gate.weight` plus `blk.*.ffn_up.weight`, one activation row, runtime-packed `Q8_0RuntimeStorage::PackedRows4`, I8 interleave, input width divisible by 32, matching gate/up output widths divisible by 4. If any guard fails or the env flag is unset/off, `gated_ffn_activation_with_plan` falls back to the existing safe gate/up path.
- This avoids the failed duplicate packed-copy sidecar direction: it consumes backend-owned packed/runtime storage attached to the two FFN tensors and does not add a row-major+packed duplicate as the final design.
- llama.cpp/Camelid grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI in `artifacts/cron-95495a91-20260516T0136Z-x86-ffn-gate-up-consumer.txt`.
- Canonical Ubuntu x86_64 validation passed in `<ubuntu-workdir>/camelid-ffngateup-consumer-20260516T0136Z` on `ubuntu@<validation-host>`: `cargo fmt --check`, `cargo test -q q8_ffn_gate_up_consumer --lib` (`2 passed`), and `cargo test -q --lib` (`245 passed`). Output: `artifacts/cron-95495a91-20260516T0136Z-x86-ffn-gate-up-consumer-tests.txt`.
- No throughput/support promotion is claimed from this slice. It is parity/unit evidence for a default-off Ubuntu x86_64 experiment path only.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-15T19:33Z

- Small technical slice added a directly usable one-row decode output-projection consumer for backend-owned packed Q8_0 runtime storage, gated by the new default-off x86 flag `CAMELID_X86_Q8_OUTPUT_DECODE_OWNER`.
- The path is intentionally narrow: dense Llama Q8_0 `output.weight`, token-major output projection, one activation row, runtime-packed `Q8_0RuntimeStorage::PackedRows4`, I8 interleave, input width divisible by 32, vocab rows divisible by 4. If any guard fails or the env flag is unset/off, `output_projection_with_layout` falls back to the existing borrowed transposed matmul path.
- This avoids the failed duplicate packed-copy sidecar direction: it consumes the backend-owned packed/runtime storage already attached to `output.weight` by `CAMELID_X86_Q8_REPACK=on` and does not add a row-major+packed duplicate as the final design.
- llama.cpp/Camelid grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling/thread/OpenMP/GOMP, AVX2, AVX512, and VNNI in `artifacts/cron-95495a91-20260515T1933Z-x86-output-decode-owner.txt`.
- Canonical Ubuntu x86_64 validation passed in `<ubuntu-workdir>/camelid-output-decode-owner-20260515T193322Z` on `ubuntu@<validation-host>`: `cargo fmt --check`, `cargo test --lib x86_q8 -- --nocapture` (`4 passed`, including `x86_q8_output_decode_owner_path_uses_runtime_packed_storage`), and `cargo test --test tensor_store x86_q8_repack_loads_output_projection_as_token_major_packed_runtime -- --nocapture` (`1 passed`). Output: `artifacts/cron-95495a91-20260515T1933Z-x86-output-decode-owner-tests.txt`.
- No throughput/support promotion is claimed from this slice. It is parity/unit evidence for a default-off Ubuntu x86_64 experiment path only.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-15T17:59Z

- Small technical slice added a directly usable decode-time FFN-down consumer for backend-owned packed Q8_0 runtime storage, gated by the new default-off x86 flag `CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER`.
- The path is intentionally narrow: dense Llama Q8_0 `ffn_down`, one activation row, runtime-packed `Q8_0RuntimeStorage::PackedRows4`, I8 interleave, input width divisible by 32, output width divisible by 4. If any guard fails or the env flag is unset/off, `linear_for_role_runtime` falls back to the existing path.
- This avoids the failed duplicate packed-copy sidecar direction: it consumes the backend-owned packed/runtime storage already attached to the tensor and does not add a row-major+packed duplicate as the final design.
- llama.cpp/Camelid grep evidence was refreshed again for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI in `artifacts/cron-95495a91-20260515T1759Z-x86-ffn-down-decode-owner-grep.txt`.
- Canonical Ubuntu x86_64 validation passed in `<ubuntu-workdir>/camelid-ffndown-owner-20260515T1759Z` on `ubuntu@<validation-host>`: `cargo test --lib q8_0_runtime_packed -- --nocapture` (`5 passed`) and `cargo test --lib x86_q8 -- --nocapture` (`3 passed`, including `x86_q8_ffn_down_decode_owner_path_matches_runtime_packed_baseline`). Output: `artifacts/cron-95495a91-20260515T1759Z-x86-ffn-down-decode-owner-tests.txt`.
- No throughput/support promotion is claimed from this slice. It is parity/unit evidence for a default-off Ubuntu x86_64 experiment path only.

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-15T12:35Z

- Small follow-on slice widened the default-off `CAMELID_X86_Q8_REPACK=on` runtime-packed loader to include dense Llama `blk.*.ffn_down.weight` in backend-owned `Q8_0RuntimeStorage::PackedRows4`.
- The new FFN-down case packs the GGUF descriptor shape `[ffn, hidden]` as directly consumable transposed runtime rows `[hidden, ffn]`, matching the existing `linear_for_role_runtime` hot path without retaining `data`, `q8_0_blocks`, file backing, or debug packed sidecars.
- Fallback is unchanged: with the x86 repack env unset/off, `CAMELID_Q8_0_BLOCK_DOT=off`, unaligned shapes, or tensors outside the selected x86 allowlist, the existing safe/load paths remain in force.
- llama.cpp grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI; selected hits plus implementation evidence are captured in `artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt`.
- Canonical Ubuntu x86_64 validation passed: `<ubuntu-cargo> test --test tensor_store x86_q8_repack_loads_dense_ffn_family_as_transposed_packed_runtime -- --nocapture` in a synchronized scratch checkout (`1 passed; 0 failed; 23 filtered out`).

## CAMELID BACKEND ENGINEER UBUNTU X86 Q8 — cron 95495a91, 2026-05-15T11:08Z

- Small follow-on slice widened the default-off `CAMELID_X86_Q8_REPACK=on` runtime-packed loader from `blk.*.attn_q.weight` to the dense attention projection family: `blk.*.attn_q.weight`, `blk.*.attn_k.weight`, `blk.*.attn_v.weight`, and `blk.*.attn_output.weight`.
- The implementation still uses backend-owned `Q8_0RuntimeStorage::PackedRows4` for selected tensors and keeps `data`, `q8_0_blocks`, file backing, and debug packed sidecars empty/absent for that path.
- Fallback is unchanged: with the x86 repack env unset/off, or for tensors outside the selected x86 allowlist, the existing safe/load paths remain in force.
- llama.cpp grep evidence was refreshed for `q8_0`, `tinyBLAS`, `ggml_vec_dot_q8_0_q8_0`, `repack`, `MUL_MAT`, scheduling, OpenMP, AVX2, AVX512, and VNNI; selected hits are captured in `artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`.
- Canonical Ubuntu x86_64 validation passed: `cargo fmt --check`, `cargo test -q x86_q8_repack_loads_dense_attention_family_as_packed_runtime --test tensor_store`, and `cargo test -q x86_q8_avx2_packed_rows4_i8_matches_scalar_dot --lib` using the installed Rust 1.90.0 toolchain because the host default cargo is too old for the lockfile/MSRV.

## CAMELID TPM UBUNTU X86 Q8 handoff — cron 0719640b, 2026-05-14T22:49Z

- CAMELID TPM UBUNTU X86 Q8: Active evidence root remains this directory; latest recheck artifact is `artifacts/cron-0719640b-20260514T2249Z-verification.txt`.
- CAMELID TPM UBUNTU X86 Q8: Canonical host path was re-verified as Ubuntu x86_64 on Intel Xeon Platinum 8488C with 16 vCPUs and AVX2/AVX512/VNNI/AMX hardware flags; this lane claims only the measured Ubuntu CPU path.
- CAMELID TPM UBUNTU X86 Q8: llama.cpp evidence still shows Release CPU build with `GGML_CPU_REPACK=ON`, `GGML_OPENMP=ON`, AVX/AVX2/F16C/FMA on, and AVX512/VNNI/AMX build gates off for the measured binary.
- CAMELID TPM UBUNTU X86 Q8: perf proof remains `artifacts/perf-bench-pp-symbols.txt`, dominated by `tinyBLAS_Q0_AVX<block_q8_0, block_q8_0>::gemm4xN<2>` via `llamafile_sgemm`/`ggml_compute_forward_mul_mat`; this proves the actual measured win is tiled Q8_0 MUL_MAT + OpenMP scheduling, not an AVX512/VNNI/AMX kernel.
- CAMELID TPM UBUNTU X86 Q8: same-host llama.cpp and Camelid benchmark artifacts remain under `benchmarks/`; Camelid baseline/default-parallel/parallel-off retained-block microbench stayed ~16 ms with equal checksum, while the bounded default-off `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2` API smoke cut first-token wall from 147425.30 ms to 75650.18 ms and kept token id 8586.
- CAMELID TPM UBUNTU X86 Q8: bounded safe port slice is commit `80f6271` in `src/tensor/mod.rs`, `src/inference.rs`, and `tests/tensor_store.rs`; current Q8 path remains fallback when env gates are absent/off or AVX2 is unavailable.
- CAMELID TPM UBUNTU X86 Q8: remaining promotion gap is full end-to-end Camelid API throughput equivalence against llama.cpp; next owner should extend the default-off packed/tiled Q8 GEMM architecture to FFN down and more dense linears only after Ubuntu x86 parity/perf evidence per tensor family.

## Repositories and status

- Camelid local worktree: `main...origin/main [ahead 1]` with pre-existing unrelated dirty files; this active Ubuntu x86 evidence root records only the `UBUNTU_X86_Q8` findings/slice and should not be used as evidence for other platforms. This lane touched the x86 Q8 implementation files plus this evidence bundle; unrelated dirty evidence from other lanes was left unstaged.
- llama.cpp local/remote reference: `3e037f313c2c4cfce897d9be8f43954283a61de1` (`version: 9158`, commit `HIP: RDNA3 mma FA, faster AMD transpose, tune AMD (#22880)`).
- Canonical host: `ubuntu@<validation-host>`, AWS Ubuntu 24.04 x86_64, Intel Xeon Platinum 8488C, 16 vCPUs.
- Model: `<ubuntu-model-path>/Llama-3.2-3B-Instruct-Q8_0.gguf`.

Evidence:
- `artifacts/cron-95495a91-20260515T1933Z-x86-output-decode-owner.txt`
- `artifacts/cron-95495a91-20260516T0136Z-x86-ffn-gate-up-consumer.txt`
- `artifacts/cron-95495a91-20260516T0136Z-x86-ffn-gate-up-consumer-tests.txt`
- `artifacts/cron-95495a91-20260515T1933Z-x86-output-decode-owner-tests.txt`
- `artifacts/cron-95495a91-20260515T1759Z-x86-ffn-down-decode-owner-grep.txt`
- `artifacts/cron-95495a91-20260515T1759Z-x86-ffn-down-decode-owner-tests.txt`
- `artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt`
- `artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`
- `artifacts/cron-0719640b-20260514T2249Z-verification.txt`
- `artifacts/ubuntu-host-repos-models.txt`
- `artifacts/ubuntu-llamacpp-build-symbols.txt`
- `artifacts/llamacpp-git-grep.txt`
- `artifacts/llamacpp-git-grep-full.txt`
- `artifacts/llamacpp-repack-selected-source.txt`
- `artifacts/camelid-x86-repack-tests.txt`
- `artifacts/camelid-x86-repack-build.txt`

## llama.cpp Ubuntu x86 Q8_0 path findings

### Actual compiled capabilities on the canonical host

CPU hardware exposes AVX2, AVX512, AVX_VNNI, AVX512_VNNI, and AMX (`amx_int8`, `amx_bf16`, `amx_tile`), but the llama.cpp build used for evidence is narrower:

- `GGML_CPU_REPACK=ON`
- `GGML_OPENMP=ON`
- `GGML_AVX=ON`, `GGML_AVX2=ON`, `GGML_F16C=ON`, `GGML_FMA=ON`
- `GGML_AVX512=OFF`
- `GGML_AVX_VNNI=OFF`, `GGML_AVX512_VNNI=OFF`
- `GGML_AMX_INT8=OFF`, `GGML_AMX_BF16=OFF`, `GGML_AMX_TILE=OFF`
- `llama-cli` links `libgomp.so.1`.

Runtime `llama-server` system info likewise reported `AVX2 = 1`, `LLAMAFILE = 1`, `OPENMP = 1`, `REPACK = 1`, with no AVX512/VNNI/AMX runtime path in this build.

### Source map

Key source locations in current llama.cpp:

- `ggml/src/ggml-cpu/arch/x86/quants.c`
  - `quantize_row_q8_0`
  - `ggml_vec_dot_q8_0_q8_0`
- `ggml/src/ggml-cpu/ggml-cpu.c`
  - Q8_0 trait wiring: `.from_float = quantize_row_q8_0`, `.vec_dot = ggml_vec_dot_q8_0_q8_0`
  - `GGML_OP_MUL_MAT` scheduling/compute dispatch
- `ggml/src/ggml-cpu/ggml-cpu.cpp`
  - CPU backend extra buffer registration including `GGML_USE_CPU_REPACK`
- `ggml/src/ggml-cpu/repack.cpp` / `repack.h`
  - Q8_0 repack layouts and generic `ggml_gemv/gemm_q8_0_*_q8_0` hooks
  - graph rewrite hooks for `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID`
- `ggml/src/ggml-cpu/llamafile/sgemm.cpp`
  - `tinyBLAS_Q0_AVX`, the observed hot prompt-processing kernel

### Perf proof of actual Ubuntu path

Perf evidence is from the canonical host against the Q8_0 model with CPU-only llama.cpp.

Best hot-symbol run: `artifacts/perf-bench-pp-symbols.txt`

Top path:

```text
88.46% libggml-cpu.so.0.11.1  tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>
       tinyBLAS_Q0_AVX<...>::mnpack
       llamafile_sgemm
       ggml_compute_forward_mul_mat
       ggml_graph_compute_thread.isra.0
       GOMP_parallel / ggml_graph_compute / llama_context::process_ubatch
```

Other selected symbols:

- `quantize_row_q8_0`: 0.71%
- `ggml_vec_dot_q8_0_q8_0`: 0.43%
- `libgomp.so.1`: present in hot samples

Interpretation: for the measured Ubuntu x86_64 prompt-processing path, the dense Q8_0 hot loop is not AVX512/VNNI/AMX. It is AVX2-era llamafile/tinyBLAS Q8_0 x Q8_0 through `GGML_OP_MUL_MAT`, with OpenMP/GOMP scheduling. Repack support is compiled in and source-visible, but the selected hot evidence is dominated by `tinyBLAS_Q0_AVX`, not repack `q8_0_4x4/4x8/16x1` symbols.

Perf caveat: kernel symbols were restricted by host perf settings; user-space symbols were sufficient for this lane. See `artifacts/perf_event_paranoid.txt`, `artifacts/perf-bench-pp.stderr`, and `artifacts/perf-run.stderr`.

## Benchmarks

### llama.cpp same-host CPU-only

Files:
- `benchmarks/llama-bench-t16-p128-n16.json`
- `benchmarks/llama-bench-t1-p128-n16.json`

| Mode | Prompt processing | Token generation |
|---|---:|---:|
| llama.cpp `-t 16`, `p=128`, `n=16` | 90.421 tok/s | 25.635 tok/s |
| llama.cpp `-t 1`, `p=128`, `n=16` | 12.168 tok/s | 2.670 tok/s |

### Camelid Q8 hot-path same-host microbench

Command shape: `target/release/camelid bench-q8-blocks <ubuntu-model-path>/Llama-3.2-3B-Instruct-Q8_0.gguf --tensor blk.0.ffn_gate.weight --swap-rank2-shape --repeats 5 --warmup 1 --all-rows-dot --single-input-row-dot`

Files:
- `benchmarks/baseline.json`
- `benchmarks/parallel_on.json`
- `benchmarks/parallel_off.json`
- `benchmarks/avx2.json`

| Camelid mode | avg all-row Q8 dot | avg single-input-row Q8 dot | checksum |
|---|---:|---:|---:|
| baseline env | 16.109 ms | 16.051 ms | `-0.05126936` |
| `CAMELID_PARALLEL_LINEAR=on` | 16.152 ms | 16.073 ms | `-0.05126936` |
| `CAMELID_PARALLEL_LINEAR=off` | 16.049 ms | 16.093 ms | `-0.05126936` |
| `CAMELID_X86_Q8_KERNEL=avx2` | 16.102 ms | 16.058 ms | `-0.05126936` |

Interpretation: the bounded AVX2 scalar-block replacement is parity-clean but does not materially improve the retained-block microbench by itself. That is expected: llama.cpp’s win is primarily a wider tiled GEMM/MUL_MAT architecture with tinyBLAS/OpenMP scheduling, not only a faster single 32-byte dot primitive.

### Camelid same-host API smoke benchmark: default vs x86 runtime repack

Command shape: `CAMELID_BIN=target/release/camelid node scripts/bench-unique-chat.mjs --start-backend --model <ubuntu-model-path>/Llama-3.2-3B-Instruct-Q8_0.gguf --max-tokens 1 --repeats 1 --warmup 0`.

Files:
- `benchmarks/unique-chat-baseline-1tok.json`
- `benchmarks/unique-chat-x86-repack-avx2-1tok.json`

| Camelid API mode | output token text | avg wall | avg generate | avg layers | attention projections | FFN gate | FFN up | FFN down | FFN total | RSS after first token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline env | `Here` | 147425.30 ms | 144962.00 ms | 144515.97 ms | 36277.35 ms | 35464.56 ms | 35772.57 ms | 36891.29 ms | 108140.19 ms | 3808.82 MiB |
| `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2` | `Here` | 75650.18 ms | 72077.00 ms | 71463.04 ms | 24606.97 ms | 5174.13 ms | 5127.87 ms | 36484.12 ms | 46796.54 ms | 3836.46 MiB |

Interpretation: this is a one-request Ubuntu x86_64 smoke benchmark, not a production throughput or support-contract claim. It demonstrates parity for the measured first token (`Here`) and materially reduced the gate/up timings for the earlier default-off x86 runtime-repacked path captured in these benchmark files. The later FFN-down loader/runtime-storage widening has test evidence above, but no FFN-down performance measurement is claimed from this smoke run.

Full llama.cpp-vs-Camelid API harness note: `scripts/bench-llama3-same-host.mjs` was previously attempted with `max_tokens=8`, `repeats=2`, `threads=16`; the Camelid side did not produce measured output before the run was killed after several minutes. This README therefore keeps llama.cpp `llama-bench`, Camelid microbench, and Camelid API default-vs-repack smoke as separate same-host evidence, not a full end-to-end API throughput equivalence claim against llama.cpp.

## Camelid bounded default-off port slice

Implemented in `src/tensor/mod.rs`, `src/inference.rs`, `src/execution_plan.rs`, `docs/CONFIGURATION.md`, and `tests/tensor_store.rs`:

- `CAMELID_X86_Q8_REPACK=on` is a default-off GGUF load/read gate for selected Llama dense Q8 linears in this slice (`blk.*.attn_q.weight`, `blk.*.attn_k.weight`, `blk.*.attn_v.weight`, `blk.*.attn_output.weight`, `blk.*.ffn_gate.weight`, `blk.*.ffn_up.weight`, `blk.*.ffn_down.weight`, `output.weight`).
- When the gate is on, `TensorStore::{load_q8_0_file_backed_linear,load_q8_0_block_backed_linear}` build `Q8_0RuntimeStorage::PackedRows4` directly from GGUF Q8_0 bytes and return a tensor with empty `data`, no `q8_0_blocks`, and no file-backed row-major sidecar for those selected tensors.
- FFN gate/up/down descriptor shapes are packed in runtime output-row order so `linear_for_role_runtime` consumes the backend-owned packed storage directly.
- `CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER=on` is a second default-off experiment gate for decode-time `ffn_down` to consume `Q8_0RuntimeStorage::PackedRows4` directly through `try_x86_q8_ffn_down_decode_owner_path`; it falls back unless the tensor/shape/interleave guards match exactly.
- `CAMELID_X86_Q8_OUTPUT_DECODE_OWNER=on` is a default-off experiment gate for one-row decode `output.weight` to consume `Q8_0RuntimeStorage::PackedRows4` directly through `try_x86_q8_output_decode_owner_path`; it falls back unless the tensor/shape/interleave guards match exactly.
- `CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 experiment gate for multi-row dense attention Q/K/V to consume backend-owned `Q8_0RuntimeStorage::PackedRows4` through `try_x86_q8_attention_qkv_packed_rows4_matmul_path`; it falls back unless all three Q/K/V weights are runtime-packed Q8_0, dimensions match, row count is greater than one, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 experiment gate for multi-row dense attention-output to consume backend-owned `Q8_0RuntimeStorage::PackedRows4` through the guarded attention-output packed-runtime matmul path; it falls back unless the runtime plan, tensor type, dimensions, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 experiment gate for multi-row `output.weight` to consume backend-owned `Q8_0RuntimeStorage::PackedRows4` through the guarded output packed-runtime matmul path; it falls back unless the runtime plan, exact tensor name, Q8_0 tensor type, dimensions, row grouping, and packed interleave guards match exactly. Current evidence is local parity/gate only; no fresh canonical Ubuntu timing/profiling bundle is attached to this slice.
- `CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 experiment gate for multi-row dense FFN gate/up to consume backend-owned `Q8_0RuntimeStorage::PackedRows4` with one shared input quantization for both projections; it falls back unless both tensors, the runtime plan, dimensions, row count, row grouping, and packed interleave guards match exactly.
- The one-row packed-runtime decode consumers now share a guarded output-group helper; wide output projections can parallelize independent rows4 groups inside the existing default-off gates, but this local helper slice has no Ubuntu timing claim yet.
- `x86_q8_kernel_avx2_enabled()` reads `CAMELID_X86_Q8_KERNEL` and accepts `avx2/on/1/true` (case variants included).
- `q8_0_i8_block_avx2()` and `q8_0_packed_4x8_block_avx2()` are `#[target_feature(enable = "avx2")]` and default-off behind both the env gate and `std::arch::is_x86_feature_detected!("avx2")`.
- Existing path fallback is preserved when the env gates are absent/off or AVX2 is not detected.
- Unit tests: `x86_q8_avx2_kernel_matches_scalar_dot`, `x86_q8_avx2_packed_rows4_i8_matches_scalar_dot`, `x86_q8_repack_loads_attn_q_as_packed_runtime_without_row_major_duplicate`, `x86_q8_repack_loads_dense_attention_family_as_packed_runtime`, `x86_q8_repack_loads_dense_ffn_family_as_transposed_packed_runtime`.

Validation:

- Ubuntu x86_64 test pass: `artifacts/camelid-x86-repack-tests.txt`
- Ubuntu x86_64 release build pass: `artifacts/camelid-x86-repack-build.txt`
- Same-host microbench parity: all retained-block Camelid modes had identical `dot_checksum = -0.05126936`.
- Same-host API smoke parity: baseline and `CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2` both emitted first-token text `Here` for the measured Ubuntu x86_64 prompt; timings are in `benchmarks/unique-chat-*.json`. Those timings cover the earlier measured gate/up runtime-repacked slice and are not evidence for FFN-down throughput.
- Ubuntu x86_64 attention Q/K/V packed-rows4 matmul test pass: `artifacts/cron-95495a91-20260517T1013Z-x86-attn-qkv-packed-rows4-matmul.txt`; this is a unit/planner evidence slice only and does not claim throughput improvement.
- Ubuntu x86_64 FFN gate/up packed-rows4 matmul test pass: `artifacts/cron-95495a91-20260517T1359Z-x86-ffn-gate-up-packed-rows4-matmul.txt`; this is unit/planner/runtime-gate evidence only and does not claim throughput improvement.
- Local-only x86 packed-rows4 decode output-group parallel helper evidence: `artifacts/cron-1eeef0a5-20260517T1744Z-x86-packed-rows4-decode-output-parallel-local.txt`; Ubuntu timing/profiling remains pending and no measured effect is retained.
- Non-Ubuntu test gates are not claimed here. The Ubuntu x86_64 slice compiled and passed in `/tmp/camelid-ubuntu-x86-q8-20260514T2221Z` on the canonical host.

This slice intentionally avoids a performance-mode row-major+packed duplicate for the selected runtime-packed tensors. Existing opt-in debug/parity sidecars remain separate gates.

## Pass/fail table

| Requirement | Result | Evidence |
|---|---|---|
| Verify git status before edits / avoid clobber | PASS | status checked before and after; unrelated dirty files left unstaged |
| Map current llama.cpp x86 Q8_0 source path | PASS | `artifacts/llamacpp-git-grep*.txt`, `artifacts/llamacpp-repack-selected-source.txt` |
| Prove actual Ubuntu build flags | PASS | `artifacts/ubuntu-llamacpp-build-symbols.txt` |
| Prove actual hot symbols | PASS | `artifacts/perf-bench-pp-symbols.txt` |
| Benchmark llama.cpp same host | PASS | `benchmarks/llama-bench-t16-p128-n16.json`, `benchmarks/llama-bench-t1-p128-n16.json` |
| Benchmark Camelid baseline/default-parallel/parallel-off | PASS (microbench) | `benchmarks/baseline.json`, `parallel_on.json`, `parallel_off.json` |
| Implement bounded default-off x86 slice | PASS / local-only for latest output matmul | `src/tensor/mod.rs`, `src/inference.rs`, `src/execution_plan.rs`, `docs/CONFIGURATION.md`, `tests/tensor_store.rs`; reference retained env remains `CAMELID_X86_Q8_REPACK=on` plus `CAMELID_X86_Q8_KERNEL=avx2`; narrow consumer/matmul flags are separate default-off developer experiments only when named by a slice. Ubuntu-validated/default-off evidence includes attention-family loader evidence in `artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`; FFN-down runtime-storage evidence in `artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt`; default-off FFN-down decode-owner evidence in `artifacts/cron-95495a91-20260515T1759Z-x86-ffn-down-decode-owner-tests.txt`; default-off output decode-owner evidence in `artifacts/cron-95495a91-20260515T1933Z-x86-output-decode-owner-tests.txt`; default-off attention Q/K/V packed-rows4 matmul evidence in `artifacts/cron-95495a91-20260517T1013Z-x86-attn-qkv-packed-rows4-matmul.txt`; default-off attention-output packed-rows4 matmul evidence in `artifacts/cron-95495a91-20260517T0825Z-x86-attn-output-packed-rows4-matmul.txt`; default-off FFN gate/up packed-rows4 matmul evidence in `artifacts/cron-95495a91-20260517T1359Z-x86-ffn-gate-up-packed-rows4-matmul.txt`. The `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL` slice currently has local-only parity/gate evidence in `artifacts/cron-1eeef0a5-20260517T1850Z-x86-output-packed-rows4-matmul-local.txt` because no fresh canonical Ubuntu timing/profiling bundle is attached to that slice; it is not Ubuntu throughput/support/default-on evidence. |
| Parity test on Ubuntu x86_64 | PASS | `artifacts/camelid-x86-repack-tests.txt`; microbench checksum parity; API first token `Here` in both JSON files |
| Demonstrate performance movement from bounded measured slice | PASS (bounded smoke) | `benchmarks/unique-chat-baseline-1tok.json` vs `unique-chat-x86-repack-avx2-1tok.json`; gate/up timings and total first-token wall time reduced in the one-request Ubuntu x86_64 API smoke; no FFN-down, production-throughput, portability, or support-contract claim |
| Full end-to-end Camelid API vs llama.cpp API | BLOCKED / partial | llama.cpp-vs-Camelid API harness did not complete promptly; this bundle has llama.cpp bench plus Camelid default-vs-repack API smoke, not API equivalence vs llama.cpp |

## Recommended next slice

Continue toward the actual winning llama.cpp architecture without widening support claims:

1. Keep widening the default-off x86 runtime-packed path only with Ubuntu x86 parity/bench evidence per tensor family; FFN down now has loader/runtime-storage coverage but still needs performance measurement.
2. Add any future tiled Q8_0 matmul/GEMM path only under a concrete, newly validated default-off Ubuntu x86_64 experiment gate; do not cite a placeholder matrix-owner flag as current evidence. Consider `avx512_vnni` only after rebuilding/benchmarking llama.cpp with VNNI enabled for comparison.
3. Tile over multiple output rows and input blocks, quantize the f32 activation row once to Q8_0, and amortize env/dispatch outside the innermost 32-byte block loop.
4. Add perf evidence on Ubuntu x86 before claiming broader speedup: hot symbols should move from scalar Rust loops toward a tiled x86 kernel, with unchanged checksums/output tokens.
