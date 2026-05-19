# Ubuntu x86 Q8 Performance Work

## Status

This is an active, evidence-gated performance lane. Optimized paths are default-off while validation continues.

The current goal is production-directional runtime improvement on a narrow Ubuntu x86_64 dense-Q8 lane. It is not a production-ready or broad platform claim.

## What changed

- Added and validated default-off AVX2 Q8 acceleration paths in the measured Ubuntu x86 lane.
- Added packed Q8 runtime storage work for selected dense tensors.
- Kept matrix-level Q8 GEMM/MUL_MAT as an evidence-gated direction while documenting concrete default-off FFN/attention/output slices by their per-artifact evidence level; local-only follow-ons are not Ubuntu x86_64 validation.
- Separated cold materialization from warm inference.
- Documented rejected candidates when they failed parity, wall-clock, or clean-host discipline.

## Validated principles

- Warm inference should not rebuild packed rows.
- Cold materialization and warm decode are separate problems.
- `from_q8_0_bytes` is cold/reload-only on the measured Ubuntu x86 lane.
- Row-dot micro-optimizations are not enough by themselves.
- Matrix-level ownership remains a direction, not a support or throughput claim until a fresh Ubuntu x86_64 run proves a concrete default-off path.
- Retain decisions require parity plus repeated wall-clock evidence on a clean host.

## Current retained paths

Only list the paths that are currently evidence-backed and default-off:

- `CAMELID_X86_Q8_REPACK=on` for the retained Ubuntu x86 runtime-packed lane used in current evidence.
- AVX2 packed-kernel work in the measured Ubuntu x86 lane where parity and bounded timing evidence support keeping the path under default-off gating.
- Packed Q8 runtime storage for the dense attention projection family plus dense FFN gate/up/down rows in the measured lane.
- Default-off decode consumers that directly use backend-owned packed runtime storage for narrow one-row dense projection families, including output, attention Q/K/V, attention output, FFN down, and the FFN gate/up activation slice while validation remains opt-in.
- Default-off packed-rows4 matmul slices consume backend-owned packed runtime storage for concrete dense projection families with per-slice evidence recorded below: FFN down, multi-row FFN gate/up, multi-row attention Q/K/V, multi-row attention output, and local-only multi-row `output.weight`; the newest chunked output-group traversal and quantized-input scratch-reuse follow-ons remain local-only until a fresh canonical Ubuntu timing/profiling bundle exists. This is planner/runtime-gate/allocation-shape evidence, not a blanket throughput, support, portability, or default-on claim.
- Default-off FFN-down GEMM4 follow-ons now include prefill, row-group scheduling, a retained min-input-groups scheduler guard, and an AVX2 experiment gate. Current public docs retain these as developer experiments only: canonical Ubuntu parity plus repeated same-host timing/profiling evidence is still required before any throughput/RSS/support/default-on claim.
- ExecutionPlan now treats the x86 attention Q/K/V, attention-output, output, FFN gate/up/down decode-consumer, FFN gate/up single-owner, packed-rows4 FFN-down matmul, packed-rows4 FFN gate/up matmul, packed-rows4 attention-Q/K/V matmul, packed-rows4 attention-output matmul, packed-rows4 output matmul, and FFN-down GEMM4 flags as managed default-off knobs, so appliance planning clears stale owner experiments instead of inheriting them accidentally.

## Active experimental direction

Current work is focused on:

- explicit `CAMELID_X86_Q8_KERNEL=avx2` packed-kernel execution
- matrix-level Q8 GEMM/MUL_MAT ownership only after a concrete default-off flag/path has fresh Ubuntu x86_64 evidence
- FFN projection optimization, especially deeper `ffn_down` decode ownership and one-quantization FFN gate/up decode consumption
- attention projection optimization, including narrow one-row Q/K/V decode-consumer and multi-row Q/K/V or attention-output packed-runtime matmul slices only when guarded by fresh Ubuntu x86_64 evidence; current Q/K/V helpers use one shared input quantization and paired/triplet projection helpers under existing default-off gates.
- output-group scheduling for one-row packed-runtime decode consumers, including the latest local-only helper that can parallelize wide rows4 decode projections inside existing default-off gates; Ubuntu timing/profiling proof is still pending before retaining any measured effect.
- multi-row output projection ownership through backend-owned packed runtime storage; the current `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL` slice has local parity/gate evidence only. This run did not attempt remote validation, so it does not imply host failure or provide Ubuntu timing/profiling proof.
- bounded packed-rows4 matmul scheduling follow-ons that reduce Rayon task granularity by chunking output groups across single/pair/triplet helpers; current proof is local semantic coverage only, not a retained Ubuntu speed claim.
- bounded packed-rows4 matmul activation-quantization scratch reuse, so existing default-off single/pair/triplet matmul consumers can reuse cleared thread-local input blocks rather than allocating a fresh quantized-input vector per helper call; current proof is local allocation-shape/timing-smoke coverage only, not a retained Ubuntu speed claim.
- FFN-down GEMM4 AVX2 and output-route-resolver cleanup are evidence-needed tracer bullets: keep them default-off, preserve backend-owned packed runtime storage, and require parity plus same-host guard evidence before retaining any performance claim.
- FFN-down row-group scheduling now has a retained default-off min-input-groups guard for the shallow-prefill synthetic surface; model-backed same-host FFN-down timing remains evidence-needed before any measured throughput claim.
- reducing wrapper/callback overhead in hot inference
- keeping the default/reference path safe while experimental paths stay opt-in

## Rejected paths

Rejected paths stay documented when they fail for any of the following reasons:

- no wall-clock win
- parity fail
- contaminated host
- microbench-only improvement
- old-baseline-only improvement
- context-switch regression

Examples already treated this way include row-dot lookalikes, tile16 hsum/lane/simd-scale variants, wrapper-style GEMM detours, and contaminated benchmark runs.

## Clean-host discipline

Ubuntu x86 Q8 benchmarking now requires a clean host before major runs:

- check disk headroom
- inspect stale Camelid / perf / benchmark jobs
- clear conflicting ports
- remove abandoned scratch trees from invalid runs
- preserve retained evidence and model files

Contaminated runs are not used as retained evidence.

## How to reproduce

Use only the current reference default-off gates for the retained Ubuntu x86_64 experiment and keep the host clean before running:

```bash
CAMELID_X86_Q8_REPACK=on \
CAMELID_X86_Q8_KERNEL=avx2
```

Do not add older sketch flags or matrix-owner placeholders to reproduction commands unless a fresh Ubuntu x86_64 evidence entry proves that exact flag and shape. Narrow decode-consumer/owner flags are separate default-off developer experiments; enable them only when the current evidence report names the exact flag and validation target for that slice.

For bounded warm-request measurement, use the same host, same model, same request shape, repeated runs, parity checks, and paired `perf stat` / `perf record` evidence.

## Evidence bundles

Primary public evidence anchors for this lane:

- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/README.md`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260515T1108Z-x86-attn-family.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260515T1235Z-x86-ffn-down-runtime.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260516T0136Z-x86-ffn-gate-up-consumer.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260516T0136Z-x86-ffn-gate-up-consumer-tests.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T0320Z-x86-ffndown-consumer-planner.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T0503Z-x86-attn-output-consumer.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-5e4b0b83-20260517T0511Z-doc-claim-guard.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T0655Z-x86-ffndown-packed-rows4-matmul-planner.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-5e4b0b83-20260517T0733Z-doc-claim-guard.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T0825Z-x86-attn-output-packed-rows4-matmul.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T1013Z-x86-attn-qkv-packed-rows4-matmul.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T1148Z-x86-attn-qkv-shared-input-quant.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T1359Z-x86-ffn-gate-up-packed-rows4-matmul.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-5e4b0b83-20260517T1458Z-doc-claim-guard.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-1eeef0a5-20260517T1516Z-x86-ffn-gate-up-paired-projection-local.txt` (local follow-on only; Ubuntu x86_64 timing proof still pending)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T1522Z-x86-attn-qkv-triplet-packed-rows4.txt`
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-1eeef0a5-20260517T1645Z-x86-ffn-gate-up-decode-paired-local.txt` (local follow-on only; Ubuntu x86_64 timing proof still pending)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T1655Z-x86-attn-qkv-decode-triplet.txt` (local follow-on only; no retained Ubuntu timing/profiling proof in the public evidence list)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-1eeef0a5-20260517T1744Z-x86-packed-rows4-decode-output-parallel-local.txt` (local follow-on only; Ubuntu x86_64 timing/profiling proof still pending)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T1834Z-x86-q8-local-gate-blocker.txt` (local gate/source-inspection only; no retained Ubuntu timing/perf claim)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-1eeef0a5-20260517T1850Z-x86-output-packed-rows4-matmul-local.txt` (local parity/gate/timing-smoke only; no retained Ubuntu timing/perf claim)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-1eeef0a5-20260517T2001Z-x86-packed-rows4-matmul-chunking-local.txt` (local fmt/clippy/unit parity only; no retained Ubuntu timing/perf claim)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-1eeef0a5-20260517T2118Z-x86-packed-rows4-input-scratch-local.txt` (local scratch-reuse parity/timing-smoke only; no retained Ubuntu timing/perf claim)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260517T2207Z-x86-output-packed-rows4-canonical-host-blocker.txt` (default-off output packed-rows4 matmul validation-staging artifact; no retained Ubuntu timing/perf claim)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-5e4b0b83-20260518T1526Z-docs-claim-guard/README.md` (docs/context claim guard: FFN-down GEMM4 AVX2 remains default-off evidence-needed work; latest same-host guard rejects new performance promotion; output route resolver remains implementation guidance only)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-0719640b-20260518T1515Z-ffndown-rowgroup-threshold/README.md` (retained default-off FFN-down GEMM4 row-group min-input-groups scheduler guard; synthetic scheduler evidence only, not broad throughput/support/default-on evidence)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260518T1616Z-ffn-gate-up-single-owner-env-guard/README.md` (retained default-off execution-plan guard for stale FFN gate/up single-owner env leakage; control-plane hygiene only, not throughput/parity/support evidence)
- `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-5e4b0b83-20260518T1730Z-docs-claim-guard-sync/README.md` (docs/context claim guard sync: sanitized public evidence references and aligned default-off scheduler/control-plane wording without support-contract widening)
- the retained/reject notes for bounded Ubuntu x86 Q8 experiments kept under `qa/evidence-bundles/`

## Product/runtime note

Camelid is moving toward an appliance-style execution plan where validated runtime paths can be selected automatically while experimental acceleration remains opt-in.

The intended product/runtime mode split is:

- `safe`
- `auto`
- `experimental`
- `debug`

This note does **not** claim full multi-model runner orchestration today. It describes the direction for exposing validated runtime behavior more clearly without pushing env-var complexity onto normal users.
