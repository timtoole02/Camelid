# Runtime Cross-Lane Sync

This file is the shared bridge between Camelid's Ubuntu x86 Q8 performance lane, Mac arm64 Q8/product UX lane, and ExecutionPlan/runtime product lane.

Rules:
- Do not mix evidence across platforms.
- Ubuntu numbers are Ubuntu-only.
- Mac numbers are Mac-only.
- Report Ubuntu validation status as evidence status, not host-access status. Keep exact probe commands, raw stderr, return codes, host addresses, users, key paths, and working directories in private operator notes only. If remote validation is unavailable for the current slice, use the public-safe wording from `docs/performance/ubuntu-x86-q8.md`.
- Architecture lessons can be shared.
- Kernel implementations cannot be blindly copied.
- Mac must not copy AVX2 assumptions.
- Ubuntu must not assume Apple Silicon behavior.
- Product/runtime should expose and select only validated paths.

## Latest cross-lane digests

### Cross-lane sync

Source lane: Ubuntu x86 Q8 / Discord

Finding: `d9ad412` caches the x86 Q8 kernel gate outside hot dot loops; proof remains pending and the active P0 is runtime-overhead validation, not new kernel tuning.

Why it matters: Hot-path env/config reads can dominate optimized kernels and invalidate benchmark conclusions if path sanity is not captured first.

Applies to: Mac arm64 Q8 and Product/ExecutionPlan as an architecture/process lesson: audit hot inference paths for env/config/rwlock reads and expose selected runtime paths before benchmarks.

Does not apply to: Mac performance numbers, Apple Silicon kernel choices, or any claim that Ubuntu AVX2 evidence proves Mac behavior.

Action for other lane: Mac should audit env/config/rwlock hot paths before trusting Q8 results; Product/ExecutionPlan should expose selected backend and Q8 path in `/health` and `/api/capabilities` before benchmark runs.

Evidence / file / commit: `d9ad412 Cache x86 Q8 kernel gate outside hot dot loops`; merged to `origin/main` by `49349db`; source file `src/inference.rs`.

Owner: Cross-lane sync owner; Ubuntu proof owner retains numeric validation.

### Cross-lane sync

Source lane: Mac arm64 Q8 / OpenClaw

Finding: Mac packed-prefill stays default-off/experimental because retained Mac evidence favors the existing auto/direct-pack path over RSS-only packed-prefill wins.

Why it matters: Product defaults should not promote a path that improves memory while regressing wall-clock/prefill compute on the same platform.

Applies to: Ubuntu and Product/ExecutionPlan as a process lesson: require balanced evidence and keep experimental profiles explicit.

Does not apply to: Ubuntu timings, AVX2 gates, or any claim that Mac packed-prefill evidence proves Ubuntu packed-runtime behavior.

Action for other lane: Ubuntu should avoid promoting RSS-only or micro-kernel-only wins without warm wall-clock/path sanity; Product should keep experimental paths opt-in.

Evidence / file / commit: `docs/runtime/cross-lane-sync.md`; prior Mac retained path notes summarized in this document.

Owner: Cross-lane sync owner; Mac lane owner retains Mac numeric validation.

### Cross-lane sync

Source lane: Product / ExecutionPlan / runtime UX

Finding: Benchmarks need product-visible selected backend/Q8 path/profile before results are treated as optimized-path evidence.

Why it matters: If a run silently selects `cpu_reference` or `safe_q8_0_block_dot`, it is safe/reference evidence, not optimized packed-Q8 evidence.

Applies to: Ubuntu x86 Q8 and Mac arm64 Q8 benchmark discipline.

Does not apply to: Any numeric performance transfer between Ubuntu and Mac, or any claim that UX visibility validates a kernel.

Action for other lane: Capture selected backend, selected Q8 path, active profile, gates, and packed-runtime state with every major benchmark.

Evidence / file / commit: `docs/runtime/cross-lane-sync.md`; ExecutionPlan visibility requirements for `/health` and `/api/capabilities`.

Owner: Cross-lane sync owner; Product/ExecutionPlan owner implements UX/runtime exposure.

## Current Ubuntu x86 Q8 status

- Current evidence posture: keep public status evidence-scoped. This sync note does not establish host availability or failure; where no current remote run was attempted, say that no Ubuntu timing/profiling validation is recorded for the slice.
- Latest docs/context host-reporting retained audit (`qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-5e4b0b83-20260520T1244Z-docs-context-host-reporting-audit/README.md`) passed the public docs/context scan for stale host-access wording and private host aliases. Remote validation was not attempted in that docs-only audit.
- Recent retained control-plane/scheduler hygiene includes the FFN-down GEMM4 row-group min-input-groups guard. That evidence is default-off, synthetic scheduler evidence only; it is not broad throughput, support, portability, RSS, or default-on evidence.
- Runtime/config hot-path overhead remains a benchmark-validity lesson: path sanity must precede new Q8 kernel or owner experiments.
- `f65eac8` runtime-plan candidate is rejected: it timed out and still showed `getenv` / env-lock contention dominating.
- Surgical root cause found: `x86_q8_kernel_avx2_enabled()` was reading `CAMELID_X86_Q8_KERNEL` inside hot Q8 dot paths, effectively per Q8 block / packed rows4 group.
- Surgical fix branch: `camelid-runtime-overhead`.
- Surgical fix commit: `d9ad412 Cache x86 Q8 kernel gate outside hot dot loops`.
- This document records no retained before/after timing/profiling promotion for `d9ad412`; do not treat it as an accepted performance result until warm timing, parity, path sanity, and top-symbol evidence pass.
- Owner/kernel A/B work should not resume from this sync note alone; require fresh same-host path sanity plus parity/timing evidence for the exact default-off flag under test.
- `3244b35` is rejected as an active packed-runtime baseline: packed-runtime smoke timed out and safe/control was catastrophically slow.
- `80f6271` is an infrastructure anchor only, not a usable performance baseline.
- `83063cf` is historical retained evidence only, not a packed-runtime implementation baseline.

## Current Mac arm64 Q8 / product UX status

- Mac packed-prefill remains default-off / experimental.
- Reason: repeatability evidence showed RSS improved, but wall-clock and prefill compute regressed versus the retained Mac auto/direct-pack path.
- Mac auto profile should remain on the last proven retained path, not packed-prefill.
- ExecutionPlan/appliance UX work is moving forward:
  - safe / auto / experimental / debug profile concepts
  - selected plan exposure in `/health`
  - selected path/capability exposure in `/api/capabilities`
  - appliance-style runtime behavior for product use
- Mac must audit hot inference paths for the same env/config/rwlock issue Ubuntu found.

## Shared lessons

- Separate cold materialization from warm inference. Do not treat packed-row build/load cost as warm decode unless evidence proves it is happening in warm decode.
- Expose selected backend and selected Q8 path before benchmarking. If a run reports `cpu_reference` / `safe_q8_0_block_dot`, label it as safe/reference evidence, not optimized packed-Q8 evidence.
- Remove env/config/rwlock reads from hot inference paths. Per-layer, per-projection, per-row, or per-block `getenv` calls are performance bugs.
- Benchmark only clean-host runs. Capture disk, process, and port state before major runs.
- Do not promote RSS-only wins if wall-clock regresses.
- Default-off experimental paths must stay default-off.
- Baseline/gate drift can invalidate days of work; record exact SHA and gates.
- Row-dot micro-tuning is not enough if matrix ownership and runtime scheduling are wrong.
- Server/harness failures must distinguish real server crashes from script cleanup artifacts.

## Platform-specific lessons

### Ubuntu x86 Q8

- AVX2-specific gates and kernels are Ubuntu/x86 evidence only.
- `CAMELID_X86_Q8_KERNEL` should be resolved once per process for the hot Q8 kernel gate; changing that env var after process start should not be expected to affect a running server.
- Path sanity is mandatory before perf: selected backend, selected Q8 path, active profile, active gates, packed runtime active/not active.
- Owner/kernel A/B must stay paused until runtime overhead proof passes.

### Mac arm64 Q8

- Apple Silicon decisions must be based on Mac evidence only.
- Packed-prefill RSS improvement was not enough because wall-clock regressed.
- Mac must audit for hot-path env/config/rwlock reads before trusting new kernel results.
- Mac should use ExecutionPlan-selected validated paths rather than ad hoc env-var piles.

### Product / ExecutionPlan

- ExecutionPlan should be the product-facing way to select validated paths.
- Profiles must remain explicit: safe, auto, experimental, debug.
- `/health` and `/api/capabilities` should make selected plan/path visible enough to prevent invalid benchmarks.
- Product should consume only validated, evidence-backed paths.

## Active baselines / anchors

- Ubuntu active retained evidence: historical `83063cf` only for old retained sanity; not a packed-runtime baseline.
- Ubuntu default-off scheduler guard: FFN-down GEMM4 row-group min-input-groups threshold retained for the bounded synthetic scheduler surface only; model-backed same-host timing remains evidence-needed before any throughput claim.
- Ubuntu runtime-overhead candidate: `d9ad412` remains evidence-needed in this sync note; no retained before/after timing/profiling promotion is recorded here.
- Mac active auto path: retained direct-pack/I8MM-style path with packed-prefill off.
- Mac packed-prefill: rejected for promotion, default-off/experimental only.

## Rejected or paused paths

- Older FFN-down GEMM4 shorthand experiments: rejected for promotion where parity held but wall/`ffn_down` timing regressed; use only the current documented default-off flags when a fresh evidence bundle names the exact gate and shape.
- `3244b35` as active packed-runtime baseline: rejected; timeout/catastrophic safe-control timing.
- `f65eac8`: rejected runtime-overhead candidate; `getenv`/env-lock still dominated and request timed out.
- FFN-down owner and attention-output owner: do not resume from stale status alone; require fresh exact-flag evidence with path sanity, parity, timing, and a clean harness/server distinction.
- Mac packed-prefill promotion: rejected; default-off/experimental only.

## What each lane should not repeat

- Ubuntu: do not benchmark against wrong histories or missing gates; do not proceed without path sanity. If the canonical SSH command was not attempted in the current run, report remote validation as not attempted rather than implying host outage/auth failure.
- Mac: do not promote RSS-only improvements; audit env/config hot-path overhead before trusting kernel wins.
- Product: do not let runtime path selection become an unobservable env-var pile.

## Next action per lane

- Ubuntu: record the next model-backed same-host FFN-down timing/profiling proof for the retained default-off scheduler guard, or explicitly mark remote validation as not attempted; keep `d9ad412` performance promotion evidence-needed until a retained before/after bundle exists.
- Mac: audit hot inference paths for env/config/rwlock reads and ensure selected plan/path are visible before benchmarks.
- Product/ExecutionPlan: keep turning validated paths into explicit profiles and health/capability visibility.
- Sync owner: post short cross-lane digests whenever either lane finds a major lesson, accepted path, rejected path, or benchmark-validity rule.

## Digest template

### Cross-lane sync

Source lane:

Finding:

Why it matters:

Applies to:

Does not apply to:

Action for other lane:

Evidence / file / commit:

Owner:
