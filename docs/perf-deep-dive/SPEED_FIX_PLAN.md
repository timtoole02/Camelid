# SPEED_FIX_PLAN.md

Ranked by (1) expected speedup, (2) risk to correctness, (3) complexity, (4) Rust-only fit, (5) evidence-gated-support fit. Every claim traces to `PERF_RECEIPTS/` or an in-repo campaign. The headline result of this deep-dive is uncomfortable but honest: **on this hardware there is no low-risk flag/routing CPU speed win — the obvious ones were measured and they regress or do nothing.** So the ranking leans on one real architectural lever (P1) and protects the codebase from fake wins (P0).

---

## P0 — must do now: lock in honest measurement + stop the tempting wrong fix

**Problem.** The repo's CPU-perf direction is littered with traps that produce false "wins": (a) `llama-bench -ngl 0` is GPU-offloaded for prefill on a CUDA build (pp512 inflated ~24×); (b) Camelid prompt-prefix-caches, so naive prefill re-measurement is a cache hit; (c) the gated `CAMELID_X86_Q8_*` SIMD kernels look like a win on paper but **measurably regress −8–11% on this box**. Without a guarded harness, someone will "just enable AVX2" and ship a regression.

**Evidence.** `PERF_RECEIPTS/same-host/cpu-prefill-matrix-*.json` (config B regresses, byte-identical), `llama-bench/cpu-TRUE-cudaHidden.txt` vs the contaminated `cpu-ngl0-t8.txt`, routing sweep flat.

**Implementation.** Commit the three reproducible probes (`docs/perf-deep-dive/scripts/{cpu-prefill-matrix,prefill-routing-probe,prefill-decode-probe}.mjs`) which already (i) hide CUDA for a true CPU lane, (ii) defeat the prompt cache with a unique nonce, (iii) A/B any flag set with a byte-identical parity diff. Keep the negative-result receipts as the regression guard.

**Files.** `docs/perf-deep-dive/scripts/*` (new, additive), `docs/perf-deep-dive/*.md`. No engine code.
**Test/parity.** Harness is test-only; no engine change ⇒ parity trivially intact.
**Benchmark.** The harness *is* the benchmark.
**Rollback.** Delete the docs dir. Zero blast radius.

---

## P0.5 — shippable today, bit-identical, no code change: decode thread tuning (this box)

**Problem.** This box's memory bandwidth (2-channel DDR4) saturates at ~1–2 threads for the batch-1 decode matvec; Camelid's default (physical-core cap = 8) **over-subscribes**, costing ~20% decode.

**Evidence (bit-identical, receipt `same-host/decode-thread-sweep-llama3b-*.json`, `all_parity_identical: true`):**
| RAYON_NUM_THREADS | decode tok/s | prefill tok/s |
|---|---:|---:|
| default (8 physical) | 5.14 | 19.14 |
| **2** | **6.11** | 18.45 |
| **1** | **6.16** | 19.24 |

Decode +20% at 1–2 threads; prefill flat (memory-saturated regardless). Output byte-identical across all thread counts.

**The fix is already exposed:** run with **`CAMELID_THREADS=2`** (or `=1`) on bandwidth-starved laptops → +20% decode, byte-identical, zero risk. **No code change is warranted** — changing the *default* would over-fit this box and regress high-bandwidth servers (which want more threads). The default (physical-core cap) is the correct universal choice.

**Caveat:** the team's `windows_physical_core_count` comment frames decode as "compute-bound"; it's actually *memory-bound*, so even physical-core capping over-subscribes here. A memory-channel-aware decode thread count would be the principled universal fix, but reliable cross-platform channel detection is hard — deferred.

## P1 — high impact, real, parity-safe, but medium–high effort: unified tiled Q8 GEMM owner

**Problem.** Camelid CPU is a uniform **0.61–0.68×** of llama.cpp (prefill 18.9 vs 30.6; decode 5.97 vs 9.08). The cause is architectural, not a flag: llama.cpp routes every projection through ONE tiled tinyBLAS GEMM (register-blocked M×N×K, AVX-512+FMA, REPACK) with an in-kernel chunk scheduler and a single activation quantization; Camelid runs **AVX2 bespoke per-role kernels** that re-quantize the input per projection and aren't register-tiled. The repo's own mapping note proposes exactly this owner (`CAMELID_X86_Q8_MATMUL_OWNER=ffn_down`).

**Evidence.** Uniform CPU ratio (`same-host/…cpuonly-nocache…json`); `qa/evidence-bundles/x86-q8-llamacpp-mapping-*/NOTES.md`; `LLAMA_CPP_ARCHAEOLOGY.md §1–2`.

**Implementation (incremental, one projection family at a time).**
1. Add `matmul_q8_0_runtime_packed_x86(input_rows[M], packed_weight, out[M,N])` in `src/inference.rs` (or `src/tensor/mod.rs`): a register-blocked Q8×Q8→f32 micro-kernel over the EXISTING `q8_0_runtime_packed_rows4` repack, using `core::arch` AVX2 FMA, accumulating int8→i32 per block (associative ⇒ bit-exact) and applying the f16 block-scale product in fixed per-block order.
2. Single activation quantize per input row, shared across Q/K/V (and gate/up) — reused, not recomputed.
3. Rayon `par_chunks` over output tiles for the scheduler.
4. Route `ffn_down` first (mapping note's first slice), then attention Q/K/V/O, then gate/up.
5. Gate it `default-off` (`CAMELID_X86_Q8_MATMUL_OWNER`) until it BEATS the current default on this host AND on the Ubuntu validation host (the existing gated kernels regress here — the new one must not).

**Files.** `src/inference.rs`, `src/tensor/mod.rs`, `src/inference/q8_runtime.rs` (flag), `src/execution_plan.rs` (capability string), tests.
**Test plan.** Unit: new kernel == current path bit-for-bit on representative widths (Llama 3072/8192, Qwen 1024/3072). `cargo test --all-features` on the role paths.
**Parity plan.** int8→i32 accumulation is order-independent (bit-exact); only requirement is fixed per-block scale-sum order. Verify token-identical greedy vs the default path (`prefill-decode-probe.mjs` byte diff) AND vs llama.cpp via the existing exact-row bundles.
**Benchmark plan.** `cpu-prefill-matrix.mjs` default vs `CAMELID_X86_Q8_MATMUL_OWNER=ffn_down,attn,ffn` — require a measured prefill+decode gain on THIS host before promotion.
**Rollback plan.** Flag default-off; revert is removing the route. No default changes until proven on both hosts.

**Why not now (P0)?** It's a multi-slice refactor (~6 role paths), and beating the team's existing (regressing) SIMD attempt requires careful register-tiling + profiling — not a one-session small patch. Highest-confidence lever; right next investment.

---

## P2 — good, not urgent

- **GPU multi-stream overlap of independent projections (the one real remaining parity-safe lever).**
  - **Problem.** Decode runs Q/K/V and gate/up GEMVs serially on one stream; each is DRAM-latency-bound at batch-1, so the GPU idles between them.
  - **Evidence.** `qa/perf/qwen3-cuda-resident-phase3-findings.md:131-139` (campaign deferred it); GPU audit estimate ~10–15% decode lift. This is the **default path on a GPU box**, so it's user-facing.
  - **Implementation.** Issue the independent Q/K/V (and gate/up) GEMVs on separate CUDA streams in `src/cuda_resident.rs`, join before the dependent op. Math is unchanged ⇒ parity-safe (no re-association — each GEMV is computed identically, just concurrently).
  - **Files.** `src/cuda_resident.rs`. **Parity plan.** token-identical greedy vs current (`scripts/validate-cuda-prefill-row.sh` pattern). **Benchmark.** `bench-qwen3-cuda-resident.mjs`. **Difficulty.** medium (stream lifetime + event sync). **Rollback.** env flag default-off until validated.

- **Extend the physical-core thread cap to Linux x86_64.** `configure_rayon_threads` (`src/main.rs:4008`) only caps to physical cores on `target_os = "windows"`; on Linux it leaves Rayon at logical-core sizing, so SMT siblings over-subscribe the memory-bound matvec (the exact issue the Windows path fixes, validated there). Add a `linux_physical_core_count()` (parse `/sys/devices/system/cpu/cpu*/topology/core_id` unique count, or `num_cpus::get_physical`) and include it in `resolved`. Bit-identical (thread count only). **Low-risk, but measure on the Ubuntu validation host** — can't be compiled/measured from this Windows box (Linux-cfg code), so it's scoped here rather than shipped blind.

- **AVX-512 tiled GEMM for prefill ONLY.** Prefill is compute-bound and may amortize the Tiger Lake AVX-512 downclock (the 2026-06-20 char only tested AVX-512 on memory-bound *decode*, where it can't help). Research + measure; parity-safe if accumulation order fixed. Couples to P1.

> **Audit result — no CPU micro-win exists.** An 8-agent audit proposed 5 CPU decode-loop micro-optimizations (buffer reuse, logits-clone removal, shared quantization); adversarial verifiers **rejected all 5** (`PERF_RECEIPTS/audit-workflow-result.json`): two are parity-unsafe or impossible (buffer is *moved* into the tensor; clone removal corrupts dense diagnostics), three are already-optimal/no-win. The decode loop is already allocation-clean and shares input quantization. Confirms: the matvec is the only CPU lever (P1).

## P3 — research lane only
- **CPU spec-decode drafter** — blocked: `cpu-perf-characterization-20260620` shows the 0.6B CPU drafter at ~28 tok/s vs the ~70 needed; not viable on this box without the P1 kernel.
- **Match llama.cpp's flat-at-depth CUDA decode** — requires non-bit-exact flash attention (coalesced K/V re-associates the dot, flips near-tie greedy tokens). **Rejected by the losslessness contract.** Camelid's split-K already extracts the max parity-safe occupancy win.
- **AVX-512 decode / VNNI** — rejected: downclock cancels the gain (measured +0.28–4.5%, reverted).

---

## What moves tokens/sec first
The first patch that moves CPU tok/s is **P1 (tiled GEMM owner)** — nothing smaller works on this hardware (proven). The first patch that ships *today* is **P0 (harness + regression guard)**, which is what keeps the next "enable AVX2" idea from regressing the engine.
