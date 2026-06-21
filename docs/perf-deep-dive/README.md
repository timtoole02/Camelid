# Camelid vs llama.cpp — performance deep dive (2026-06-21)

Blunt, evidence-first. Same model, GGUF, Q8_0 quant, host, OS, build mode (both Release), greedy. Camelid stays Rust-native; llama.cpp was studied for design only (nothing linked/copied).

**Host:** i7-11800H (8C/16T, Tiger Lake), 16 GiB DDR4, RTX 3060 Laptop 6 GiB (sm_86), Win11. **Camelid** `ce7dceb6`. **llama.cpp** `acd79d6`.

## Documents
- **`PERF_GAP_REPORT.md`** — the gap table + the blunt read + methodology traps.
- **`LLAMA_CPP_ARCHAEOLOGY.md`** — the machinery that explains the gap, with verified `file:line` refs + Rust-native verdicts.
- **`CAMELID_HOTSPOTS.md`** — measured hotspots; the audit's zero-confirmed-wins result.
- **`SPEED_FIX_PLAN.md`** — ranked P0–P3.
- **`PERF_RECEIPTS/`** — env, model SHA-256, raw llama-bench logs, same-host JSONs, parsed metrics, the audit-workflow JSON. No claim without a receipt.
- **`scripts/`** — the reproducible probes (CUDA-hidden true-CPU lane + prompt-cache defeat + A/B parity diff).

## The five findings (TL;DR)

1. **CPU: Camelid is a uniform ~0.62–0.68× of llama.cpp** on BOTH prefill (18.9 vs 30.6 tok/s) and decode (5.97 vs 9.08). Uniform ⇒ it's a **per-kernel-throughput** gap, not a batching gap. Cause: llama.cpp = one **tiled tinyBLAS GEMM (AVX-512+FMA+REPACK) + in-kernel chunk scheduler**; Camelid = AVX2 bespoke per-role kernels.

2. **The easy CPU fixes are dead — proven, not assumed.** Enabling the gated `CAMELID_X86_Q8_*` SIMD kernels **regresses −8…−11%** here (byte-identical) — vindicating the team's default-off discipline. Prefill routing (layer-major / chunk size) is flat <3%. The "+39% AVX2" is already shipped (`target-cpu=x86-64-v3`+fat-LTO). AVX-512 deliberately excluded (downclocks Tiger Lake).

3. **CUDA: near-optimal, and the residual is a deliberate correctness choice.** `q8_gemv` ~76% of DRAM bandwidth; the depth gap needs coalesced K/V reads that re-associate the attention dot and flip near-tie greedy tokens — refused under the losslessness contract. On a GPU box (the default here) Camelid's prefill is excellent (~905 tok/s).

4. **No low-risk CPU win exists — adversarially verified.** An 8-agent audit proposed 5 decode-loop micro-opts; skeptics **rejected all 5** (parity-unsafe or sub-noise vs the matvec). The decode loop is already allocation-clean and shares input quantization.

5. **Two methodology traps caught** (each would have produced a false claim): `llama-bench -ngl 0` offloads prefill to the GPU (pp512 inflated ~24×); Camelid prompt-prefix-caches (naive prefill re-measure is a cache hit). Both defeated in the committed harness.

## Implemented now vs scoped

- **Implemented (low-risk, additive):** the reproducible benchmark harness + the negative-result receipts (P0) — a regression guard so nobody ships "just enable AVX2."
- **NOT shipped, on purpose:** no engine speed patch. Every low-risk candidate was a measured regression, parity-unsafe, or below measurement noise (so it couldn't even produce a credible before/after). Shipping one would violate Camelid's correctness-first contract.
- **Scoped real wins:** P1 = unified tiled Q8 GEMM owner (the only CPU lever; medium–high effort, parity-safe). P2 = GPU multi-stream QKV/gate-up overlap (~10–15% decode, parity-safe, medium effort).

## Reproduce
```
# true CPU lanes (CUDA hidden), greedy, cache-defeated:
CUDA_VISIBLE_DEVICES=-1 CAMELID_BIN=target/release/camelid.exe \
  LLAMA_SERVER_BIN=<llama.cpp>/build/bin/llama-server.exe \
  MODEL_GGUF=<Llama-3.2-3B-Instruct-Q8_0.gguf> MODEL_ID=llama32-3b-q8-cpu \
  node docs/perf-deep-dive/scripts/cpu-prefill-matrix.mjs
# llama.cpp ground truth:
CUDA_VISIBLE_DEVICES=-1 llama-bench.exe -m <gguf> -ngl 0 -t 8 -p 512 -n 128 -r 3
```
