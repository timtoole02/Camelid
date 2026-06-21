# PERF_GAP_REPORT.md — Camelid vs llama.cpp

Blunt, evidence-first. Every number traces to `PERF_RECEIPTS/`. Same model, same GGUF, same Q8_0 quant, same host, same OS, same build mode (both Release), greedy/temp=0.

- **Host:** i7-11800H (8C/16T, Tiger Lake — AVX2+AVX-512+VNNI), ~16 GiB DDR4 (~51 GB/s peak), RTX 3060 Laptop 6 GiB (sm_86), Win11.
- **Camelid** `ce7dceb6` (release: `target-cpu=x86-64-v3`, fat-LTO). **llama.cpp** `acd79d6` (Release; CPU = AVX-512+FMA+tinyBLAS+REPACK+OpenMP; CUDA = FA+graphs, arch 86).
- **Models:** Llama-3.2-3B-Instruct-Q8_0 (primary, shared on disk), Qwen3-4B-Q8_0, Qwen3-0.6B-Q8_0. SHA-256 in `PERF_RECEIPTS/env/model-sha256.txt`.

## Headline gap table

| Model | Quant | HW | Stage | Camelid | llama.cpp | Gap | Suspected cause | Evidence | Proposed fix | Parity risk | Difficulty |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|
| Llama-3.2-3B | Q8_0 | CPU | prefill tok/s | 18.9 | 30.6 | **0.62×** | No tiled AVX-512 GEMM; AVX2 fragmented per-role kernels | `same-host/…cpuonly-nocache…json` | Unified tiled Q8 GEMM owner (§ARCH) | Low (int acc bit-exact) | High |
| Llama-3.2-3B | Q8_0 | CPU | decode tok/s | 5.97 | 9.08 | **0.66×** | Memory-bandwidth bound (~33% peak); llama AVX-512+repack streams better | same + `cpu-perf-characterization-20260620` | None cheap (see below) | — | High |
| Qwen3-4B | Q8_0 | CPU | pp512 / tg128 (llama true CPU) | — | 23.6 / 7.44 | — | (Camelid CPU ratio ≈ 3B's 0.62×) | `llama-bench/cpu-TRUE-cudaHidden.txt` | — | — | — |
| Qwen3-0.6B | Q8_0 | CPU | tg128 | ~28* | 45.8 | ~0.61× | same memory wall | llama-bench (true CPU) + char-20260620 | — | — | — |
| Llama-3.2-3B | Q8_0 | CUDA | decode tok/s | ~53** | 69.3 | ~0.77× | parity-locked attention (no non-bit-exact flash-attn) | `llama-bench/cuda-ngl99.txt` + campaign | (already optimized) | locked | n/a |
| Qwen3-4B | Q8_0 | CUDA | decode low-ctx | 41.6 | 54.4 | **0.77×** | `q8_gemv` ~76% BW; rest parity-locked | `qa/perf/decode-attention-campaign.md` (cross-validated: llama 54.41 measured this session) | none low-risk | locked | n/a |
| Qwen3-4B | Q8_0 | CUDA | decode ctx~1881 | 26.2 | 51.1 | **0.51×** | uncoalesced K/V (coalescing flips near-tie greedy tokens) | campaign | non-bit-exact flash-attn (rejected) | **breaks parity** | n/a |
| Llama-3.2-3B | Q8_0 | CUDA(default) | prefill tok/s | ~905 | (CPU 30.6) | **>2× faster** | Camelid GPU batched prefill (shipped) | `same-host/…162052Z` (GPU-confound run) | — | — | done |

\* Qwen3-0.6B Camelid CPU decode from `cpu-perf-characterization-20260620` (~28 tok/s optimized). \** Llama-3B Camelid CUDA decode extrapolated from the campaign's ratio; a direct measure is a follow-up.

## Commands (exact)

```
# llama.cpp ground truth
llama-bench.exe -m <gguf> -ngl 0  -t 8 -p 512 -n 128 -r 3     # CPU
llama-bench.exe -m <gguf> -ngl 99       -p 512 -n 128 -r 3     # CUDA
# Same-host CPU (CUDA hidden, cache defeated), greedy temp=0:
CUDA_VISIBLE_DEVICES=-1 node docs/perf-deep-dive/scripts/cpu-prefill-matrix.mjs   # llama vs Camelid A vs Camelid B
```
Camelid forced CPU with `CAMELID_CUDA_RESIDENT_DECODE=0 CAMELID_CUDA_RESIDENT_PREFILL=0` and `CUDA_VISIBLE_DEVICES=-1`.

## Methodology traps found (both would have produced false claims)

- **`-ngl 0` is not CPU-only on a CUDA build.** ggml offloads compute-bound prefill matmuls to the GPU even at `ngl0`; llama-bench pp512 (740/545/2585) was GPU-assisted. True CPU prefill (CUDA hidden) is 30.9/23.6/167. Without catching this, the "prefill gap" would have been mis-stated as ~40×.
- **Camelid prompt-prefix-caches by default.** A warmup request caches the prefix, so a naive "prefill" re-request is a cache hit (measured 998 tok/s — impossible for real CPU compute). The harness defeats it with a unique system-message nonce. (llama gets `cache_prompt:false`.)
- Both engines were therefore measured CPU-only (`CUDA_VISIBLE_DEVICES=-1`), cache-defeated, greedy, same prompt.

## The blunt read

1. **CPU is where Camelid is behind, and the gap is uniform ~0.62–0.68× across BOTH prefill and decode** (prefill 18.9 vs 30.6; decode 5.97 vs 9.08). Because it's uniform, it's a **per-kernel-throughput** gap, NOT a prefill-batching problem — Camelid's prefill amortizes ~3.3× over decode, the same as llama's ~3.4×. Cause is architectural: llama.cpp runs ONE tiled tinyBLAS GEMM (AVX-512+FMA+REPACK) with an in-kernel chunk scheduler; Camelid runs AVX2 bespoke per-role kernels. See `LLAMA_CPP_ARCHAEOLOGY.md §1–2`.

2. **The "easy" CPU fixes are dead — proven, not assumed.**
   - Enabling the gated x86 SIMD packed-rows4/GEMM4 kernels (`CAMELID_X86_Q8_*`) **REGRESSES** prefill −11% / decode −8% on this box, byte-identical output. *This vindicates the team's default-off discipline.* (`same-host/…cpuonly-nocache…json`, config B.)
   - Prefill **routing** (layer-major on/off, chunk 64/256/512/all) moves <3% — noise. (`same-host/cpu-prefill-routing-*.json`.)
   - The "+39% AVX2" from the stale note is **already shipped** (`target-cpu=x86-64-v3`+fat-LTO). AVX-512 is deliberately excluded (downclocks Tiger Lake; VNNI gave +0.28–4.5% on memory-bound decode, reverted).

3. **CUDA is near-optimal and the residual is a deliberate correctness choice, not lost performance.** `q8_gemv` is ~76% of DRAM bandwidth; the depth gap needs coalesced K/V reads that re-associate the attention dot and flip near-tie greedy tokens — which Camelid refuses (losslessness). On a GPU box (the default here) Camelid's prefill is excellent (~905 tok/s).

4. **Sampling, SSE/server, tokenizer, mmap/load are NOT bottlenecks** in either engine.

## What's fixable now vs not

- **P1 (real, fixable, parity-safe, but medium-high effort):** unified tiled Q8 GEMM owner with register-blocked AVX2 (and an AVX-512 *prefill-only* variant). This is the only lever that moves CPU tok/s. Plan in `SPEED_FIX_PLAN.md`.
- **Not worth chasing now:** CUDA decode (parity-locked), AVX-512 decode (downclock), the existing gated packed kernels on this host (measured regression), sampler/server.
