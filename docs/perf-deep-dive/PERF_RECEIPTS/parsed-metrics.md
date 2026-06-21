# Parsed metrics — Camelid vs llama.cpp perf deep dive

Captured 2026-06-21 (UTC). All numbers are from receipts in this directory. No claim without a receipt.

## Identity / environment (see `env/`)

- Camelid HEAD: `ce7dceb6ebbd730166be38483e2dd0229273d5c9` (branch `feat/bactrian-experimental-fast`)
- llama.cpp HEAD: `acd79d603cb2e1c84c0886137b80f1ad649b6857` (`build: acd79d6 (1)`, MSVC 19.44)
- Rust 1.95.0; CUDA 12.9 (nvcc V12.9.86); MSVC build.
- Host: i7-11800H (8C/16T, Tiger Lake; AVX2+AVX-512+VNNI), ~16 GiB DDR4 (~51 GB/s peak), RTX 3060 Laptop 6 GiB (sm_86), Win11.
- llama.cpp CPU backend (from `system_info`): `AVX2=1 | AVX512=1 | F16C=1 | FMA=1 | LLAMAFILE=1 (tinyBLAS) | OPENMP=1 | REPACK=1`, n_threads=8.
- Camelid CPU backend (from startup banner): `SIMD AVX-512 (avx2=true avx512f=true fma=true)` but kernels target AVX2 (x86-64-v3) by choice; CUDA hidden => "GPU: none detected — CPU backend is the inference path".

### Model SHA-256 (`env/model-sha256.txt`)
- Llama-3.2-3B-Instruct-Q8_0.gguf: `f34112a11b7dad74ab517dedf6dcf00d624c9adac2dc0c72c719ca0478554ef2` (3.18 GiB, 3.21 B)
- Qwen3-4B-Q8_0.gguf: `8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300` (3.98 GiB, 4.02 B)
- Qwen3-0.6B-Q8_0.gguf: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` (604 MiB, 596 M)

## llama.cpp `llama-bench` ground truth (`llama-bench/`)

⚠️ **Methodology finding — `-ngl 0` is NOT CPU-only on a CUDA build.** With the CUDA backend initialized, ggml's scheduler offloads compute-bound batched matmuls (prefill/pp512) to the GPU even at `-ngl 0`. The first `cpu-ngl0-t8.txt` pp512 numbers (740/545/2585) are therefore **GPU-assisted, not CPU** — pp512=2585 for 0.6B would need ~3.1 Tflop/s, ~10× this CPU's AVX2 peak (physically impossible on CPU). **True CPU-only** numbers below use `CUDA_VISIBLE_DEVICES=-1` (`cpu-TRUE-cudaHidden.txt`). Decode (tg128) was never offloaded (per-op too small), so tg128 is unchanged.

| model | lane | pp512 t/s | tg128 t/s |
|---|---|---:|---:|
| Llama-3.2-3B Q8_0 | **CPU true** (CUDA hidden) | **30.93** | 8.82 |
| Qwen3-4B Q8_0 | **CPU true** | **23.64** | 7.44 |
| Qwen3-0.6B Q8_0 | **CPU true** | **167.49** | 45.80 |
| Llama-3.2-3B Q8_0 | CUDA (ngl99) | 3603.6 | 69.29 |
| Qwen3-4B Q8_0 | CUDA (ngl99) | 2652.7 | 54.41 |
| Qwen3-0.6B Q8_0 | CUDA (ngl99) | 11476.0 | 243.76 |
| (contaminated, kept for the record) | CPU+GPU-offload @ngl0 | 740 / 545 / 2585 | — |

Cross-validations: (1) llama CPU prefill llama-bench=30.93 ≈ same-host server=30.6 (independent). (2) Qwen3-4B CUDA decode this session=54.41 vs prior campaign=54.42 — identical environment.

## Same-host CPU, Llama-3.2-3B Q8_0 (`same-host/`)

Greedy/temp=0, CUDA hidden (CPU-only), prompt-prefix cache defeated (unique system nonce), llama `cache_prompt:false`.
Receipt: `cpu-prefill-matrix-llama3b-cpuonly-nocache-20260621T162052Z.json`.

| metric | llama.cpp | Camelid A (default) | Camelid B (gated x86 SIMD) |
|---|---:|---:|---:|
| prefill tok/s (~365–385 tok prompt) | 30.6 | 18.9 (0.62×) | 16.8 (−11% vs A) |
| decode tok/s | 9.08 | 5.97 (0.66×) | 5.48 (−8% vs A) |
| greedy output | identical | identical | identical (parity ✓ A==B) |

Gated x86 packed-rows4/GEMM4 SIMD kernels REGRESS on this Tiger Lake box (vindicates default-off). Output byte-identical.

## Decode thread sensitivity (`same-host/decode-thread-sweep-*.json`, all byte-identical)

CUDA hidden, Llama-3.2-3B Q8, decode = (t_48 − t_1) isolation. `all_parity_identical: true`.

| RAYON_NUM_THREADS | decode tok/s | prefill tok/s |
|---|---:|---:|
| default (Camelid physical-cap = 8) | 5.14 | 19.14 |
| 4 | 5.84 | 18.94 |
| 2 | 6.11 | 18.45 |
| 1 | 6.16 | 19.24 |
| 8 (env, confounded by build_global) | 3.26 | 15.45 |

Decode is best at 1–2 threads (+20% vs default); prefill memory-saturated (flat). Actionable: `CAMELID_THREADS=2` on bandwidth-starved hosts = bit-identical +20% decode. Changing the default would over-fit this box (servers want more threads), so no default change is made.

## GPU-confounded run (kept as evidence of GPU default path)

With only `CAMELID_CUDA_RESIDENT_DECODE=0` (prefill still on CUDA): Camelid GPU prefill = 905–998 tok/s (>2× llama CPU prefill). Confirms Camelid's shipped GPU batched-prefill is excellent and is the DEFAULT path on a GPU box.

## Prior in-repo campaigns harvested (not re-run)

- GPU decode (Qwen3-4B Q8, RTX 3060): Camelid split-K 41.6 t/s vs llama 54.4 t/s (0.77×) low-ctx; 26.2 vs 51.1 (0.51×) at ctx~1881. `q8_gemv` ~76% DRAM BW ("already efficient"). Residual gap parity-locked (coalesced K/V re-associates the dot, flips near-tie greedy tokens). [`qa/perf/decode-attention-campaign.md`]
- GPU prefill batching shipped: 2.56–3.16× (Qwen3 0.6B/1.7B/4B). [`qa/perf/qwen3-cuda-resident-phase3-findings.md`]
- CPU characterization (2026-06-20, same box): decode memory-bandwidth-bound at ~33% of peak; VNNI +4.5% (reverted; AVX-512 downclock); `target-cpu=x86-64-v3` + fat-LTO already shipped (the historical "+39%"). [`qa/evidence-bundles/cpu-perf-characterization-20260620/`]
