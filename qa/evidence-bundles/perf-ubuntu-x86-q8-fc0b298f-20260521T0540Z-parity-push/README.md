# Ubuntu x86 Q8 DPWSSD parity push evidence

Run window: 2026-05-21T05:40Z-05:52Z on the canonical Ubuntu x86_64 host.

## Host action and topology

Canonical SSH access was used for this run and succeeded. The host reported Intel Xeon Platinum 8488C, 16 logical CPUs, 8 physical cores, 2 threads/core, one socket, one NUMA node, CPUs 0-15, and about 123 GiB RAM. Raw topology is in `logs/preflight.log`.

## Hot kernels

- llama.cpp hot kernel: `ggml/src/ggml-cpu/amx/mmq.cpp`, `(anonymous namespace)::tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, 1, 64, 32>::apply`; perf-only `llama-bench` sample showed 49.32% self. Secondary llama.cpp hot kernel: `tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>` at 8.60% self through `ggml_backend_amx_mul_mat`.
- Camelid hot kernel: `src/inference.rs`, `camelid::inference::q8_0_packed_4x8_block_avx2` at 14.93% self and `q8_0_packed_rows4_dot` at 11.26% self in mixed same-host perf record.

## ASM comparison

- llama.cpp evidence: `asm/llama-tinygemm-vnni-q8q8-1x64.asm`, `asm/llama-tinygemm-amx-q8q8.asm`, and filtered `asm/llama-vnni-instruction-evidence.txt`; the hot VNNI kernel uses repeated `vpdpbusd` with broadcasted activation groups and packed Q8_0 weights.
- Camelid baseline evidence: `asm/camelid-avx2-kernel.asm`; the current hot kernel uses AVX2 `vpmaddubs`/`vpmaddwd` style lowering in `q8_0_packed_4x8_block_avx2`.
- Camelid retained slice evidence: `asm/camelid-dpwssd-kernel.asm` and `asm/camelid-dpwssd-instruction-evidence.txt`; the new default-off kernel uses AVX512BW sign-extension plus AVX512VNNI `vpdpwssd`/`vpmaddwd` lowering for rows4/I8 Q8 dot blocks.

## Benchmarks

Same host, taskset CPUs 0-15, `RAYON_NUM_THREADS=16`, `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, Llama-3.2-3B-Instruct-Q8_0, warmup=1, max_tokens=16, llama.cpp `--threads 16`, marker guard passed.

- Baseline A r3: Camelid total 425.88 ms, TTFT 425.69 ms, backend generate 365.33 ms; llama.cpp total 345.32 ms, TTFT 157.75 ms.
- DPWSSD B r5: Camelid total 418.88 ms, TTFT 418.66 ms, backend generate 360.20 ms; llama.cpp total 364.37 ms, TTFT 173.30 ms.
- Baseline A2 r5: Camelid total 425.78 ms, TTFT 425.62 ms, backend generate 366.00 ms; llama.cpp total 354.07 ms, TTFT 165.45 ms.

Retained wall-clock effect: DPWSSD improved Camelid total by 7.00 ms vs A and 6.90 ms vs A2, and backend generate by 5.13 ms vs A and 5.80 ms vs A2. The <0.4 s target is not reached.

## Perf counters

Counters cover the same-host harness plus child servers and are profiler evidence, not wall-clock retain evidence because `perf stat` slows Camelid.

- Baseline perfstat r3: task-clock 89003786145, cycles 283471943405, instructions 304164505206, branches 48263914766, branch-misses 128619239, context-switches 315786, cpu-migrations 29388, page-faults 2542925, elapsed 17.054146457 s.
- DPWSSD perfstat r3: task-clock 89202554026, cycles 283356800296, instructions 283975197062, branches 41508086729, branch-misses 130426864, context-switches 320553, cpu-migrations 28328, page-faults 2532304, elapsed 17.116453432 s.

## Retain/reject

Retained: default-off `CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT=on` in `src/inference.rs`.

Rejected in this run:

- `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on`: r3 regressed to Camelid total 462.05 ms / backend 399.33 ms.
- FFN-down GEMM4 row-group + AVX2 with DPWSSD: r3 was 427.61 ms / backend 367.00 ms, not a win.
- Broad optional decode/matmul gates with DPWSSD: r3 regressed to 642.17 ms / backend 584.00 ms.

## Gates

Gate outputs are in `gates/`. This bundle is bounded performance evidence for this host/model/request only; it does not widen support or default-on claims.
