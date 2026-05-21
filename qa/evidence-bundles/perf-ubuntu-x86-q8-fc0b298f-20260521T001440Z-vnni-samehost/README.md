# Ubuntu x86 Q8 AVX512VNNI Rows4 Dot Evidence

## Scope

This bundle records the 2026-05-21 UTC Ubuntu x86_64 same-host performance slice for Llama 3.2 3B Instruct Q8_0. The retained code path is the default-off `CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT=on` rows4/I8 dot-kernel experiment. It is not a support, portability, accelerator-backend, or default-on claim.

## Host shape

- CPU: Intel Xeon Platinum 8488C
- Topology: 16 logical CPUs, 8 cores, 2 threads per core, 1 socket
- NUMA: one node, node0 CPUs 0-15, about 123.8 GiB RAM
- Relevant features: AVX2, AVX512F, AVX512BW, AVX512VNNI, AMX_INT8
- Thread policy used for retained measurements: 16 threads, NUMA node 0 CPU and memory binding

## Hot-path evidence

- llama.cpp hot kernel: `ggml/src/ggml-cpu/amx/mmq.cpp`, `(anonymous namespace)::tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, 1, 64, 32>::apply(...)`
- llama.cpp perf record: 54.62% in the VNNI tinygemm kernel, with the AMX tinygemm path next at 11.93%.
- Camelid hot function before the retained slice: `src/inference.rs`, `camelid::inference::q8_0_packed_rows4_dot`.
- Camelid retained function: `src/inference.rs`, `q8_0_packed_4x8_block_avx512vnni_dpwssd`.
- ASM comparison: llama.cpp emits broad unrolled AVX512 VNNI `vpdpbusd`; Camelid now emits AVX512VNNI `vpdpwssd` for signed Q8 rows4/I8 blocks while preserving the safe fallback.

## Same-host timing

| Path | Repeats | Camelid total ms | Camelid backend generate ms | llama.cpp total ms | Guardrail |
| --- | ---: | ---: | ---: | ---: | --- |
| control: `CAMELID_PROFILE=experimental CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2` | 5 | 430.42 | 369.00 | 365.49 | PASS |
| retained: control plus `CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPWSSD_DOT=on` | 5 | 416.93 | 357.40 | 367.37 | PASS |
| rejected: control plus `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on` | 3 | 438.57 | 378.33 | 368.32 | PASS |

Retained delta versus the r5 control: -13.49 ms total wall time and -11.60 ms backend generate time. The FFN-down VNNI decode slice is rejected because its model-backed same-host wall time regressed despite a synthetic microbench win.

## Perf counters

The paired `perf stat` r3 run adds instrumentation overhead, so it is counter evidence, not retained wall-clock evidence.

| Path | Task clock | Cycles | Instructions | Branches | Branch misses | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 94,294,433,465 | 293,872,953,904 | 304,384,529,116 | 48,820,036,675 | 118,442,321 | 18.067 s |
| retained DPWSSD | 92,811,515,376 | 289,418,074,412 | 289,218,480,396 | 42,406,597,568 | 116,838,126 | 17.945 s |

## Gates

- `scripts/with-rustup-cargo.sh fmt --check`
- `scripts/with-rustup-cargo.sh test x86_q8_avx512vnni_dpwssd_packed_rows4_i8_matches_scalar_dot -- --nocapture`
- `scripts/with-rustup-cargo.sh test ubuntu_experimental_preserves_explicit_x86_q8_gate_opt_ins -- --nocapture`
- `RUSTFLAGS="-C target-cpu=native" scripts/with-rustup-cargo.sh build --release`
