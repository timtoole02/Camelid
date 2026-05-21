# Ubuntu x86 Q8 DPBUSD parity push evidence

Run window: 2026-05-21T08:53Z on the canonical Ubuntu x86_64 host.

## Host action and topology

Used the required SSH command successfully: `ssh -o IdentitiesOnly=yes -i /Users/timtoole/Documents/cert/ubuntu.pem ubuntu@16.146.143.184`.

Topology from `logs/preflight.log`: Intel Xeon Platinum 8488C, 16 logical CPUs, 8 physical cores, 2 threads/core, one socket, one NUMA node, CPUs 0-15. Benchmarks used `taskset -c 0-15`, `RAYON_NUM_THREADS=16`, `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, and llama.cpp `--threads 16`.

## Hot kernels

- llama.cpp hot kernel: `ggml/src/ggml-cpu/amx/mmq.cpp`, `(anonymous namespace)::tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, 1, 64, 32>::apply`; this run's `perf/llama-bench-tg16-perf-report.txt` shows 44.27% under `ggml::cpu::amx::tensor_traits::compute_forward`. Secondary llama.cpp hot path is `tinygemm_kernel_amx<block_q8_0, block_q8_0, float, 32, 0>` through `ggml_backend_amx_mul_mat`.
- Camelid hot path: `src/inference.rs`, `camelid::inference::q8_0_packed_rows4_dot`; mixed same-host perf record `perf/dpbusd-perfrecord-r1-report.txt` shows 24.50% self in that function with the DPBUSD gate enabled.

## ASM comparison

- llama.cpp asm: `asm/llama-tinygemm-vnni-q8q8-1x64.asm`, `asm/llama-tinygemm-amx-q8q8.asm`, and `asm/llama-instruction-evidence.txt`. The hot VNNI kernel uses repeated `vpdpbusd` with broadcasted activation bytes and packed Q8_0 weights; the AMX path uses tile load/store instructions.
- Camelid asm: `asm/camelid-q8-rows4-dot-i8-matmul.asm` and `asm/camelid-dpbusd-instruction-evidence.txt`. The retained slice uses `vpabsb`, masked `vpsubb`, and repeated `vpdpbusd`, replacing the prior default-off DPWSSD path for this gate.

## Benchmarks

Same-host guard: Llama-3.2-3B-Instruct-Q8_0, `max_tokens=16`, warmup=1, r5, marker guard required and passed.

- Retained DPWSSD baseline r5: Camelid total 423.29 ms, TTFT 423.17 ms; local llama.cpp total 356.43 ms, TTFT 167.70 ms.
- DPBUSD candidate r5: Camelid total 413.70 ms, TTFT 413.55 ms; local llama.cpp total 353.97 ms, TTFT 165.29 ms.
- Retained effect: DPBUSD improved Camelid total by 9.59 ms / 2.27% vs DPWSSD and TTFT by 9.62 ms. The <0.4 s target is still not reached; candidate is 13.70 ms over target.

## Perf counters

Counters are from r3 `perf stat` harness runs and include the Node harness plus child servers.

- DPWSSD r3: task-clock 88,128,736,487; cycles 282,375,962,122; instructions 270,934,353,846; branches 40,954,643,044; branch-misses 112,516,256; context-switches 319,483; cpu-migrations 28,946; page-faults 2,517,623.
- DPBUSD r3: task-clock 87,883,982,351; cycles 281,945,157,037; instructions 255,975,638,897; branches 38,055,939,616; branch-misses 108,516,133; context-switches 317,345; cpu-migrations 26,229; page-faults 2,499,563.

## Retain/reject

Retained: default-off `CAMELID_X86_Q8_PACKED_ROWS4_AVX512VNNI_DPBUSD_DOT=on` in `src/inference.rs` for rows4/I8 Q8 dot. It is bounded to AVX512F/BW/VNNI x86 hosts and remains lower priority than the target until further same-host evidence promotes it.

Rejected: no additional gate promotion and no default-on change; the slice does not reach the 0.4 s target.

## Gates

Green gates in `gates/`: `cargo fmt -- --check`, `cargo test x86_q8_avx512vnni_dpbusd_packed_rows4_i8_matches_scalar_dot`, `cargo test x86_q8`, `cargo test q8_0`, `node scripts/test-bench-llama3-same-host.mjs`, full `cargo test -q`, and `scripts/check-public-scrub.sh` on this bundle. The full Ubuntu test pass also includes a platform-aware correction for an existing mac-only scheduler counter assertion.

## Push status

Pending at evidence creation time.
