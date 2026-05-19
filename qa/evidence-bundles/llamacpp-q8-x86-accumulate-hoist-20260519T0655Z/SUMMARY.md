# llama.cpp Q8 x86 accumulate hoist — 2026-05-19T0655Z

- Host: Ubuntu x86_64 via `ssh -o IdentitiesOnly=yes -i /Users/timtoole/Documents/cert/ubuntu.pem ubuntu@16.146.143.184`; CPU Intel Xeon Platinum 8488C; AVX2 detected by llama.cpp.
- Baseline: `e5b141c37fe9063663344fb5d7df1b7c232bd4bf` (`/home/ubuntu/work/camelid-backend-95495a91-20260519T0431Z-main`).
- Candidate: same HEAD plus dirty `src/inference.rs` slice routing I8 packed rows4 accumulation through `q8_0_packed_rows4_dot_i8_matmul` with AVX2 hoist enabled under existing x86 Q8 gates.
- Bench command shape: `CAMELID_PROFILE=experimental CAMELID_X86_Q8_REPACK=on CAMELID_X86_Q8_KERNEL=avx2 CAMELID_BIN=<baseline|candidate>/target/release/camelid CAMELID_SAME_HOST_BENCH_OUT=<json> node scripts/bench-llama3-same-host.mjs --model /home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf --llama-server /home/ubuntu/work/llama.cpp-clean-20260517/build/bin/llama-server --max-tokens 16 --warmup 1 --repeats <5|10> --threads 8 --require-marker --expected-marker CMLD-BENCH`.
- Parity guard: marker guard passed for all Camelid and llama.cpp measured runs in all four JSON reports.
- Same-host wall-clock results: 5-repeat pair baseline Camelid total 979.83 ms vs candidate 972.84 ms (-0.713%); 10-repeat pair baseline 999.88 ms vs candidate 988.85 ms (-1.103%).
- Same-host reference boundary: candidate remains slower than llama.cpp total elapsed (candidate 972.84/988.85 ms vs llama.cpp 376.89/379.80 ms); retained only as a small current-baseline improvement, not parity with llama.cpp.
- Gates: `cargo fmt --check`; `cargo test q8_avx2_packed_rows4 --lib`; `cargo build --release`; `cargo clippy --lib -- -D warnings`.
- Retain/reject: retain candidate slice under existing default-off x86 experimental gates; next target is profiling the remaining Camelid TTFT gap against llama.cpp.
