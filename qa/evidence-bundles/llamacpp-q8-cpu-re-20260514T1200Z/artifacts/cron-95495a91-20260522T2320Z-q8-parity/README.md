# Cron 95495a91 Ubuntu x86 Q8 Parity Slice 2026-05-22T23:20Z

This bundle records a same-host Camelid versus llama.cpp Llama 3.2 3B Instruct Q8_0 timing slice on Ubuntu Linux x86_64 at `f1433ed0d8425858595be1b0cbe0e8a7f4936ed1`.

Method:

- Host gate: `uname -sm` returned `Linux x86_64`.
- Disk guard was run before cargo; cargo used `CARGO_TARGET_DIR=<validation-host-home>/work/camelid-targets/backend-95495a91`.
- Model: `<validation-host-home>/models/Llama-3.2-3B-Instruct-Q8_0.gguf`.
- Camelid gates: `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on` and `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR=on`.
- Harness: `node scripts/bench-llama3-same-host.mjs --backend http://127.0.0.1:19381 --llama-url http://127.0.0.1:19383 --backend-bin <validation-host-home>/work/camelid-targets/backend-95495a91/release/camelid --llama-server <validation-host-home>/work/llama.cpp-fc0b298f-20260520T0325Z/build/bin/llama-server --model <validation-host-home>/models/Llama-3.2-3B-Instruct-Q8_0.gguf --model-id llama32-3b-q8-current-main --row-id llama32_3b_instruct_q8_0 --max-tokens 4 --warmup 0 --repeats 1 --threads 16 --out same-host-vnni-rawptr-t16-n4.json`.

Measured result:

| Engine | TTFT ms | Total elapsed ms | Decode tok/s | ms/token after first | Completion token estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Camelid rawptr VNNI gates | 7662.03 | 7662.20 | 23757.49 | 0.04 | 4 |
| llama.cpp CPU | 144.51 | 267.62 | 32.49 | 30.78 | 4 |

Interpretation:

- This is a negative parity/performance slice for promotion: llama.cpp was far faster on TTFT and total elapsed for the same bounded workload.
- The marker guard recorded `passed=false`; both engines streamed non-empty output, but neither measured run contained the requested exact `CMLD-BENCH` marker. This bundle therefore does not support deterministic-output parity or any support/default-on claim.
- Camelid emitted all measured content after first content quickly enough that post-TTFT decode tok/s is not comparable to llama.cpp token-level decode; retain decisions should use TTFT/total elapsed plus a deterministic-output guard until the harness has token-grounded Camelid accounting.

Files:

- `same-host-vnni-rawptr-t16-n4.json`: machine-readable same-host harness output.
- `same-host-vnni-rawptr-t16-n4.log`: command log with llama.cpp CPU feature disclosure and human summary.
- `SHA256SUMS`: checksums for this bundle.
