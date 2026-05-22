# cron-95495a91 Ubuntu x86 Q8 same-host benchmark

- Generated: 2026-05-22T16:18:22Z
- Host: Linux x86_64
- Repo head: 84a4a83bf881550f29dcea8349c2284439dfd900
- PR #74 visible on origin/main: yes
- Model: $UBUNTU_HOME/models/Llama-3.2-3B-Instruct-Q8_0.gguf
- llama.cpp server: $UBUNTU_HOME/work/llama.cpp-clean-20260517/build/bin/llama-server
- llama.cpp repo head: 4f0e43da6f8f6e9390d88409610098ec2d2dc5c7
- CARGO_TARGET_DIR: $UBUNTU_HOME/work/camelid-targets/backend-95495a91
- Path placeholder: `$UBUNTU_HOME` denotes the canonical Ubuntu worker home used during the retained run.

## Result

- Camelid avg TTFT ms: 8669.83
- Camelid avg stream total ms: 8670.1
- Camelid avg decode tok/s estimate: 17946.72
- llama.cpp avg TTFT ms: 309.52
- llama.cpp avg stream total ms: 502.09
- llama.cpp avg decode tok/s estimate: 25.96
- TTFT delta vs llama.cpp: 2701.06%
- Total elapsed delta vs llama.cpp: 1626.8%
- Guardrails passed: true

## Retain/reject

Retain as current-main same-host timing artifact for the Llama 3.2 3B Instruct Q8_0 row. It is not a support-contract promotion and not a full parity claim; marker guard only proves the benchmark prompt contract survived both engines.

## Inspected files

- scripts/bench-llama3-same-host.mjs
- scripts/test-bench-llama3-same-host.mjs
- src/inference/q8_runtime.rs
- src/inference.rs
- $UBUNTU_HOME/work/llama.cpp-clean-20260517/tools/server/server.cpp
