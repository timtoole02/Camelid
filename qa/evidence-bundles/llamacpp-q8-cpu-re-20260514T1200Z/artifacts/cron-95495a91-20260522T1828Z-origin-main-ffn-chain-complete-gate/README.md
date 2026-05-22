# cron-95495a91 origin/main FFN-chain complete-gate same-host evidence

Retain as Ubuntu x86_64 Q8 route/timing evidence for the default-off FFN decode-chain slice on current `origin/main`.

Result:
- Host/head: Ubuntu x86_64, `dd3c7923e606bebebda8e5780fad899fba493fd7`.
- Gates: `CAMELID_PROFILE=experimental`, `CAMELID_X86_Q8_REPACK=on`, `CAMELID_X86_Q8_KERNEL=avx2`, `CAMELID_X86_Q8_OUTPUT_DECODE_OWNER=on`, `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER=on`, `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING=on`, `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION=on`, `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT=on`, `CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER=on`, `CAMELID_X86_Q8_FFN_DECODE_CHAIN=on`, `CAMELID_Q8_SCHED_TELEMETRY=on`, `CAMELID_STREAM_TIMING_DIAGNOSTICS=on`.
- One-token parity passed against llama.cpp for prompt `Reply with exactly one capital letter: C`: prompt tokens matched, generated token `[66]` matched, text `c` matched.
- Same-host 4-token timing: Camelid TTFT `5703.86 ms`, backend first content `1823 ms`, backend generate `2008 ms`; llama.cpp TTFT `162.56 ms`, total `284.37 ms`.
- Route proof: `ffn_decode_chain_taken=112`, `ffn_decode_chain_total_us=125561`, `ffn_decode_chain_down_us=42778`, `logits.x86_output_decode_owner.calls=4`.

Boundary: bounded same-host evidence only; it does not promote support, production throughput, portability, larger context, neighboring rows, or broad Llama-family claims.

Next exact action: move the `ffn_down_decode_consumer` dependency into either the execution-plan preset for FFN decode-chain or the route guard/error telemetry, then rerun this artifact's complete-gate command as a regression check.

Files:
- `parity-one-token.json`
- `same-host-bench.json`
- `same-host-bench.stream-summary.json`
- `route-summary.json`
- `logs/host.txt`
