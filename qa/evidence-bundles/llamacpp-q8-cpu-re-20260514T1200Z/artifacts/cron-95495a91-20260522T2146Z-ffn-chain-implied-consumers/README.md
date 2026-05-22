# cron-95495a91 FFN-chain implied-consumer evidence

Retain as Ubuntu x86_64 Q8 route/timing evidence for making the default-off FFN decode-chain gate self-contained.

Result:
- Host/head: Ubuntu x86_64, `f1433ed0d8425858595be1b0cbe0e8a7f4936ed1` plus local patch.
- Gates: `CAMELID_PROFILE=experimental`, `CAMELID_X86_Q8_REPACK=on`, `CAMELID_X86_Q8_KERNEL=avx2`, `CAMELID_X86_Q8_OUTPUT_DECODE_OWNER=on`, `CAMELID_X86_Q8_FFN_DECODE_CHAIN=on`, `CAMELID_Q8_SCHED_TELEMETRY=on`, `CAMELID_STREAM_TIMING_DIAGNOSTICS=on`.
- Lower consumer gates intentionally unset: `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER`, `CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER`, `CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER`, and `CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER`.
- One-token parity passed against llama.cpp for prompt `Reply with exactly one capital letter: C`: prompt tokens matched, generated token `[66]` matched, text `c` matched.
- Same-host 4-token timing: Camelid TTFT `6595.95 ms`, backend first content `2628 ms`, backend generate `2835 ms`; llama.cpp TTFT `285.90 ms`, total `410.20 ms`.
- Route proof: `ffn_decode_chain_taken=112`, `ffn_down_decode_consumer_taken=112`, `ffn_decode_chain_total_us=128318`, `ffn_decode_chain_down_us=45948`, `logits.x86_output_decode_owner.calls=4`.

Boundary: bounded same-host evidence only; this is not a support, portability, production-throughput, RSS, default-on, or broad Llama-family claim.

Next exact action: use this self-contained chain gate for a paired-dot/VNNI down-chain A/B, then reject or retain based on route-level `ffn_decode_chain_total_us` and same-host TTFT versus this artifact.

Files:
- `parity-one-token.json`
- `same-host-bench.json`
- `same-host-bench.stream-summary.json`
- `route-summary.json`
- `logs/host.txt`
- `logs/env.txt`
