# cron-95495a91 origin/main FFN-chain missing-gate same-host evidence

Retain as negative route evidence only. This run intentionally enabled the new FFN decode-chain gate with gate/up and output decode-owner opt-ins, but did not enable `CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER`.

Result:
- Host/head: Ubuntu x86_64, `dd3c7923e606bebebda8e5780fad899fba493fd7`.
- One-token parity passed against llama.cpp for prompt `Reply with exactly one capital letter: C`: prompt tokens matched, generated token `[66]` matched, text `c` matched.
- Same-host 4-token timing: Camelid TTFT `8747.51 ms`, backend generate `3157 ms`; llama.cpp TTFT `162.87 ms`.
- Route proof: `ffn_decode_chain_taken=0`; the run used split `ffn_gate_up.decode_fused_activation` and `ffn_down.x86_vnni_decode_consumer` routes.

Measured reason: `try_x86_q8_ffn_decode_chain_path` requires `runtime_plan.q8.ffn_down_decode_consumer`; enabling only `CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE=on` allows the down leg elsewhere, but does not satisfy the chain guard.

Files:
- `parity-one-token.json`
- `same-host-bench.json`
- `same-host-bench.stream-summary.json`
- `route-summary.json`
- `logs/host.txt`
