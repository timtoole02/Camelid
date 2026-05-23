# cron-0719640b attention-output GEMM4 same-host benchmark

Scope: default-off Ubuntu/Linux x86_64 Q8 attention-output prefill follow-up on `8f193219362e41f6f76a8058b2053113e79c20d9`.

Result:
- Targeted remote Linux x86_64 validation ran successfully for `q8_attention_output_gemm4`; both unit tests passed.
- One-token parity against llama.cpp passed for `Reply with exactly one capital letter: C`: prompt tokens matched, generated token matched, and generated text matched (`c`).
- The candidate route was exercised only when measured runs forced fresh prefill with unique prompts. On the first measured candidate run, telemetry recorded `attention_output.x86_gemm4_prefill_row_group` across 28 layers, 28 calls total, `rows=75`, `input_width=3072`, `output_width=3072`, and `elapsed_us=150332`.
- Same-host 4-token unique-prompt timing rejected the candidate. Baseline Camelid measured `TTFT 2957.05 ms` / `total 2957.43 ms`; candidate Camelid measured `TTFT 2989.27 ms` / `total 2989.67 ms`. The candidate regressed baseline by `32.22 ms` TTFT and `32.24 ms` total elapsed.
- llama.cpp remained substantially faster on the same host: baseline reference `TTFT 305.42 ms` / `total 418.18 ms`; candidate reference `TTFT 306.54 ms` / `total 419.91 ms`.

Decision:
- Reject this benchmark for retention. Keep the telemetry and route-surfacing implementation, but do not make a performance, support, or default-on claim for the attention-output GEMM4 prefill path from this run.
- The baseline attention-output packed-rows4 fallback did not emit a comparable route entry in stream timing diagnostics, so the route proof in this artifact is candidate-only.

Public artifact contents:
- `README.md`
- `summary.json`
