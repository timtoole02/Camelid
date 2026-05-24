# AVX2 decode-accumulator hoist Ubuntu x86 Q8 retain

- Date: 2026-05-24
- Candidate: `perf(q8): hoist packed rows4 avx2 decode accumulators`
- Feature commit: `a39a493`
- Result: retained inside the existing default-off Ubuntu Linux x86_64 AVX2 lane for a narrow Camelid-on-Camelid wall-clock win; no default-on, portability, or broad throughput claim is made

## What ran

- Release builds on current `main` and the candidate head
- One-token exact-row parity for `hello` on current `main` and the candidate head
- Same-host benchmark with `warmup=0`, `repeats=2`, `max_tokens=8`, unique prompt, and marker guard enabled
- Warmed-cache control rerun on current `main` after the candidate benchmark

## Common benchmark env

- `CAMELID_X86_Q8_REPACK=on`
- `CAMELID_X86_Q8_KERNEL=avx2`

## Measured result

- Current `main` first pass `07c912c`
  - One-token parity: exact
  - Camelid TTFT `8618.40 ms`
  - Camelid total `8618.91 ms`
  - llama.cpp TTFT `320.06 ms`
  - llama.cpp total `525.50 ms`
- Candidate `a39a493`
  - One-token parity: exact
  - Camelid TTFT `4447.75 ms`
  - Camelid total `4448.28 ms`
  - llama.cpp TTFT `317.16 ms`
  - llama.cpp total `524.74 ms`
- Warmed-cache current-`main` control
  - Camelid TTFT `8889.41 ms`
  - Camelid total `8889.69 ms`
  - llama.cpp TTFT `315.95 ms`
  - llama.cpp total `522.21 ms`

## Guardrails

- Prompt tokens matched the llama.cpp reference on current `main` and the candidate head
- Generated token ids matched the llama.cpp reference on current `main` and the candidate head
- Generated text matched the llama.cpp reference on current `main` and the candidate head
- The unique-prompt marker guard passed for every measured run

## Decision

- The warmed-cache control stayed effectively flat on current `main`, so the candidate win is not explained by simple second-run cache warming
- Versus the warmed-cache current-`main` control, the candidate reduced Camelid TTFT by about `50.0%` and total elapsed by about `50.0%`
- llama.cpp remained much faster overall, so this slice retains only the narrow Camelid-on-Camelid improvement inside the existing default-off AVX2 lane
- Keep the lane default-off and evidence-gated while retaining this accumulator-hoist follow-on as a measured Ubuntu x86_64 win within that lane
