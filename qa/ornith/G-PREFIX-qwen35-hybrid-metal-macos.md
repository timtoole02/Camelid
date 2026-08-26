# ORNITH 9B — hybrid prefix-cache Metal receipt

**Result:** PASS — exact changed-tail prefix reuse with greedy-token parity.

**Date:** 2026-08-15  
**Platform:** Apple M4, 16 GB unified memory  
**Model:** Ornith 1.0 9B Q4_K_M, exact model SHA-256
`5720d1f671b4996481274fffe01868c3c36e87c135cc8538471cc7bd6087b106`  
**Backend:** `metal_resident_qwen35_kquant_runtime`

## Controlled comparison

The two 2,332-token prompts were identical except for an 18-token tail. Both
runs used greedy decoding and produced the same first token ID (`44061`). The
cache-disabled run is the cold control for the changed-tail prompt.

| Run | Common | Reused | Prefilled | Prompt processing | Wall time | First token |
|---|---:|---:|---:|---:|---:|---:|
| Initial cold prompt | — | 0 | 2,332 | 184.597 s | 184.710 s | 44061 |
| Hybrid cache, changed tail | 2,314 | 2,304 | 28 | 2.408 s | 2.522 s | 44061 |
| Cache disabled, changed tail | — | 0 | 2,332 | 186.107 s | 186.227 s | 44061 |

The warm changed-tail request was about **74× faster wall-clock** than its cold
control. The cache decision was `qwen35_hybrid_block_prefix_hit`; four aligned
recurrent checkpoints occupied 210,763,776 bytes. Exact output-token parity
demonstrates that attention KV plus the SSM convolution/recurrent snapshots
restore the same greedy state as a cold prefill.

## Agent-loop implications

The runnable Qwen35 path previously reset and fully prefilled on every Web Code
request. Camelid now preserves stable native tool schemas across active Modify
and Verify phases and reports exact `prompt_reused_tokens`,
`prompt_prefilled_tokens`, common-prefix, block, and checkpoint diagnostics on
each Workspace model step. Host-proven completion is summarized without a final
zero-tool inference.

The first request remains cold. This receipt proves repeated-step acceleration,
not model-load or first-prompt acceleration.

## Real Web Code Goal

The release binary then ran a fresh full-auto Workspace Goal to create
`hello.py`, execute `python3 hello.py`, and verify the exact `hello` output. It
finished `answered`; the host observed exit code 0 and stdout `hello`, then
published the ledger summary without another model inference.

| Request | Prompt | Reused | Prefilled | Total |
|---|---:|---:|---:|---:|
| 1, cold creation | 1,245 | 0 | 1,245 | 100.684 s |
| 2, capture/compile | 1,474 | 1,024 | 450 | 38.005 s |
| 3, requested-path correction | 1,517 | 1,408 | 109 | 12.490 s |
| 4, capture/compile | 1,736 | 1,024 | 712 | 60.100 s |
| 5, required runtime execution | 1,791 | 1,664 | 127 | 13.318 s |

This end-to-end receipt also demonstrates the remaining optimization target:
tool/capsule changes can still move the LCP backward and make some steps prefill
hundreds of tokens, but the agent no longer cold-prefills every complete prompt.
