# gemma4 CUDA-resident lane: the tied head uploaded raw wire into repacked-layout GEMVs

`gemma-4-E2B-it-Q4_0.gguf` decoded as `passe dép oficialmenteynam shalthapp lenghtynam`
on the CUDA-resident lane where the CPU gemma4 runtime said `Paris`, reproducibly, on an
RTX 3060 Laptop. The lane loaded without error and produced fluent-looking wrong tokens —
it failed silently rather than refusing.

**The Q4_0 projections were never the defect.** `q4_0_gemv` and its repack are correct.

## Root cause

`Gemma4CudaResident::load` picks a tied-head lane from `token_embd`'s format and uploads
the weight. The Q8_0 arm repacked to SoA; the **Q4_K and Q6_K arms `clone_htod`'d the raw
GGUF wire**. Neither of those kernels reads the stock wire:

| Lane | What the GEMV indexes | What it was fed |
|---|---|---|
| `q8_gemv` | SoA split (`q8_wire_to_soa`) | ✅ SoA |
| `q4k_gemv` | quant-byte **swizzle** (`swz_q4k_blocks`) — each aux lane's four stride-8 bytes as one aligned `i32` for `__dp4a` | ❌ raw 144 B wire |
| `q6k_gemv` | **224 B padded** stride (`pad_q6k_blocks`) | ❌ raw 210 B wire |

Every other resident lane in the tree routes through `cuda_resident::repack_for_lane`,
which applies exactly these. The gemma4 head bypassed it.

A Q4_0-quantized gemma4 export carries a **Q4_K `token_embd`**. So the E2B Q4_0 row ran
`q4k_gemv` over unswizzled bytes: correctly-addressed but **wrongly-paired** nibbles. The
hidden states stayed correct and only the logits were garbage — which is exactly why the
output looked fluent instead of crashing.

The Q6_K arm was the same defect one step worse: the 210-vs-224 stride mismatch also reads
past the end of the allocation.

### Why this survived so long

Head lane is a property of `token_embd`, which no admission or coverage check inspects
(`nvfp4_cuda_lane_check` covers layer *projections*, correctly — the head is simply not in
its remit):

| Row | `token_embd` → head lane | Before |
|---|---|---|
| E4B Q8_0 — *the bring-up row* | Q8_0 → `Q8_0` | ✅ correct |
| E2B Q8_0 | Q8_0 → `Q8_0` | ✅ correct |
| **E2B Q4_0** | **Q4_K → `Q4K`** | ❌ **garbage** |
| 26B Q4_0 | Q6_K → `Q6K` | ❌ broken |
| E4B NVFP4 | Q6_K → `Q6K` | ❌ broken |

**The only gemma4 row ever validated on this lane is the one whose head happened to be
Q8_0.** Every row with a K-quant head was broken.

Note what did *not* catch it: `q4k_gemv` has a passing bit-parity unit test, and so does
`q4_0_gemv`. Both kernels were correct throughout. **Kernel parity is not row parity** — a
per-kernel test cannot see a layout mismatch at the upload site.

## Fix

All three head lanes route through one function, `gemma4_head_upload`, mirroring
`repack_for_lane`. Two kernel parameter comments that claimed "raw 210-byte Q6_K wire" and
"RAW 144-byte Q4_K wire" — contradicting their own `WIRE` constants, and the most likely
reason the head was wired this way — are corrected.

`gemma4_head_upload_matches_each_lane_gemv_layout` pins each lane's layout with explicit
index arithmetic rather than by calling the repack helpers back, so it still fails if a
helper is later changed to agree with a broken caller. Verified both directions: **fails**
with raw upload restored, **passes** with the fix.

## Receipt: greedy parity, CUDA-resident vs CPU gemma4 runtime

`gemma-4-E2B-it-Q4_0.gguf`, greedy, gemma chat template, one engine resident at a time.
Capture: `run-q4_0-parity.sh` → `q4_0-parity.json`.

| Prompt | Token-identical |
|---|---|
| Name the capital of France in one word. | ✅ |
| What color is the sky on a clear day? | ✅ |
| Name the largest ocean on Earth. | ✅ |
| What is 2 + 2? | ✅ |
| List three primary colors. | ✅ |

**5/5 legs token-identical (`all_pass: true`)** — full token streams, not just first-token
argmax. The in-tree `gemma4_cuda_matches_cpu_greedy` also passes on this row.

End-to-end through `serve` on the surface the defect was reported on, with the lane
actually resident (1557 MiB VRAM in use):

| | Output |
|---|---|
| Before | `passe dép oficialmenteynam shalthapp lenghtynam` |
| **After** | **`Paris`** |

## Scope

Measured on RTX 3060 Laptop (6 GB), Windows 11, `gemma-4-E2B-it-Q4_0.gguf`. The **Q6_K**
head fix is proven by unit test and by inspection against `pad_q6k_blocks`, **not** by a
row receipt — no Q6_K-head gemma4 row fits this card (26B Q4_0, E4B NVFP4). No throughput
claim is made.

Serving a Q4_0 gemma4 row on the CUDA lane also requires the admission predicate on
`feat/gemma3-cuda-resident`, which pins admission to Q8_0; relaxing that pin onto the
receipt above is tracked on that branch, not here. This PR is the correctness fix alone,
and applies regardless of which rows are admitted.
