# K-quant GEMV v2 + MMA verify lane — measurement record

Branch: `perf/metal-mc-gemv-spec-verify` (session branch
`perf/metal-mc-gemv-spec-verify-session`).
Host: Mac16,10 / Apple M4 / 16 GB, macOS 25.5.0.
Model: `Qwen3-4B-Q4_K_M.gguf` (Q4_K 216 tensors, Q6_K 37 incl. `ffn_down`,
`attn_v`, and the 151936-row `token_embd`/output head).
Binary: one build, gate flipped between arms (`CAMELID_KQUANT_V2`).

## Headline

Every run emits 128 tokens; rates are the emitted-token rate.

| arm | before | after | delta |
| --- | --- | --- | --- |
| plain greedy decode | 16.82 t/s | 21.13 t/s | +26% |
| speculative, code (draft γ=6) | 13.27 t/s | 25.67 t/s | +93% |
| verify cost per round (γ=6) | 289.7 ms | 103.4 ms | 2.8x cheaper |

The speculative row is apples-to-apples: both arms report **identical
acceptance (70.3%) and identical mean accepted tokens/round (5.08)**, i.e.
the same token stream, so only the cost of producing it changed.

Speculation flipped from losing to winning against this machine's own plain
decode: S_sync 0.79 -> 1.21 (code), and the MMA lane isolated at fixed γ=8
inside the same build is 14.55 -> 20.91 t/s (+44%).

Every run in the matrix reports LOSSLESS: the speculative token stream
equals that run's own plain greedy stream.

## Where the time actually went (measured, not assumed)

A microbench (`kbench`) extracts the shader from the worktree at runtime and
times the kernels at production shapes, cross-checking every candidate
against a reference dispatch bit-for-bit.

At the Qwen3-4B up-projection shape (rows 9728, n_sb 10):

| probe | ms | effective GB/s |
| --- | --- | --- |
| math-free weight-stream probe, same geometry | 0.157 | 89 |
| v1 single-token GEMV | 0.277 | 51 |
| v1 with the ordered f32 tail removed | 0.203 | 69 |
| v1 with the shuffle reduction removed | 0.251 | 56 |
| **v2 single-token GEMV** | **0.257** | **55** |

The one-lane ordered f32 tail — not the weight stream — was the dominant
cost of a single-token K-quant GEMV. v2 spreads those nine independent
chains (eight `sums[l]`, one mins) over nine lanes.

Multi-column, same shape:

| k | v1 mc ms/col | v2 mc ms/col | MMA ms/col |
| --- | --- | --- | --- |
| 4 | 0.296 | 0.205 | 0.053 |
| 8 | 0.266 | 0.186 | 0.054 |
| 16 | 0.249 | 0.167 | 0.045 |

The MMA kernel's cost is roughly FLAT in the window width (~2x one single
GEMV for the whole window), because the weight stream is paid once and the
per-column dot moves onto the simdgroup-matrix units.

Two secondary findings, both from ablation:

- The v1 mc kernel's `k*n_sb*9` threadgroup scratch reached 21.9 KB at
  (k=16, n_sb=38), collapsing occupancy to one threadgroup per core:
  0.20 -> 0.375 ms/column. v2 walks super-blocks in chunks against a fixed
  scratch budget.
- `q4k_scale_min` decoded three u32s a byte at a time; every caller strides
  144/176 bytes off a 16-byte-aligned buffer, so aligned loads are valid.
  This one is bit-identical to v1 and is the only change on the DEFAULT path.

## Exactness

The v2 pair is bit-identical **within itself** at every window width:
each column of `q4k_linear_simd_mc_v2` and of `q4k_linear_mma_mc_v2` equals a
`q4k_linear_simd_v2` single-token dispatch, and likewise for Q6_K. That is
precisely the losslessness contract, since a speculative verify must agree
with v2 plain decode.

Why the MMA path stays exact: every operand is an integer in
exactly-representable range. Q4_K stages `w16 = sc*q <= 63*15 = 945` and
`|y| <= 128` in half; products are formed in f32 from 11-bit mantissas, and
the f32 accumulation is bounded by `32*945*128 = 3.87M < 2^24`. Q6_K's
`w16` reaches `127*32 = 4096`, past half's exact-integer range, so that lane
stages f32 throughout. The integer half is computed in 16-bit where products
are `<= 1920` and four-term accumulators `<= 7680`; scales/mins are applied
once per accumulator, which by distributivity gives the same integers as the
v1 per-element form.

The library compiles with fast math OFF so the ordered tail is strict IEEE
program-order arithmetic wherever it appears — the pair's mutual identity
does not rest on the optimizer choosing the same contraction twice.

v2 is **not** bit-identical to v1 (v1's compiled tail contracts differently;
the two differ by ~1 ulp per output). Greedy decode of a quantized model is
chaotic, so that ulp changes the emitted text after a few tokens. This is
why the lane is gated and carries no support claim yet.

Tests: `metal_kquant_v2_pair_bit_identical`, `metal_kquant_q6k_v2_bit_identical`
(single vs mc vs MMA, k in {2,3,8,16}, over a chunked-scratch deep-`n_sb`
shape and a ragged 130-row shape exercising the 8-row tile guard). The
pre-existing v1 identity test is unchanged and still passes.

## Reference tracking

Against `llama-server` (same GGUF, greedy, raw completion), longest common
prefix of the generated text:

| lane | tracks llama.cpp | tok/s |
| --- | --- | --- |
| v1 (branch HEAD kernels) | 30 chars | 17.05 |
| v2 | 186 chars | 21.46 |

v2 follows the reference *longer* than v1 does on this prompt. One prompt at
64 tokens is suggestive, not proof — but it refutes the worry that v2's ulp
differences are a degradation.

## Rejected by measurement

- **Flattening (row, unit) across 4 rows per simdgroup.** At n_sb=10 a
  32-lane simdgroup running one row leaves 24 of 64 lane slots idle (37%),
  and the hypothesis was that packing them would help. It measured *worse*
  at every shape (0.257 -> 0.277 at 9728x10; 0.221 -> 0.295 at 2560x38):
  the M4 retires the idle slots cheaply, while the flattening adds a
  division per unit and a threadgroup round-trip for the tail. Reverted.
- **Simdgroup striping of the mc kernel** (already rejected upstream on the
  M3 at 56e5b06b) reproduced as a wash-to-regression here.

## Measurement hygiene notes

- After ~3 hours of continuous benchmarking this M4 downclocks: the
  *unchanged* kernel drifted 0.257 -> 0.418 ms. Kernel microbenchmarks taken
  in that state are worthless; re-measure a known baseline before trusting
  any ablation. The end-to-end matrix is robust to this because it
  interleaves before/after arms in one pass (drift penalizes the later
  "after" arm, making the reported gains conservative).
- Spec throughput is only comparable across arms when acceptance matches.
  v2's ulp differences changed the text on the repetitive prompt, dropping
  n-gram acceptance 58.7% -> 15.9% — that arm's spec numbers are NOT a
  kernel comparison, and are excluded from the headline.
- `llama-cli` in this build rejects `-no-cnv`, silently falls into
  interactive chat, and streams forever (it wrote 10.5 GB before being
  killed). Use `llama-server` with a bounded `n_predict`, and cap captures
  with `head -c`.

## Gates

- `CAMELID_KQUANT_V2=1` — route Q4_K/Q6_K resident GEMVs through the v2
  lane. Default OFF. Latched once per process.
- `CAMELID_KQUANT_MMA=0` — inside v2, send 4+-column windows back to the
  scalar mc kernel (A/B only). Default on within v2.

## Not done yet

- Q5_K has no v2 sibling (falls back to v1 kernels; correct but unaccelerated).
- Model-level parity receipts for the v2 lane, which is what a support-ledger
  promotion would require.
- The drafter is now the larger half of a speculative round on prose
  (~68 ms/round drafting vs ~101 ms verifying at γ=6).
