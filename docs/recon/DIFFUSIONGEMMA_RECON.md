# DiffusionGemma lane recon (experimental/diffusiongemma)

> [!WARNING]
> **Active development — recon/evidence-only. Not supported.** This document
> describes the `experimental/diffusiongemma` lane. Nothing here is a support
> claim; the public support truth is [`COMPATIBILITY.md`](../../COMPATIBILITY.md)
> and [`STATUS.md`](../reference/STATUS.md). The lane's mission — a Rust-native
> DiffusionGemma runtime — is earned by parity receipts, not claims; until the
> Phase 5 multi-canvas parity artifact exists, the only claim made anywhere is
> "in progress."

This recon supersedes the architecture-level notes in
[`DIFFUSIONGEMMA_26B_A4B_RECON.md`](DIFFUSIONGEMMA_26B_A4B_RECON.md) where the
two disagree: that document predates llama.cpp's diffusion runtime and states no
oracle exists. One now does (see §2), which is what unblocked this lane.

All facts below are verified against one of two sources, named per section:

- **the tracked GGUF** — tensor inventory + metadata extracted by
  `camelid inspect` (unmodified; the metadata layer already represents every
  quant type in the file). Artifacts: `target/dg-recon-20260611T160428Z/
  {tensor-inventory.json, metadata.json, quant-coverage.json, inspect-raw.json,
  make-inventory.py}`.
- **the pinned reference** — llama.cpp source at the pin (§2). File/line
  citations are against that commit.

Credit: the reference runtime, model graph, and sampler recon'd here are the
work of the llama.cpp / ggml authors (ggml-org/llama.cpp PR #24423). Parity for
this lane is defined against that implementation at the pinned commit.

## 1. Tracked artifact

One file is the tracked row; nothing inherits from resemblance:

| | |
|---|---|
| File | `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (16,806,810,336 bytes) |
| Source | `unsloth/diffusiongemma-26B-A4B-it-GGUF` |
| SHA256 | recorded in `target/dg-recon-20260611T160428Z/llamacpp-pin.json` |
| Why this quant | smallest quant published in the official repo (Q5_K_M 19.2 / Q6_K 22.6 / Q8_0 26.9 / BF16 50.5 GB); Q8_0 of a 26B exceeds the dev-machine envelope |

## 2. Pinned reference runtime

llama.cpp master cannot generate from DiffusionGemma. The reference is
**ggml-org/llama.cpp PR #24423** at pinned commit
**`7ea238c68b34ec8c99c28a68b9beed5b150cabef`** (head repo
`danielhanchen/llama.cpp`, branch `diffusion-visual-updates`; PR open at pin
time, so the pin is the commit SHA, never the branch). Local checkout + Metal
build: `/Volumes/Untitled/reference/llama.cpp-dg`. Full build/invocation/
determinism details: `target/dg-recon-20260611T160428Z/llamacpp-pin.json`.

Three binaries matter:

- `llama-diffusion-cli` — end-to-end generation (EB sampler auto-enabled for
  canvas models, `examples/diffusion/diffusion-cli.cpp:310`).
- `llama-diffusion-gemma-eval` — **the per-step logits oracle**: feeds golden
  token ids `[prompt | canvas]` through one forward and dumps raw f32
  canvas-row logits; optional previous-step logits input enables
  self-conditioning. Flash attention off by default. This is the Phase 2/3
  comparator; no instrumentation patch is needed.
- `llama-diffusion-gemma-server` — not used by this lane yet.

## 3. What the GGUF actually is (verified, `metadata.json`)

- `general.architecture = "diffusion-gemma"` (llama.cpp arch string,
  `src/llama-arch.cpp:61`); `general.size_label = "128x2.6B"`; file type 15
  (Q4_K_M).
- 30 layers, `n_embd` 2816, context 262144, **`diffusion.canvas_length = 256`**.
- MoE: 128 experts, 8 used, expert FFN 704; dense (shared-expert) FFN 2112.
- Attention: 16 Q heads; sliding-window pattern **5 SWA : 1 global**
  (`sliding_window_pattern`, global at layers 5,11,17,23,29). Per-layer-type
  dims: SWA layers head dim 256, 8 KV heads, rope base 10000; global layers
  head dim 512, 2 KV heads, rope base 1e6 + proportional rope via a shared
  `rope_freqs.weight` [256]. Sliding window 1024. QK-norm per head; RMS eps 1e-6.
- `final_logit_softcapping = 30.0`. `attention.causal = false`.
- Head is **tied** to `token_embd.weight` (Q6K, [2816, 262144]) — no
  `output.weight` in the file.
- Tokenizer: `tokenizer.ggml.model = "gemma4"`, vocab 262144, merges 514906,
  BOS 2 / EOS 1 / PAD 0 / UNK 3 / **MASK 4**, `add_bos_token = true`,
  `add_space_prefix = false`, chat template 17.5k chars (tool-calling macros).
  **Phase 1 recon verdict:** at the pin this is `LLAMA_VOCAB_TYPE_BPE` with
  pre-type `GEMMA4` (`src/llama-vocab.cpp:2043,506`): SPM-style **rank-based
  BPE** — spaces escaped to `▁`, merges run over raw UTF-8 (no GPT-2 byte
  encoding), pre-split only on newline runs (`[^\n]+|[\n]+`), all-newline
  words emitted directly when in vocab (the file has `\n`,`\n\n`,… tokens),
  `add_bos` force-overridden true, byte fallback to `<0xXX>` tokens (type 6).
  The file's `tokenizer.ggml.scores` are **all -1000.0 placeholders**, so a
  score-greedy SPM merge cannot reproduce it — the merges list is the only
  merge authority. Phase 1 gate: 12/12 cases (raw edges + newline runs +
  byte-fallback emoji + 5 chat-template prompts rendered by llama.cpp minja)
  with 100% token-id match and decode == per-token pieces == detokenize —
  artifact `target/dg-tokenizer-parity-20260611T171928Z.json`, runner
  `scripts/dg-tokenizer-parity.sh` + `scripts/dg-tokenize-dump.cpp`
  (credited), pack `qa/prompt-packs/diffusiongemma-tokenizer-parity-v1.json`.
  One shared-path fix fell out: the tokenizer's multi-space (`▁▁`) deferral is
  now scoped to the score-merge path only — rank-based BPE merges multi-space
  runs into single tokens (e.g. `▁▁`/`▁▁▁`), as the reference does; the
  supported gemma-4 family rows (same tokenizer construction: merges +
  placeholder scores) get the same correction, and their committed packs never
  pinned the old multi-space behavior.
- **No `diffusion.eb_*` keys in the file** — the reference sampler runs on its
  in-code defaults (§6).
- **No vision/video/audio tensors in this file at all** (inventory class
  `vision_video` is empty). The text-only scope is satisfied by the artifact
  itself; there is nothing to skip at load time.

Tensor inventory: 692 tensors, **zero unclassified**
(`tensor-inventory.json`; classes: embedding 1, attention 115, attention_norm
120, ffn_norm 150, dense_ffn 90, moe_router 60, moe_experts 90, region_scalar
60, self_conditioning 4, rope 1, final_norm 1). Notable shapes: fused
`ffn_gate_up_exps` Q4K [2816, 1408, 128] + `ffn_down_exps` Q8_0 [704, 2816,
128] + per-expert `ffn_down_exps.scale` F32 [128]; global layers have **no
`attn_v`** (V reuses the K projection, `src/models/gemma4-common.h:48-50`).

Quant types present: Q4K (194), Q5_0 (33), Q6K (14), Q8_0 (28), F32 (423).
Camelid already eager-decodes all five; what Phase 0.5 added is the
**lazy/file-backed dequant path** (`src/tensor/wire_dequant.rs`: bounded
block-range reads off the GGUF mmap, no whole-tensor f32 materialization,
fail-closed outside the proven format set) plus reference-block parity:
**bit-exact vs llama.cpp's own dequant at the pin** across all five formats,
18 ranges (head + middle of 9 tensors), 155,648 values, zero bit mismatches —
artifact `target/dg-quant-parity-20260611T165546Z.json`, runner
`scripts/dg-quant-parity.sh` + `scripts/dg-dequant-dump.cpp` (the reference
side is ggml's `to_float`, credited).

## 4. Forward pass (pinned source: `src/models/diffusion-gemma.cpp`)

The launch-day "encoder-decoder with cross-attention" description is
**virtual, not structural**. There is one shared Gemma-4 backbone (shared with
the AR `gemma4` arch via `src/models/gemma4-common.h`); no second weight
stack, no dedicated cross-attention projections. A single no-cache
bidirectional forward over `[prompt | canvas]` reproduces the two-pass
(encoder prefill + decoder denoise) result. Exactly three things are
region-aware, split at `P = n_tokens − canvas_length` (header comment,
`diffusion-gemma.cpp:9-16`):

1. **Input embedding**: prompt rows = `embed × √n_embd`; canvas rows =
   `rms_norm_noscale(embed × √n_embd [+ self-conditioning])` (lines 321-384).
2. **Per-layer scalar**: prompt rows × `enc_layer_output_scale`, canvas rows ×
   `layer_output_scale` (lines 473-487). This is the entire "encoder mode".
3. **Attention mask** (lines 21-86): prompt queries causal over prompt only
   (SWA-clipped on sliding layers, never see canvas); canvas queries
   bidirectional over all canvas + prompt ("cross-attention" is just canvas
   queries hitting prompt keys in self-attention; on sliding layers the canvas
   sees only the last `n_swa−1` prompt positions).

Layer body (identical to AR gemma4, `gemma4-common.h`): RMS attn_norm → Q
proj + per-head q-norm + rope (proportional rope via `rope_freqs` on global
layers) → K proj + k-norm + rope; V = v_proj or raw K proj when absent, then
**no-scale RMS v-norm** → attention (no pre-attn scaling,
`f_attention_scale = 1.0`) → post_attention_norm → residual. FFN: dense
shared-expert MLP (gelu, `post_ffw_norm_1`) **plus** 128-expert MoE — router
input is the *unnormed* post-attention residual: `rms_norm_noscale(attn_out)
× 1/√n_embd × ffn_gate_inp.scale → ffn_gate_inp.weight → softmax top-8`
(`gemma4-common.h:92-106`); expert output → `post_ffw_norm_2`; dense + MoE
summed, → `post_ffw_norm`, residual. Final: output_norm → tied head →
logit softcap tanh(z/30)·30.

**Self-conditioning** (lines 321-365; tensors `self_cond_{pre_norm,gate,up,
down}`): previous step's raw canvas logits → `softmax(logits / prev_t)` →
matmul with the (dequantized, transposed) embedding → × √n_embd → gated MLP
(`pre_norm → gelu(gate)·up → down`) → added to the canvas embedding before its
final no-scale RMS norm. Gated to zero on the first step (`sc_use = 0`), which
is the "zero-SC exactness forward" the eval binary exercises by default.

Three execution phases (`llama_diffusion_set_phase`, lines 730-742): UNIFIED
(one `[prompt|canvas]` pass, the parity-simplest), PREFILL (prompt only, writes
an F32 prompt-KV store), DECODE (canvas only, prepends cached prompt K/V).
The cached path matches unified only **to F32 round-off**
(`diffusion-gemma-eval.cpp:112`) — the lane's parity contract therefore pins
**UNIFIED** mode.

## 5. Loader contract (what Camelid must enforce)

From `load_arch_hparams`/`load_arch_tensors` (lines 166-277): requires
`diffusion.canvas_length > 0`; `n_embd_head_k == n_embd_head_v` (both
per-layer-type); `attn_v` optional per layer (absent ⇒ V = K proj); fused
`ffn_gate_up_exps` preferred over split gate/up (this file fuses); per-expert
scale loads from `ffn_down_exps.scale`; `rope_freqs` is a single shared
tensor; head tied when `output.weight` absent; 30 layers ⇒ type 26B-A4B.

## 6. Entropy-Bound sampler (pinned source: `examples/diffusion/diffusion.{h,cpp}`)

The real DiffusionGemma decode loop (`diffusion_generate_entropy_bound`,
`diffusion.cpp:442-711`), auto-enabled when `diffusion.canvas_length > 0`:

- **Canvas init is RANDOM TOKENS, not mask tokens**: uniform draw over the
  whole vocab per position (line 473). The `<mask>` token (id 4) exists in the
  vocab but plays no role in this sampler.
- **Defaults** (in-code; the GGUF carries no `diffusion.eb_*` overrides):
  `max_denoising_steps 48, t_max 0.8 → t_min 0.4 (linear in step), entropy
  bound 0.1, stability_threshold 1, confidence_threshold 0.005, seed 0`.
- **Per step** at temperature `t`: one forward over the working canvas; per
  position compute argmax, entropy of `softmax(raw/t)`, and one multinomial
  sample (inverse-CDF with pre-drawn uniform `u[pos]`); sort positions by
  entropy ascending and **accept** while the cumulative entropy of strictly
  earlier positions ≤ 0.1 (the lowest-entropy position always passes); rebuild
  the working canvas = sampled token where accepted, **fresh random token**
  where not (renoise); the *output* canvas is the **argmax** canvas.
- **Self-conditioning**: this step's raw canvas logits feed the next step at
  `1/prev_t`; off at step 0.
- **Adaptive stop**: argmax canvas unchanged for `stability_threshold` steps
  AND mean entropy < 0.005.
- **Determinism**: one `std::mt19937(seed)`; all randomness pre-drawn
  single-threaded in fixed order (canvas init, then per step `u[pos]` +
  renoise tokens; lines 467-473, 587-591). Host path is seed-reproducible.
  The CUDA `gpu_sample_reduce` path is explicitly **not** bit-exact (reduction
  order; `diffusion.cpp:637-638`) and must be off for parity runs. Residual
  cross-backend FP nondeterminism of the forward itself remains — every parity
  artifact must name the backend and comparison level. This caveat travels
  with every downstream parity claim.

## 7. Multi-canvas (block-autoregressive) loop (`diffusion-cli.cpp:413-485`)

Lives in the CLI, not the library: per block, run the EB denoiser over
`[prefix | 256-token canvas]`; trim the finished **argmax** canvas at the
first end-of-generation token or repetition loop (a token recurring at stride
1-2 for ≥ 6 steps); append the trimmed canvas to the prefix; repeat until an
end token, the block budget (`-n` → ⌈n/256⌉ blocks), or the ubatch limit.
The published "re-encoded and appended to the KV cache" is, in the reference,
simply the next block's forward over the grown prefix (re-prefill; in cached
mode a new PREFILL of the full prefix). The RNG **re-seeds with the same seed
at the start of every block**: the per-block call is
`diffusion_generate_entropy_bound`, which constructs `std::mt19937 rng(params.
seed)` locally (`diffusion.cpp:467`); the `run_turn` loop
(`diffusion-cli.cpp:444`) calls it once per block, so each block's canvas init +
per-step `u`/renoise draws restart from seed (NOT one stream continued across
blocks — an earlier reading here said "continues", which the pinned source
disproves; the `mc_generate` impl and the `dg-mc-loop` oracle both re-seed).
Phase 5's gate is parity over ≥ 2 blocks of exactly this loop.

## 8. Phase 2 status — encoder checkpoint parity: PASSED, BIT-EXACT

`src/diffusion_gemma.rs` implements the encoder prefill and, under the
maintainer's option A, was driven to **bit-exact parity with the pinned
reference at ZERO tolerance**: sealed bundle
`target/dg-encoder-parity-20260611T223204Z/` — 242/242 checkpoints
(embeddings, per-layer K/V, attention residuals, router logits, both FFN
branches, every expert-chain stage, layer outputs, final norm) with
max-abs 0.0 and 510/510 expert selections exact on the `hello` chat prompt
(CPU backend, no repack, flash attention off).

Getting there required reproducing the exact float semantics the pinned
build executes on this machine, discovered by checkpoint-ladder forensics
and, twice, by disassembling the pinned dylib: the reference-order kernel
set in `src/diffusion_gemma/refmath.rs` (double-sum rms_norm, ggml_v_expf
chunked softmax, vec_dot trees, ARM nrc=1 quant dots, nearest-even Q8_0
activation quantize, iterative-theta rope with the decoded rotation fusion),
Apple's `__sincosf_stret` for the rope trig, the table-contraction form of
GELU, and two Accelerate diversions that are NOT IEEE-equivalent
(`vDSP_sve` for weight sums, `vDSP_vdiv` — reciprocal-based — for the MoE
weight normalization). An earlier honest stop and its analysis are
preserved in `target/dg-encoder-parity-20260611T194014Z/FAILURE_REPORT.md`
and the loop log. The pinned reference itself is bit-deterministic across
thread counts (control in that bundle).

## 8b. Phase 3 status — single denoise step parity: PASSED, BIT-EXACT

`src/diffusion_gemma.rs::unified_forward` implements the Phase 3 decode
surface — one zero-self-conditioning bidirectional forward over
`[prompt | canvas]` (canvas embeddings rms_norm'd after the embed scale,
region mask: prompt causal/SWA over prompt only, canvas bidirectional with
sliding layers reaching the last `n_swa−1` prompt positions, decoder
per-layer scalars on canvas rows, tied Q6_K lm_head + tanh softcapping at
30.0) — and `eb_step` + `refrng` implement one Entropy-Bound sampler step
with the reference's exact host RNG. **Bit-exact at ZERO tolerance**:
sealed bundle `target/dg-decode-parity-20260612T034000Z/` — all
**67,108,864 canvas logits** (256 × 262144), every per-layer trace
checkpoint, the full mt19937/libc++ RNG streams (canvas init + u +
renoise), and every EB step-0 output (argmax canvas, entropies, multinomial
draws, MI-bound accepted set of 24, renoised next canvas) identical to the
pinned reference.

Two reference-identity facts this phase established:

1. **The default macOS build's big-batch matmuls are NOT the CPU kernels.**
   ggml registers a BLAS (Accelerate) backend whose device claims every
   contiguous `mul_mat` with `ne0/ne1/ne10 >= 32` (`ggml-blas.cpp`,
   `min_batch = 32`) and computes it as dequantize-src0-to-f32 +
   `cblas_sgemm` — no activation quantization, closed-source blocking,
   ~1e-2-relative different from the vec_dot path on Q4_K rows. The Phase 2
   prompt (17 rows) sat under the threshold; the unified forward (273 rows)
   crosses it for all dense projections, KQV, the router, and the lm_head.
   GPU op-offload behaves the same way (Metal takes `n_tokens >= 32` graphs
   even at `ngl=0`). The lane's parity contract therefore NAMES the
   kernel configuration: **CPU-pure build of the same pinned commit**
   (`build-cpu`: `GGML_BLAS=OFF`, `GGML_METAL=OFF`, `GGML_ACCELERATE=ON` so
   the CPU backend's vDSP diversions stay), with an empty `mparams.devices`
   list in the dumper. Phase 2's sealed result is unaffected (verified:
   camelid's unified prompt rows are byte-identical to the Phase 2 ref).
2. **Sampler host math contracts.** The reference EB worker's
   `expf(row[v]*temp_inv − m)` argument and `H -= p*logf(p)` update both
   contract to single-rounding fma forms under clang's default
   `-ffp-contract=on` (fmadd/fmsub in the oracle disassembly); the linear
   temperature schedule `t_min + (t_max−t_min)*ratio` contracts too. The
   distributions are libc++-specific ports (`refrng.rs`): for the
   full-range mt19937, `uniform_int_distribution` collapses to one masked
   draw with rejection, and `uniform_real_distribution<float>(0,1)` is
   `(float)draw / 2^32` — which CAN return exactly 1.0f, a quirk the
   sampler inherits.

## 8c. Phase 4 status — full EB denoise loop parity: PASSED, BIT-EXACT

`eb_generate` runs the complete Entropy-Bound denoise loop (the reference's
default unified no-KV-cache path) with live self-conditioning:
`softmax(prev raw logits × prev 1/t)` through the f16-transposed soft
embedding (`dg_ensure_sc_embT` semantics) and the gated MLP into the canvas
embedding, per-step pre-drawn randomness, MI-bound acceptance, renoise, and
the adaptive stop. **Bit-exact at the FULL default parameters** (S=48,
seed 0): sealed bundle `target/dg-eb-loop-parity-20260612T063936Z/` — the
reference's adaptive stop fired at step 7 (executed=8, finished=true) and
camelid matched the trajectory step for step: every discrete output equal,
entropies bit-exact on all steps, 268,435,456 dumped canvas-logits values
bit-identical.

Phase 4 kernel facts:

1. **fp16 is a first-class kernel precision in the SC path.** The
   soft-embedding matmul is `ggml_vec_dot_f16` (NEON: 4 float16x8
   accumulators, `vfmaq_f16` fused single-rounding lanes, an f16 reduce
   tree, f32 lane conversion, double total). Camelid ports it as an exact
   f64 emulation (one round-to-nearest-even to f16 per op; double rounding
   is innocuous since 53 ≥ 2·11+2) plus an aarch64 inline-asm `fmla v.8h`
   fast path — both pinned bit-for-bit against hardware ground truth
   (`scripts/dg-f16-dump.cpp`).
2. **The even-ne11 nrc=2 mmla path is bit-identical to nrc=1 for Q4_K.**
   The SC gate/up matmuls (ne11=256) are the first surface where ggml's
   i8mm 2×2 tile kernel engages; its integer sums and its two fused float
   ops per superblock are the same values in the same order as the nrc=1
   body, so no separate port is needed (confirmed by the passing gates).
3. **Step 0 self-conditioning is not a no-op at the bit level.** The EB
   loop keeps the SC subgraph in the graph with `use_sc = 0.0`; `sig × 0.0`
   adds a signed zero per element to the canvas embedding. Camelid runs the
   full chain on step 0 to reproduce the ±0.0 pattern.

## 8d. Phase 5 status — multi-canvas parity: PASSED, BIT-EXACT

`DgEncoderRuntime::mc_generate` runs the reference's `run_turn` block-
autoregressive loop (§7): per block, EB denoise over `[prefix | 256-token
canvas]`, trim at the first EOG token or a stride-1/2 repetition (≥ 6 reps),
commit the trimmed canvas to the prefix, re-seed `mt19937(seed)` and repeat
until an end token, the block budget, or the ubatch guard. **Bit-exact at
zero tolerance over a 2-block run** (prompt: the 41-token Andes-caravan
message; UNIFIED phase, EB in-code defaults, seed 0): sealed bundle
`target/dg-mc-loop-parity-20260616T011145Z/` (commit `999f6011`; oracle
`scripts/dg-mc-loop.cpp` against the CPU-pure pin build, `GGML_BLAS=OFF
GGML_METAL=OFF GGML_ACCELERATE=ON`, empty `mparams.devices`, wall ~7.26 h;
Apple Silicon host).

- block 0: prefix 41, 22 EB steps, cut 256 (full canvas) → committed; PASS.
- block 1: prefix 297, 26 EB steps, cut 256 → committed; PASS.
- EOG set (3 ids, derived from the GGUF and equal to the reference vocab's
  set, fail-closed), per-block trim cut, commit-or-stop chaining, and the
  ubatch budget guard all match; `executed_blocks` 2, stop reason `ubatch`,
  512-token response byte-identical to the reference. Discrete outputs
  exact; entropies bit-exact (atol=0/rtol=0).
- Gate test: `tests/dg_mc_loop_parity.rs` (env-gated on `CAMELID_DG_GGUF`,
  `CAMELID_DG_MC_REF`, `CAMELID_DG_MC_IDS`); forensics
  `dg_mc_block0_diag`, `dg_block1_logit_ladder`, `dg_block1_step3_probe`.

Phase 5 facts:

1. **Block 0 is Phase 4 under the per-block re-seed** — it re-passed
   unchanged; everything new was at block 1, where the prompt is
   `original-prompt + committed-block-0-canvas` (re-prefill over the grown
   prefix).
2. **The block-1 divergence was a sort-tie order, not region math.** A
   router-logit tie at block-1 step 3 ordered differently under the
   portable sort than under the reference build's libc++ introsort,
   swapping two MoE accumulation slots and seeding compounding FP drift.
   Sealed fix: `src/dg_argsort.cpp` (+ `build.rs`) binds the platform
   libc++ `std::sort` for the expert argsort and the eb_step entropy sort.
   NOTE: the later Windows port (`c847894b`) removed this C++ shim in
   favor of a pure-Rust sort that breaks exact ties by lower index — the
   seal attaches to the shimmed revision; re-validate on ties (see the
   README caveat).
3. **RNG re-seeds per block with the same seed** (§7): each block's canvas
   init and per-step `u`/renoise draws restart from `mt19937(seed)`; the
   SC buffer is likewise fresh per block with `use_sc = 0.0` at step 0.

## 8e. Windows x86_64 host + GPU (VRAM) evidence: PASSED (this-host contract)

First model runs of the lane on a Windows host (2026-07-03; every prior
run receipt was Apple Silicon). Sealed bundle
`target/dg-windows-run-20260703T164114Z/` (see its `SUMMARY.md`): RTX 3060
Laptop 6 GiB, 16 GiB RAM, the pinned 16,806,810,336-byte GGUF, canonical
41-token Andes prompt, seed 0.

- **CPU-pure determinism: PASS** (byte-identical run pair;
  `CAMELID_DG_CUDA=0`, `CUDA_VISIBLE_DEVICES=-1`).
- **GPU SC stage: 2.7× wall vs CPU** (200 s vs 551 s for load + 2 denoise
  steps), deterministic (byte-identical pair), VRAM peak 1675 MiB. The SC
  soft-embedding matmul accumulates in f32 on GPU and remains the one
  non-bit-exact stage (documented; `CAMELID_DG_CUDA_SC=0` opts it out).
- **MoE expert pool (VRAM-resident experts): bit-exact, proven in vivo.**
  `ffn_gate_up_exps` (Q4_K) / `ffn_down_exps` (Q8_0) row-range GEMVs run on
  resident device copies via kernels that mirror `q4_k_dot_scalar` /
  `q0_pair_dot` exactly (unit gate `expert_gemv_gpu_bit_identical_to_cpu`);
  an experts-on-GPU run (`CAMELID_DG_CUDA_SC=0`, 2.85 GiB of experts
  resident ≈ 5 layers) produced output **byte-identical to the CPU-pure
  leg**. Budget-gated (`CAMELID_DG_EXPERT_VRAM_MB`, default free-VRAM minus
  a 2.2 GiB reserve), greedy whole-tensor residency, per-call CPU fallback.
  Speed-neutral in this config; the win is capacity — resident expert bytes
  leave the CPU mmap working set (model 16.8 GB > 15.7 GB usable RAM).
  Full-GPU (SC + experts) VRAM peak: 4587 MiB of 6144.

- **Multi-canvas end-to-end on Windows (GPU SC + expert pool):** a
  natural-stop 2-block run completed in 7074 s — block 0 (prefix 41,
  23 EB steps, cut 256 → committed), block 1 (prefix **297**, equal to the
  Apple-sealed Phase 5 grown prefix, 25 steps, cut 256), stop `blocks`,
  512 tokens, coherent output. The adaptive-stop step counts (23/25 vs
  the sealed 22/26) differ as expected under the non-bit-exact SC stage
  and the off-macOS sincos/tie-order notes below; the loop structure and
  the committed-prefix chaining match.

Scope of the claim: **this-host contract** — determinism, and GPU stages
bit-exact against this host's CPU oracle. NOT a bitwise claim against the
Apple-sealed §8–8d artifacts: off macOS, `libm_sincosf` is Rust
`f32::sin/cos` (not `__sincosf_stret`) and the pure-Rust expert argsort
breaks exact router-logit ties by lower index (the sealed runs bound the
libc++ shim; see §8d note).

## 9. Phase 0 gate status

- `tensor-inventory.json` + `metadata.json` exist; **zero** unclassified
  tensors; vision/video disposition: not present in the file.
- Reference runtime pinned, built, invocation + determinism knobs recorded
  (`llamacpp-pin.json`).
- Quant-coverage delta enumerated (`quant-coverage.json`) ⇒ Phase 0.5 scope.
- This document.

Next phase: **0.5 — quantization coverage** (lazy K-quant/Q5_0 load +
dequant parity vs llama.cpp on same-file blocks).
