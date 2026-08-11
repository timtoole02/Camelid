# Gemma 4 26B A4B QAT Q4_0 — row recon (planned lane, no claims)

> [!NOTE]
> This document is a design or recon note, not the public support ledger. For
> current support truth and release status, use
> [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../reference/STATUS.md).
> The 26B A4B row remains **blocked / fail-closed** until every gap below has
> committed evidence. Nothing here is a support claim.

## Why this row re-opened

The committed 26B blocker was recorded against the Q8_0 row (26.9 GB — exceeds
the 2×16 GB distributed envelope). The official QAT release changes the memory
math: `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` → `gemma-4-26B_q4_0-it.gguf`
is **14,439,361,440 bytes (13.4 GiB)** — ~6.7 GiB per node on the proven
two-Mac distributed layer-sharding lane. Memory stops being the blocker; the
engine gaps below become the real work. Per exact-row doctrine this is a NEW
row with a fresh evidence chain; it inherits nothing from any Q8_0 row.

Local copies (T7): `/Volumes/Untitled/models/gemma-4-26B_q4_0-it.gguf` and the
dense de-risk row `/Volumes/Untitled/models/gemma-4-E4B_q4_0-it.gguf`
(5,154,939,136 bytes, from `google/gemma-4-E4B-it-qat-q4_0-gguf`).

## GGUF facts (read with `camelid inspect`, v3, 658 tensors, 46 metadata keys)

Geometry (all parsed by the existing `gemma4` metadata path):

| Field | Value |
| --- | --- |
| Layers | 30, sliding pattern 5:1 (`sliding_window_pattern` bool array), window **1024** |
| Hidden | 2816, heads 16, per-layer `head_count_kv` array: 8 sliding / 2 global |
| head_dim | 256 sliding / 512 global (`key_length(_swa)`, `value_length(_swa)`), rope dims match |
| RoPE | dual-θ 1e6 / 1e4 (same as E-series); `rope_freqs.weight` [256] present |
| PLE | **none** (`embedding_length_per_layer_input = 0`) |
| Shared KV | **none** (`shared_kv_layers = 0`) |
| Softcap | final_logit_softcapping 30 |
| MoE | `expert_count` **128**, `expert_used_count` **8**, `expert_feed_forward_length` **704** |
| Dense FFN | `feed_forward_length` **2112** (per-layer dense branch alongside the experts) |
| Context | 262144; vocab 262144 (`tokenizer.ggml.model = gemma4` SPM) |
| Tokenizer deltas vs E-series | `eos_token_id = 1`, `add_bos_token = False` — VERIFY at runtime; E-series rows use EOS/EOT 106 and add BOS |
| size_label | 128x2.6B |

Per-layer tensor map (layer 0; same shape every layer):

- Attention: `attn_q [2816,4096]`, `attn_k [2816,2048]`, `attn_v [2816,2048]`,
  `attn_output [4096,2816]`, QK norms [256] — standard gemma4, V present on
  all layers (no 12B-style V-less rows in this file).
- Dense FFN branch: `ffn_gate/ffn_up [2816,2112]`, `ffn_down [2112,2816]`.
- MoE branch: router `ffn_gate_inp [2816,128]` (**F32**) + `ffn_gate_inp.scale`
  [2816] (F32); experts `ffn_gate_up_exps [2816,1408,128]` (fused gate+up,
  3D, 128 experts) and `ffn_down_exps [704,2816,128]` + per-expert
  `ffn_down_exps.scale [128]` (F32).
- Norms: `attn_norm`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm`,
  **plus `pre_ffw_norm_2` / `post_ffw_norm_1` / `post_ffw_norm_2`** — three
  extra norms consistent with a dual-FFN (dense + routed) sub-block whose
  exact composition order must come from the reference implementation, not
  guesses.
- `layer_output_scale [1]` per layer (12B-style unconditional output scale).
- Head: `output_norm`, tied `token_embd` (no `output.weight`).

Quantization split (histogram): **265 × Q4_0** (all attention/dense-FFN/expert
matrices), **1 × Q6_K** (`token_embd`, 605.6 MB), **392 × F32** (norms, router,
scales, rope factors).

## Engine gaps (the honest work list)

1. **Q4_0 lane** — Camelid is Q8_0-only end-to-end (34-byte wire blocks, Q8×Q8
   sdot CPU path, GPU wire-Q8 GEMV). Q4_0 needs: 18-byte block wire structs,
   CPU dot (llama.cpp pattern: Q8-quantized activations × dequantized nibbles),
   a GPU GEMV variant, loader gates for mixed per-tensor types. De-risk on the
   dense E4B QAT row first — it isolates "Q4_0 kernels" from "MoE" so the two
   unknowns are proven sequentially.
2. **Q6_K head** — the tied output projection over the 262K vocab runs in
   Q6_K (K-quant superblocks, 210 B / 256 weights). Bit-parity against the
   reference requires mirroring its q6_K dot exactly; dequant-to-Q8 at load
   would diverge numerically. Third kernel family; scope it as its own step.
3. **MoE forward** — router → top-8 of 128 → fused gate+up expert GEMV →
   GeGLU → down expert GEMV → combine, PLUS the dense FFN branch and the
   `pre_ffw_norm_2`/`post_ffw_norm_1`/`post_ffw_norm_2` composition. The
   `.scale` companion tensors' application points are undocumented; derive the
   exact dataflow from the reference source before writing any kernel, the
   same way 12B Unified semantics were derived.
4. **Comparator** — verify the pinned llama.cpp 5d56eff actually runs this
   row; if not, this row pins its own newer comparator build (the
   comparator-per-row pattern 12B already established), captured with the
   recorded plain-path flags.
5. **Distributed split** — no shared KV, so any split works; ~halving 30
   layers gives ~6.7 GiB/node. Single-node 16 GB stays memory-bound (12B at
   12.7 GB already was). The MTP `gemma-4-26B-A4B-it-assistant` row stays
   fail-closed (unchanged).

## Sequencing (after the current E2B/E4B promotion + 12B serve lane closes)

1. Reference-source recon of the dual-FFN/MoE block + `.scale` semantics.
2. Q4_0 kernels proven bit-exact on `gemma-4-E4B_q4_0-it.gguf` (dense, known
   architecture, CPU then GPU).
3. Q6_K head kernel, proven on the same dense row.
4. MoE CPU forward vs oracle at single positions, then greedy parity packs.
5. Two-Mac distributed run; GPU residency decision comes after CPU parity.

Performance note (motivation, not a claim): ~4B active parameters/token means
decode reads ~2–3 GB/token instead of the full file — on the ~120 GB/s wall
this row's ceiling is well above the 12B dense pair's 6.2–6.75 tok/s.

## Derived MoE forward contract (from llama.cpp `gemma4-iswa.cpp` + `build_moe_ffn`)

Every 26B A4B layer's FFN is TWO parallel branches off the post-attention
residual `attn_out`, summed, then a final `post_ffw_norm`, then residual.
(Single-token decode; `n_embd = 2816`, `n_expert = 128`, `n_expert_used = 8`,
expert ffn `n_ff = 704`, dense ffn `2112`.)

**Branch A — dense "shared expert" MLP** (the existing gemma4 dense FFN op):
```
cur_mlp = rms_norm(attn_out, ffn_norm)
cur_mlp = down( gelu_tanh(gate(cur_mlp)) * up(cur_mlp) )    # parallel GeGLU, width 2112
cur_mlp = rms_norm(cur_mlp, ffn_post_norm_1)
```

**Branch B — sparse 128-expert MoE**:
```
cur_moe = rms_norm(attn_out, ffn_pre_norm_2)

# Router runs on attn_out, NOT cur_moe, with its OWN weightless norm:
r   = rms_norm(attn_out)                 # weightless, over n_embd
r   = r * (1/sqrt(n_embd))               # 1/sqrt(2816)
r   = r ⊙ ffn_gate_inp_s                 # elementwise [2816] (the .scale companion)
logits = ffn_gate_inp @ r                # [128]   (F32 router matrix)

probs   = softmax(logits)                # over all 128
top8    = argsort_top_k(probs, 8)        # the 8 selected expert ids
w       = probs[top8]                     # [8]
w       = w / clamp(sum(w), 6.1e-5, inf) # norm_w=true; w_scale=1.0 (no extra scale)

for each selected expert e in top8:
    gate_up = gate_up_exps[e] @ cur_moe   # fused [1408] = [gate(704) ‖ up(704)]
    g, u    = gate_up[:704], gate_up[704:]
    h       = gelu_tanh(g) * u            # geglu_split, GELU(tanh approx)
    y_e     = down_exps[e] @ h            # [2816]
    y_e     = y_e * down_exps_s[e]        # per-expert down .scale (scalar [128])
    y_e     = y_e * w[e]                  # routing weight
cur_moe = sum_e y_e                       # [2816]
cur_moe = rms_norm(cur_moe, ffn_post_norm_2)
```

**Combine + finish**:
```
cur = cur_mlp + cur_moe
cur = rms_norm(cur, ffn_post_norm)       # the same post_ffw_norm the dense rows use
cur = cur + attn_out                      # residual
# then per-layer-embedding (PLE) if present — 26B has no PLE
```

Key gotchas locked here:
- Router input is `attn_out` (pre-branch residual), re-normed weightlessly and
  scaled by `1/sqrt(n_embd)` BEFORE the elementwise `ffn_gate_inp_s` — easy to
  wire to the wrong tensor or skip the sqrt.
- `gate_up_exps` is FUSED gate‖up in one [n_embd, 2*n_ff, n_expert] tensor;
  gate = first n_ff rows, up = second n_ff. There is NO `.scale` on it.
- The only per-expert `.scale` is `ffn_down_exps_s` ([128]); applied to the
  down output before the routing-weight multiply.
- GELU is the tanh approximation (`ggml_geglu` → same `gelu_tanh` the dense
  gemma4 FFN already uses and is bit-parity for).
- top-8 weights are sum-normalized (clamped), then NOT extra-scaled (w_scale=1).
- Expert e's matrix is the e-th slice along the last GGUF dim; byte offset =
  e * (per-expert row bytes) into the Q4_0 wire tensor.

## Implementation status (WIP — branch `feature/gemma4-26b-moe-wip`, NOT merged)

The MoE forward is IMPLEMENTED and runs end-to-end, but does NOT yet reproduce
the reference — it is not merged and 26B stays **blocked** in the public ledger.

Done and verified correct against the reference / E4B QAT:
- Binding: `Gemma4MoeLayerTensors` binds router + fused experts + down experts +
  the 3 extra norms + both `.scale` companions (GGUF names `pre_ffw_norm_2`,
  `post_ffw_norm_1`, `post_ffw_norm_2`). The fail-closed guard now only rejects
  the unmodeled split-expert (`ffn_gate_exps`) layout.
- Q4_0/Q6_K wire kernels (shipped earlier) + `matvec_q_rows` for per-expert
  3D-tensor row slices (layout `(e*rows_per_expert + o)` verified against the
  GGUF dim order `[in, out, n_expert]`).
- CPU forward: the two-branch block exactly as the contract above.

Confirmed NOT the bug (each checked against the reference source / runtime):
- norm tensor names & roles; the two-branch composition; shared `post_ffw_norm`.
- router: input = `attn_out`, weightless rms_norm, `×1/√n_embd`, `⊙ gate_inp_scale`
  (uniform 31.25), softmax-over-all, top-8, sum-normalized weights (w_scale=1).
  Router selection is input-dependent (different prompts → different experts).
- fused gate‖up split (gate first), gelu = tanh approx (same `ggml_geglu_split`
  the dense `build_ffn` uses and E4B QAT is bit-parity for).
- `.scale` companions: only `ffn_down_exps.scale` (≈1.0) and `ffn_gate_inp.scale`
  (31.25); applied at the reference's points. No other per-weight scales exist
  (E4B QAT has zero `.scale` tensors — that is why it already works).
- attention scale = 1.0 (gemma4); per-layer sliding/global schedule and
  `head_count_kv` arrays parse correctly (binding shape-validation passes).
- prompt tokenization: `add_bos=False` respected → `[818,5279,529,7001,563]`,
  identical to the reference.

### ROOT-CAUSED AND FIXED (the MoE math was correct all along)

The divergence was NOT in the MoE forward — it was the prompt tokenization.
The eval-callback trace showed the reference feeds **6** tokens
(`[2, 818, 5279, 529, 7001, 563]`, BOS-led) while camelid fed **5** (no BOS).
This 26B QAT export ships an incorrect `tokenizer.ggml.add_bos_token = false`;
llama.cpp force-overrides it to true for all gemma4 (PR #21500 workaround,
`LLAMA_VOCAB_PRE_TYPE_GEMMA4`). Camelid now applies the same override in
`Tokenizer::from_gguf` (force `add_bos=true` when `tokenizer.ggml.model ==
"gemma4"`; E-series/12B already ship true so it is a no-op for them).

With the BOS restored, the 26B A4B MoE forward is **token-identical to the
reference**: `The capital of France is` → ` Paris.`
(`[9079, 236761, 107, 100, 236800, 236786]`, exact match). The entire derived
MoE contract above is therefore validated end-to-end on the real row.

Tooling note: the eval-callback example builds standalone against the existing
reference dylibs — `clang++ -std=c++17 -I common -I include -I ggml/include
examples/eval-callback/eval-callback.cpp -Lbuild/bin -lllama -lllama-common
-lggml -lggml-base -lggml-cpu` — and prints every named tensor per op, which is
how the 6-vs-5 token mismatch was caught in one run.

## Full basic_v1 distributed parity pack — VALIDATED (two-Mac)

Ran the 5-prompt basic_v1 pack distributed across both 16GB M4 Macs (master
0..15 local, worker 15..30 + head on mini2; model staged over Thunderbolt,
full-file sha256 verified identical; wire over the ssh tunnel) and compared to
the pinned reference oracle (`qa/gemma4/oracle/gemma-4-26B_q4_0-it.basic_v1.json`):

| Prompt | Result |
| --- | --- |
| count-primes | FULL (24 tok token-identical to reference) |
| translate-de | FULL (16 tok token-identical) |
| capital-france | prefix to idx 6, then knife-edge near-tie (top-2 gap 0.138) |
| haiku-sea | prefix to idx 11, then near-tie (gap 0.203) |
| rust-fn | prefix to idx 15, then near-tie (gap 0.419) |

All three divergences are probe-verified knife-edge near-ties: the reference's
greedy token is camelid's immediate #2 within ~0.14–0.42 logits, in the
low-information continuation regions (numbers/dates/code) where greedy ties
flip. `count-primes`/`translate-de` matching full-budget proves the A4B MoE
forward is complete and correct — nothing is missing (contrast the E-series,
whose "frontier" was a real missing rope-factors feature). Distributed output
equals single-node (f32 wire); the probe (single-node) top-1 equals the
distributed token in every case.

This is the established promotable standard (same shape as the E4B QAT pack:
full-budget where deterministic + probe-verified frontiers on near-ties).
Evidence bundle: `qa/evidence-bundles/gemma4-26b-it-q4-0-qat-distributed-parity-*`.
Comparator: llama.cpp 5d56eff `--no-repack -fa off -ctk f32 -ctv f32 -ub 1`
(plain-f32 path; 26B is V-full so `-ub 1` is sound, unlike the 12B V-less case).

## PROMOTED — supported exact-row smoke (two-Mac distributed lane)

The 26B A4B QAT row is promoted to `supported_exact_row_smoke` scoped to the
two-Mac distributed serve lane, mirroring the 12B treatment. Evidence:
- Distributed parity pack: `qa/evidence-bundles/gemma4-26b-it-q4-0-qat-distributed-parity-20260611T084039Z-head-b117d40cb7c3` (2/5 full-budget + 3/5 probe-verified frontiers).
- Distributed serve/WebUI promotion smoke: `qa/evidence-bundles/gemma4-26b-a4b-it-q4-0-distributed-serve-20260611T092520Z-head-6482254fca12` (model-promotion-smoke-bundle passed=true: load, /v1/completions, /v1/chat/completions, capabilities expectations, generation timings, frontend WebUI closure with streaming).

Capabilities row `gemma4_26b_a4b_it_q4_0` (status `supported_exact_row_smoke`,
scope `exact_row_distributed_serve_smoke_only`), curated catalog + frontend
`supportedModels` row, and the supported-row allowlist guard are updated
together; COMPATIBILITY/STATUS/README carry the synchronized exact-row wording.
The Q8_0 26B A4B (26.9 GB), 31B, and `gemma4-assistant` MTP rows stay blocked.
