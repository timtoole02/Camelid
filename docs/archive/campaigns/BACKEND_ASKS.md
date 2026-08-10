# BACKEND_ASKS.md

Open requests for reference data / undecided tolerances surfaced while building the
runnable lane (`RUNNABLE_LANE_SPEC.md`). Each entry: what's needed, why, blocking phase.

## RA-1 — HF reference harness (transformers + tokenizers) — **RESOLVED (Phases 3 & 5)**
- **What:** A pinned HF `transformers` + HF `tokenizers` reference harness producing
  frozen fixtures (greedy logits/token sequences; string↔id maps) per (arch, quant, tokenizer).
- **Resolution:** Anchored to **HF** (spec-literal). Both halves now stood up:
  - Tokenizer (Phase 3): `scripts/gen-tokenizer-fixtures.py` (HF `tokenizers`,
    `tokenizers==0.23.1`) → `tests/fixtures/tokenizer_hf/`; `tests/runnable_tokenizer.rs`.
  - Transformers (Phase 5): `scripts/gen-hf-parity-fixtures.py` loads the **same GGUF**
    into HF (`transformers==5.12.1`, `torch==2.12.0+cpu`) — it dequantizes Q8_0 to f32
    (= camelid's bit-exact dequant) and un-permutes Q/K — runs greedy, records token
    sequences + first-step logits → `tests/fixtures/hf_parity/tinyllama.json`;
    `tests/runnable_parity.rs` checks camelid. **Note:** transformers 5.12.1 mis-detects
    gguf's version as 'N/A'; the script monkeypatches `modeling_gguf_pytorch_utils.is_gguf_available`.

## RA-5 — SPM leading-whitespace divergence from HF — **RESOLVED (Phase 3, fixed)**
- **Was:** camelid's SPM `normalize_spm_text` suppressed the dummy `▁` prefix when text
  started with whitespace; HF's Metaspace always prepends it → 4 leading/pure-whitespace
  cases diverged.
- **Fix:** removed the `!text.starts_with(char::is_whitespace)` guard
  (src/tokenizer/mod.rs:575) so the dummy `▁` is always prepended when `add_space_prefix`
  is set (matching HF). SPM encode is now **30/30 HF-exact**; BPE remains 30/30.
- **Regression check:** no supported-lane regression — lib unit tests 475/475, existing
  `tests/tokenizer.rs` 25/25, `tests/dg_tokenizer_parity.rs` (llama.cpp anchor) green.
  The change only affects plain-text SPM encode with `parse_special=false` on
  leading-whitespace input; chat tokenization uses `parse_special=true` + control tokens
  and is unaffected. DG/gemma sets `add_space_prefix=0`, so it short-circuits.
- **Decode note (deliberate, not a defect):** camelid's `decode` is STATELESS so it can be
  called per-token during streaming (`api/mod.rs:6880/6901`, `main.rs:1659/2631`). It
  therefore retains the single dummy-prefix space rather than stripping it like HF's
  stateful Metaspace decoder. Consumers strip one leading space per the `add_space_prefix`
  convention to recover exact round-trip — `tests/runnable_tokenizer.rs` does this and
  shows 0/30 round-trip instability for both families.

## RA-2 — ggml dequant reference fixtures — **RESOLVED (Phase 2)**
- **What:** Checked-in block-level reference dumps under `tests/fixtures/dequant/` produced
  by Python `gguf`/`llama-cpp`, for F32, F16, Q8_0, Q4_0, Q4_K_M, Q5_K_M, Q6_K.
- **Resolution:** `scripts/gen-dequant-fixtures.py` emits fixtures under
  `tests/fixtures/dequant/` using the `gguf` package (gguf==0.19.0, the numpy port of
  ggml's dequant) as the reference; `tests/runnable_dequant.rs` bit-checks the runnable
  decoder against them. All 7 formats are **bit-exact** (max_abs=0, max_ulp=0).
  - F32/F16 via numpy; Q8_0/Q4_0 via `gguf.quants.quantize`; Q4_K/Q5_K/Q6_K via
    **synthetic structurally-valid blocks** (random integer fields + sanitized f16
    super-scales). A dequant bit-exactness test is independent of byte provenance.
  - **Why synthetic for K-quants:** the only on-disk K-quant model
    (`diffusiongemma-…-Q4_K_M.gguf`, ~16 GB) cannot be memmapped on this box —
    `GGUFReader` does a full-file `np.memmap` and Windows fails with
    `WinError 1455 (paging file too small)`. Real-model extraction was abandoned for
    the (equivalent, more robust) synthetic path. If real-model anchoring is later
    wanted, it needs a small K-quant GGUF or a streaming (non-memmap) reader.

## RA-3 — tolerances
- **(i) Dequant tolerance — RESOLVED (Phase 2):** every covered format (incl. F16 and all
  K-quants) is **bit-exact** vs ggml reference — `max_abs_diff == 0`, `max_ulp == 0`. No
  tolerance needed; the test asserts bit-exactness for F32/F16/Q8_0/Q4_0 and `max_abs == 0`
  for the K-quants.
- **(ii) Logit max-abs-diff threshold — RESOLVED (Phase 5):** the hard gate is greedy
  token-sequence exact-match (passed 64/64 tokens, 4 prompts, TinyLlama). Observed logit
  max-abs-diff vs HF (same dequantized weights) = **4.673e-5** — pure f32 op-order
  rounding. No numeric tolerance is gated; the diff is reported as evidence in the parity
  artifact (`qa/runnable/tinyllama-parity.json`).

## RA-4 — covered-set vs. code allowlist mismatch — **RESOLVED (Phase 1)**
- **What:** Confirm the runnable v1 covered-set is authoritative:
  spec archs `{llama, qwen2, qwen3, gemma2, gemma3, phi3}` vs. `src/model.rs:52-54`
  `{llama, mistral, qwen2, qwen3, smollm3, gemma3, gemma4, phi3, lfm2}` (note: spec has
  **gemma2**, code does not; code has mistral/smollm3/gemma4/lfm2, spec does not).
- **Resolution:** The **spec's covered-set is authoritative for the runnable lane**
  (the spec declares it so). The admission gate (`src/runnable/admit.rs`,
  `COVERED_ARCHITECTURES`) keys off `{llama, qwen2, qwen3, gemma2, gemma3, phi3}`,
  intentionally independent of `model.rs`'s optimized-lane allowlist. Revisit only if
  a model the supported lane handles (e.g. mistral) must also run runnable.

## RA-5 — `smollm3` is a NoPE architecture — **IMPLEMENTED (CPU), EVIDENCE OWED**
- **What:** `smollm3` sat in the optimized-lane allowlist (`src/model.rs`
  `is_implemented_architecture` / `from_gguf`) while the engine applied RoPE to every
  layer. llama.cpp hardcodes `n_no_rope_layer_step = 4` (`src/models/smollm3.cpp:5` —
  there is **no GGUF key** for it, so it cannot be read from the file) and its graph at
  `:69` skips `ggml_rope_ext` on **both Q and K** whenever `(il + 1) % 4 == 0`. Because
  SmolLM3 is llama-shaped it bound cleanly and then mis-roped: **silently wrong output
  under a claimed-implemented architecture**, not a clean refusal. For a 36-layer
  SmolLM3-3B the affected layers are 3, 7, 11, 15, 19, 23, 27, 31, 35 — 9 of 36,
  including the final layer.
- **Fixed:** `LlamaModelConfig::no_rope_layer_step` + `layer_uses_rope`, applied at both
  CPU RoPE call sites (single-token decode and batched prefill). The resident GPU
  engines **fail closed** to CPU for NoPE models (`resident_decode_eligible`): they
  build one cos/sin table per forward and rope every layer unconditionally, so they
  cannot express the skip and would diverge from the CPU path.
- **Audit:** the other nine admitted architectures were each checked against their
  llama.cpp graph builder and rope unconditionally (`models/llama.cpp:146,152`,
  `qwen2.cpp:86,92`, `qwen3.cpp:91,100`, `phi3.cpp:107,113`, `mistral3.cpp:137,143`).
  `gemma3`/`gemma4`/`qwen35` carry per-layer rope **bases**, not skips, and those are
  modelled elsewhere. `smollm3` was the only silent-wrong-output case.
- **OWED:** no SmolLM3 GGUF exists on the dev host, so there is **no greedy-parity
  receipt** against the pinned llama.cpp. The tests prove the layer-skip logic and its
  wiring into both CPU forward paths, and nothing more. Specifically unproven:
  end-to-end token identity on real weights, tokenizer/chat-template fidelity, the
  interaction of NoPE layers with SmolLM3's long-context rope scaling, and every GPU
  lane (deliberately refused). **No ledger row claims `smollm3`, and none should be
  added until a receipt exists.**
- **Follow-ups found during the same audit:** (a) `gemma3` was accepted by the dense
  `from_gguf` path while its per-layer dual rope base was modelled only in the runnable
  lane, so a gemma3 file reaching the dense path would have used one base for all layers
  — **CLOSED 2026-07-30 by the gemma3→Metal campaign**: dual-theta RoPE now lives in
  `src/inference/rope.rs::gemma3_rope_tables` and drives the Metal resident lane, and a
  gemma3 file reaching the CPU *dense* path fails closed at the per-layer choke point
  (`ensure_windowed_arch_off_cpu_dense_layer`, DECISIONS D20.2) rather than decoding with
  the wrong schedule; (b) `lfm2` was claimed implemented but its recurrent
  layers carry `shortconv.in_proj/out_proj` and no `attn_q/k/v`, so the dense binding
  failed to bind — a load-time error rather than silent wrongness, but still a claim
  with no path that could run it. **CLOSED by the LFM2 short-conv bring-up (RA-8).**

## RA-8 — `lfm2` had no runnable forward — **CLOSED**
- **What:** `lfm2` sat in the optimized-lane allowlist with no implementation of its
  double-gated short convolution. Its conv layers carry
  `shortconv.{conv,in_proj,out_proj}` and no `attn_q/k/v`, so the dense tensor map
  could not bind them — the same shape as `qwen35`.
- **Resolution:** `lfm2` is now classified [`is_runnable_only_arch`] and its forward
  lives in the runnable lane (`Lfm2Runtime`, `RunnableModel::forward_step_lfm2`),
  ported tensor-for-tensor from llama.cpp `src/models/lfm2.cpp`. `Lfm2Metadata`
  carries the short-conv kernel width and the per-layer conv/attention schedule,
  which llama.cpp derives from the per-layer `attention.head_count_kv` array (a `0`
  marks a conv layer, `lfm2.cpp:10`).
- **Two latent defects found and fixed in the same pass:**
  1. `arch_uses_neox_rope_pairing` omitted `lfm2`, so its attention layers would have
     roped with adjacent even/odd pairing. llama.cpp classifies LLM_ARCH_LFM2 as
     `LLAMA_ROPE_TYPE_NEOX` (`llama-model.cpp:2477` → `:2492`) and the converter
     leaves Q/K unpermuted — the old pairing is silently-wrong output, not a refusal.
  2. `attention_head_count_kv` was read as a scalar. LFM2 stores it as a per-layer
     **array** whose zeros are structural, so the scalar read missed it entirely and
     fell back to `attention_head_count` — 32 instead of 8 on the 2.6B row.
- **A third defect, found only because the forward receipt could not see it:** the
  greedy parity test feeds both sides frozen prompt ids and never constructs a
  `Tokenizer`, so it passed while `Tokenizer::from_gguf` was in fact refusing every
  LFM2 file — `resolve_gpt2_pre_tokenizer` had no arm for `tokenizer.ggml.pre =
  "lfm2"` and fell into the catch-all reject. That took down `smoke_admit`,
  `RunnableServeRuntime::load` and `bench-generate` for the row, i.e. every
  end-to-end path, behind a parity-certified forward. llama.cpp resolves `lfm2` in
  the same switch arm as `llama-bpe` (`src/llama-vocab.cpp:2111-2123`), so the fix
  is the alias — but the alias alone only makes the tokenizer *constructible*, and
  LFM2 carries its own vocab (BOS 124894, not Llama-3's 128000), so it is now
  backed by its own id-agreement receipt rather than by resemblance.
- **Evidence:** three checks against llama.cpp b9632 on the LFM2.5-2.6B Q8_0 row —
  greedy forward parity (`tests/lfm2_parity.rs::lfm2_matches_llamacpp_greedy`,
  receipt `qa/runnable/lfm2-parity.json`), tokenizer id-agreement
  (`::lfm2_tokenizer_matches_llamacpp`), and chat-template fidelity
  (`api::tests::lfm2_renderer_matches_llamacpp_applied_template`).
- **Scope of the claim:** forward graph, tokenizer, and renderer are each proven at
  the prompt level on ONE file at ONE quant on CPU. NOT proven: an end-to-end serve
  run, long context, sampling (the lane decodes greedy), tools (fail closed), other
  quants, GPU lanes, or any other LFM2 variant. No row should claim those.
