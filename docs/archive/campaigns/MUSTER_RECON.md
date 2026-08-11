# MUSTER Phase 1 — Per-row recon dossiers (Gate 1)

Campaign: [`MUSTER_CONDUCTOR.md`](./MUSTER_CONDUCTOR.md) §4. Roster: [`MUSTER_ROSTER.md`](./MUSTER_ROSTER.md) (Gate 0 signed 2026-07-15, sealed on `main` at `6f70f616`). Recon executed 2026-07-16 against that tree, read-only: no serve, no generation, no model weights loaded; the only commands run were `camelid inspect` and `camelid plan-offload` (metadata-only, both documented per row). Every dossier was independently adversarially re-verified against the code; corrections found by that pass are folded in below and marked **[verified]** / **[corrected]** where load-bearing.

**Status: Gate 1 open — Tim skims, decides the flagged items (§Gate 1 decisions), and may re-order waves. Recon-proven HOLDs (M-B5, M-B6) carry committed receipt wording here and exit the pipeline at this gate.**

---

## Cross-cutting ground truth (applies to every row)

- **Oracle binary re-verified on this host**: `llama-server --version` → `version: 9632 (acd79d603)`, built with Clang 20.1.8 for Windows x86_64. Binary SHA-256 `6c787bf07ac1d7e1bbaa1ee176c3ef0df58ea86494c8c1b1d2d9f4a9176b19ae`. Backend inventory: **CPU only** (13 per-µarch `ggml-cpu-*.dll` variants + `ggml-rpc.dll`; **no `ggml-cuda.dll`**) — every MUSTER oracle capture is a CPU reference leg (`-ngl 0`), exactly like the K-quant precedent bundles. Companion tools at the pin: `llama-cli`, `llama-tokenize`, `llama-perplexity`, `llama-bench`, `llama-quantize`.
- **Two-phase capture discipline confirmed in-tree**: `scripts/raw-decode-parity.mjs` `--reference-out` (capture oracle alone, exit) / `--reference-in` (compare live Camelid vs committed capture) at lines 45-50/150-168. **[corrected]** Its `--prompts-file` accepts ONLY a bare JSON array or `{prompts:[…]}` — `qa/speed/prompts.json` (`{columns:[{prompt}]}`) crashes it, and its `--stop`/`--variant`/`--proof-chain` defaults are Llama-3/K-quant-specific. Per-row runs must pass all three explicitly and commit a derived array-form pack before capture. → Conductor Amendment A-3.
- **§7.1 promotion-smoke command is valid as written**: all four `--expect-*` flags exist (`scripts/model-promotion-smoke-bundle.mjs:28-31,49-52`) and are implemented in `frontend/scripts/smoke.mjs:30-33`. Caveat: expectations are asserted only by the frontend leg (`--skip-frontend` would silently drop them — §7.1 doesn't use it).
- **Renderer landscape** (full inventory verified): hardcoded renderers exist for tinyllama-marker, compact-llama3, **phi3** (`render_phi3_prompt`, `src/api/mod.rs:12714-12730` — the conductor's clone-time grep was wrong → Amendment A-2), mistral, qwen3-ChatML (deliberately also catches qwen2 templates, `:12289-12295`), gemma4-marker (byte-incompatible with gemma3 — gemma4 renamed the turn markers, `:4388-4394`), and ornith-ChatML (runnable bridge, rendered **unconditionally** for any runnable-served model — a hazard for widening the bridge, `:5480-5484`). gemma3 templates fall through every detector to the non-exact role-colon fallback (`:12152-12156`). The metadata-Jinja renderer engages three ways: exact Llama-3.2 Q8_0 rows (required), `CAMELID_METADATA_CHAT_TEMPLATE` opt-in (any file), and **unconditionally for tools-bearing requests** whose template is not Mistral/ChatML-shaped (`:12022-12073`).
- **`camelid runnable-smoke`** oracle-qualified combos are exactly llama/qwen3/gemma3/phi3 at Q8_0 (`src/runnable/smoke.rs:42-60`): covers both Wave A rows, refuses every Wave B row — which does **not** gate the runnable-serve vehicle (Ornith precedent). **`camelid verify-receipt`** exists with `--self-only`/`--reference-only` halves (exit 2 = non-reproducible, 3 = divergence record); its reference half starts llama-server on the GGUF → Phase 4-only tool.
- **Env-lane map for capture manifests**: `CAMELID_RUNNABLE_SERVE=1` (runnable bridge), `CAMELID_QWEN35_CUDA=1` (+`_MAXPOS`), `CAMELID_GEMMA4_SERVE=1`, `CAMELID_GPU_RUNNABLE_TIER=0` (opt-out), `CAMELID_CUDA_RESIDENT_DECODE=0` / `CAMELID_DETERMINISTIC` (force CPU), `CAMELID_SKIP_FIT_CHECK=1`. K-quant GPU-resident routing is automatic by tensor scan (`src/execution_plan.rs:350-351`).

---

## M-A1 — `gemma-3-1b-it-Q8_0.gguf` (gemma3, Q8_0) — **PROMOTE (likely)** · verify: SOUND

**Vehicle: runnable serve lane, CPU, chat surface** (`CAMELID_RUNNABLE_SERVE=1`, Ornith precedent). The runnable engine is the only lane whose gemma3 math is proven correct — gemma norm tensors loaded (`src/runnable/model.rs:592-595`), NEOX RoPE (`:599-602`), dual local-10000/global-1e6 RoPE on the every-6th-layer pattern (`:607-622`), sqrt(d_model) embed scale + GeGLU (`:640-645`), tied output embedding — and this exact file carries the SHA-anchored HF-reference greedy-parity receipt `qa/runnable/gemma3-parity.json` (`all_greedy_match=true`, 4 fixtures, max logit diff 1.25e-4). gemma3/Q8_0/SPM is an oracle-qualified smoke combo (`src/runnable/smoke.rs:42-60`).

**Why not the optimized lane** (the load-bearing finding): it *accepts* gemma3 but silently mis-binds it — `expects_qk_norm` is qwen3-only and `forbids_qk_norm` covers only llama/mistral/qwen2 (`src/model.rs:770-771`), so all **104 gemma norm tensors are silently dropped** (verified: 26× each of attn_q_norm/attn_k_norm/post_attention_norm/post_ffw_norm in the file); no GeGLU anywhere in `src/inference.rs`; NEOX pairing explicitly unverified for gemma3 (`src/model.rs:209-218`). It runs end-to-end — *which is exactly why the chip renders today* — but the forward is mathematically wrong, so oracle parity is unearnable there without a campaign-sized build.

**Phase 3 items (named, small, precedented):**
1. Widen `is_runnable_serve_arch` (qwen35-only, `src/api/mod.rs:5183-5185`, load gate `:6421-6424`) to gemma3. **[corrected — deliberate side effect]:** after widening, a gemma3 file loaded *without* the env flag gets a 503 `model_not_ready` on chat instead of today's fall-through to the mis-bound optimized engine — desired fail-closed behavior, but it kills the current chip-visible generation path; Phase 3 makes that choice explicitly.
2. gemma3 marker renderer (`<start_of_turn>{role}` form; assistant→"model"; system folded into first user turn; multimodal branch fails closed) + `qa/prompt-packs/` shapes pack + byte-lock test, dispatched by architecture inside the runnable chat handlers (which currently hardcode Ornith ChatML, `:5481/:5580`). BOS must be supplied exactly once (bridge encodes `add_special=false`, `:5485`).

**Constraints:** raw-decode fallback is NOT available on this vehicle — the runnable bridge short-circuits only `/v1/chat/completions` (`:7831`); `raw-decode-parity.mjs` drives `/v1/completions`, which reaches the wrong engine. If the renderer gate fails, the row HOLDs. SPM merge-order caveat (SPM_MERGE_ORDER_CONDUCTOR.md:103-126) applies to every chat prompt and is **un-audited on gemma's 262k vocab with `add_space_prefix=false`** — mitigate by conservative pack construction BEFORE capture (plain-English prompt classes, anchor on the 4 known-good receipt fixtures). The runnable lane has **no sliding-window mask** and `gemma3.attention.sliding_window=512` → the promoted `tested_context` must stay ≤512 total tokens; bounded-context ladders are OFF for this row. No gemma-family receipt exists at the acd79d603 pin (gemma4 rows pin 5d56eff); arch age + 8 `gemma3` string occurrences in the pinned dll predict PASS, confirmed live only at Phase 4.

**Oracle/pack plan:** new `scripts/chat-parity-gemma3.mjs` (modeled on chat-parity-qwen3.mjs) driving chat; new gemma3 shapes pack + 5-prompt 1/5/50 gate pack committed before capture; file already on disk with catalog-matching SHA → Phase 2 is a re-anchor, not a download. Fit trivial (~1 GiB weights; ~26 MiB KV @512).

**Post-promotion observation for Tim (out of scope):** consider extending the fail-closed arch classification so gemma3 can never bind to the dense llama forward again.

---

## M-A2 — `Phi-3-mini-4k-instruct-Q8_0.gguf` (phi3, Q8_0, not on disk) — **CONDITIONAL GO** · verify: corrections folded

**Vehicle: optimized lane, CPU (`cpu_reference` safe path, deterministic lane for receipts).** phi3 is carried end-to-end today: fused attn_qkv/gate-up expansion at load (`src/model.rs:1697-1777`, called `src/api/mod.rs:6345`), a **phi3 renderer already in-tree** (`render_phi3_prompt` `:12714-12730`, detector `:12302-12306`, routed before the tinyllama look-alike), `<|end|>` in the additive EOG set (`src/tokenizer/mod.rs:388-400`). Commit `92029b7e` proved load + clean short answers and names the one defect: *"long/open-ended phi3 generation still degenerates — a forward-path divergence."* Runnable lane rejected: engine-side stronger (NEOX correct there) but the serve bridge is qwen35-only by design — widening it is more code than the optimized lane's fix, for worse throughput.

**Phase 3 items:**
1. Arch-gated NEOX RoPE flip for phi3 (`src/model.rs:218`; in-code comment: phi3 "very likely" needs it, "out of scope and unverified"; the runnable lane independently asserts NEOX for phi3, `src/runnable/model.rs:600-602`). Pre-probe with `CAMELID_ROPE_PAIRING=split_half` (`src/inference/rope.rs:62-71`) on the pulled file BEFORE writing code. If the probe doesn't restore long-generation coherence, the divergence is something else → likely HOLD.
2. Pack+gate retro-fit for the existing renderer (its only gate is one unit literal, `:4441-4464` — insufficient under §6.1): `qa/prompt-packs/phi3-template-shapes` + byte-lock test + `scripts/chat-parity-phi3.mjs`. Two renderer behaviors must be adjudicated vs the reference before locking: unconditional trailing `<|assistant|>\n` (even after a trailing assistant turn) and unknown-roles→user normalization.

**Blockers (updated by the adversarial pass):**
- **B-SWA [added by verify — load-bearing]:** Phi-3-mini-4k declares `sliding_window=2047` (HF config; verify `phi3.attention.sliding_window` at Phase 2 inspect). llama.cpp applies SWA for phi3 when the GGUF carries the key; **Camelid has no phi3 SWA anywhere** (`src/model.rs:138-222` reads no such key; SWA exists only in gemma4-CUDA/DiffusionGemma). If the pulled GGUF carries it, every parity leg crossing ~2047 positions diverges **by construction** — and SWA absence is a second suspect for the long-generation degeneration. Consequence: promoted envelope scoped ≤2047 positions, or named blocker.
- B1 acquisition: ~4.06 GB pull (`camelid pull phi3_mini` resolves uniquely); size must match the catalog literal 4,061,222,688 B exactly — bartowski re-upload risk is real; mismatch = stop-and-amend.
- B3 SPM merge-order: LIVE on main (`src/tokenizer/mod.rs:862-930`; PR #357 was docs-only) and phi3 chat prompts route `parse_special=true`. TinyLlama scar applies. Attribute or HOLD; raw-decode de-scope carries the first-of-family disclosure (§4.3).
- B6 contract-wording collision: `planned_model_families` item `phi_falcon_mamba_others` ("future lanes only," `src/api/mod.rs:2872-2876`) must be amended in the same Phase 6 commit.
- Fit realities: fit-advisor `wont_fit` at Phase 0 was the conservative advisory (never gates); honest clean-load expectation on this host is fits_resident/fits_with_offload arithmetic **[corrected — `cpu_only_ok` unreachable with a usable GPU]**. GPU lane: phi3 is excluded from the GPU-runnable tier (`src/execution_plan.rs:833-847`) — CPU promotion, GPU as stretch only.

---

## M-B1 — `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (llama BPE, Q4_K_M) — **GO** · verify: corrections folded

**Vehicle: optimized lane, GPU-resident CUDA K-quant raw-decode** — identical to the promoted 3B Q4_K_M/Q5_K_M siblings; the 3B Q4_K_M bundle was captured on THIS card. Inspect: 147 tensors = 96 Q4K + 17 Q6K + 34 F32, tied Q6K token_embd `[2048,128256]`, `file_type=15` → one run drives both `q4k_gemv` and `q6k_gemv`; static plan verified two ways: plan-offload 16/16 layers resident (557 MiB weights + 256 MiB KV @4096) and `select_kquant_plan` → `cuda_resident_kquant_runtime` (mislabel fix in place, test-asserted `src/execution_plan.rs:1557-1569`). Every contraction dim is 256-aligned. Tokenizer is gpt2/llama-bpe — SPM caveat N/A. Chat scoping: raw-decode row (`exact_row_gpu_resident_raw_decode_parity_smoke_only`), the 3B precedent applied at 1B, with the sibling 1B Q8_0 row carrying the chat surface — the conductor's own condition for raw-decode-only rows holds here. **[corrected]** A fourth renderer path exists: tools-bearing requests render the file's metadata Jinja unconditionally — contract wording must not assume the metadata renderer is unreachable.

**Run-plan corrections from the adversarial pass (all mandatory):**
1. `--prompts-file qa/speed/prompts.json` **crashes the harness** (shape mismatch). Assemble and commit the 8-prompt array-form pack (France BOS probe + the 7 speed columns — verified 7/7 identical to the 3B bundle's recorded prompts) before oracle capture.
2. The 3B bundle records **no prompt-token parity** (generated-token parity only), while §6.2 requires both. The M-B1 run must capture prompt-token parity explicitly (the IQ4_XS sibling receipt shows the pattern).
3. The 3B near-tie precedent is thinner than its reputation: 1 measured logprob gap of 3 flips, **no oracle-side control**. The oracle capture must include `n_probs` logprob data for every probe prompt plus an oracle-side control leg, or open-ended flips cannot be attributed under §6.3 (1B at 4-bit will plausibly flip more probes than the 3B did).

Other care: contract id filename-anchored (`llama_3_2_1b_instruct_q4_k_m`) so the frontend matcher resolves before the Q8_0 branch; provenance anchored at Phase 2 (don't repeat the 1B-Q8_0 unsloth/bartowski wording mismatch); `tool_capable` false-by-evidence or not claimed (the 3B K-quant FAILed agent-eval); single-engine + free-RAM discipline.

---

## M-B4 — `qwen2.5-0.5b-instruct-q4_0.gguf` (qwen2, Q4_0) — **CONDITIONAL (largest bite) — Gate 1 decides** · verify: corrections folded

As-is, **neither lane computes a correct qwen2 forward, and no serve lane can carry it** — three engine blockers, all verified in code:
1. **QKV biases dropped (both lanes):** the file carries 72 attention bias tensors (attn_q/k/v.bias × 24 layers, confirmed by inspect); no lane loads or applies them (optimized binding has no bias fields `src/model.rs:691-710`; runnable loader fetches only `.weight` names; dense forward adds no bias — decode matvecs at `src/runnable/model.rs:777-779`, prefill `:899-901` **[corrected locations]**). llama.cpp applies them → parity fails at token 1.
2. **NEOX RoPE missing for qwen2** (`src/model.rs:218` flips only qwen3/qwen35; comment admits qwen2 "very likely" needs it; runnable inherits false, `:600-602`).
3. **[added by verify — load-bearing]: the runnable serve bridge cannot carry qwen2 at all** — arch gate is qwen35-only, `/v1/completions` has no runnable short-circuit (so raw-decode scoping can't exercise the runnable lane either), the bridge renders Ornith ChatML unconditionally, and non-qwen35 runnable generation ignores the EOG stop set (`src/runnable/model.rs:1578-1585→:721-745`). Serve-bridge work (arch gate + renderer routing + stop handling, or a runnable completions bridge) is a third named Phase 3 item.

Renderer: the qwen3-ChatML renderer captures the template but is NOT byte-exact (omits the Qwen2.5 default system prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."; injects the qwen3-only `<think>\n\n</think>\n\n` prefill). Three options: qwen2.5 renderer branch + pack (detector ordered before the ChatML catch-all), raw-decode scoping with the first-of-family disclosure, or **[added by verify]** the in-tree opt-in metadata-Jinja lane (`CAMELID_METADATA_CHAT_TEMPLATE`) — the qwen2.5 template avoids the constructs that break minijinja on qwen3; plausibly byte-exact, unverified. Doc constraints: COMPATIBILITY.md:73-80 locked Qwen-2.5 wording pins the 7B file — must remain intact; a promoted 0.5B row would be the first certified non-QAT Q4_0 row, so the three Q4_0 truth surfaces (`src/api/mod.rs:2801-2805`, `:2817-2822`, `:3571`) must be updated in the same commit.

**If Gate 1 declines the engine work, this row HOLDs today** with receipt: "qwen2 QKV-bias and NEOX-RoPE unimplemented in both lanes; runnable serve bridge is qwen35-only — forward numerically wrong vs the pinned oracle; parity unattainable without engine work." Everything else about the row is easy (429 MB, tokenizer fully covered with the qwen2 pre-tokenizer tested, oracle trivially capable at the pin).

---

## M-B2 — `ornith-1.0-9b-Q6_K.gguf` (qwen35, Q6_K) — **GO (tempered)** · verify: one refutation folded

**Vehicle: runnable serve lane, CPU** — qwen35 exists only there (optimized lane fails closed by design, `src/api/mod.rs:5159-5161`); the CUDA lane's dispatch DOES map Q6K (`src/runnable/model.rs:2075-2087`) but the build is all-or-nothing full upload and **7,018 MiB weights > 6,144 MiB card** → CPU is the only vehicle (no partial residency in the qwen35 lane). Renderer, template (byte-identical hash to promoted siblings), tokenizer (BPE — SPM caveat N/A), oracle at pin (llama.cpp ran this exact file: PPL 2.3636, `-ngl 22` baseline 10.72 tok/s), and pack (frozen `qa/ornith/constrained-vram/FIXTURES_five_prompt_parity.json`) are all sibling-precedent clean.

**[REFUTED and replaced]** "CPU speed unmeasured": a committed receipt already measures it, and it is brutal — `RECEIPT_ITEM5_acceptance_economics.json:190-193`: the Q6_K 120-token batched prefill on the runnable CPU lane **did not complete one measurement in >20 minutes** (vs Q8_0 ~19 s), because non-Q8_0 prefill falls back to per-input matvec (`src/runnable/model.rs:269-276`) — implying roughly ~10 s per full weight pass. The planned 5-prompt × 64-token capture is plausibly **1+ hours**, and "promotes on whatever pack completes" is the likely path, not a contingency. Budget the Phase 4 session accordingly (or Gate 1 may prefer to deprioritize this row).

Doc sync at promotion: move Q6_K out of the Ornith do-not-claim list (COMPATIBILITY.md:35) in the same atomic commit. Memory safety: 6.85 GiB file on the 15.7 GiB host — free-RAM check + single-engine discipline mandatory.

---

## M-B3 — `ornith-1.0-9b-IQ4_XS.gguf` (qwen35, IQ4_XS) — **GO-WITH-RISK** · verify: nits folded

**Vehicle: runnable serve lane, CPU.** The runnable decode path is arch-generic per-tensor (`decode_iq4_xs_tensor` dispatched on tensor type alone, `src/runnable/dequant.rs:38`), so #455's IQ4_XS coverage composes with qwen35 at admission/dequant. The qwen35 CUDA builder rejects IQ4XS (one prep-closure mapping is the whole gap — `iq4xs_gemv` itself is shipped; llama.cpp ran this exact file fully resident @16K at 5,381 MiB); recorded as a named **optional** Phase 3 stretch, never a gate. **Session-time note (extends the M-B2 refutation):** IQ4_XS rides the same generic non-Q8_0 CPU path that DNF'd for Q6_K — expect the same order of slowness at capture; plan the session (or run depth-limited legs) accordingly.

**Distinct risks:**
1. **The single most likely HOLD path in Wave B:** the #455 receipt's unquantified systematic camelid-vs-llama.cpp near-tie offset (both Camelid paths agree with each other and flip against the reference at the same early token on 2/4 prompts; no KL probe exists). On a 9B hybrid at 4.25 bpw, expect flips; §6.3 demands attribution with oracle-side controls (the Ornith Q4_K_M ≤0.33-nat precedent) or the row HOLDs with the divergence receipt.
2. **Privacy:** this file's GGUF metadata embeds operator home paths in its `quantize.imatrix.*` keys (in-house requant) — any inspect dump or manifest reproducing raw metadata must be scrubbed before commit (§2).
3. Provenance: in-house requant (bf16 source + committed imatrix), not an HF upload — Phase 2 anchors local production, no repo id.
4. Doc sync: COMPATIBILITY.md:35 non-claim + the contract IQ4_XS quant item ("Other IQ4_XS files load to the unverified experimental lane only") gain a second named row in the same commit. Fix in passing: QUANT_QUALITY_TABLE.md:43-44 still claims "Camelid cannot currently parse IQ files at all" — false since #455.

---

## M-B5 — `ornith-1.0-9b-bf16.gguf` (qwen35, BF16) — **HOLD (seal at Gate 1)** · verify: confirmed

No vehicle exists, verified in code on both axes plus the host: (1) runnable lane rejects BF16 on the quant axis (`is_covered_quant`, `src/runnable/admit.rs:177-190`; live reject names `output.weight`); (2) the BF16-capable optimized reference path never binds qwen35 (arch exists only in the runnable lane); (3) 17,920,696,512 B = **16.7 GiB vs 15.7 GiB host RAM** — the runnable lane loads every weight resident; `plan-offload` says `PLAN FAILED … short by 454 MiB` even fully host-offloaded; the oracle capture leg is equally over-RAM, so no side-by-side is executable on this host regardless of Camelid work. Receipt wording per the runnable-lane memory policy: blockers 1-2 are Camelid gaps; blocker 3 is a **host limit, not a refusal**. COMPATIBILITY.md:35 already carries the honest non-claim; only the experimental-lane HOLD annotation is added (§0 door 2).

**HOLD receipt (ready to commit):** `ornith-1.0-9b-bf16.gguf` (sha `27bc7534…`) — HOLD. Named blockers: (1) BF16 absent from the runnable covered-quant set (requires committed BF16 dequant-parity evidence per lane spec); (2) no non-runnable qwen35 vehicle exists (`src/api/mod.rs:5159-5161`, `src/model.rs:613-622`); (3) 17.9 GB file over-RAM on the 15.7 GiB host — host limit, not refusal (precedent: the requant imatrix was deliberately computed on the Q8_0 host for this exact reason, QUANT_QUALITY_TABLE.md:6-8); GPU statically ruled out (plan-offload PLAN FAILED, −454 MiB).

---

## M-B6 — `ornith-1.0-9b-IQ3_XXS.gguf` — **HOLD (instant, seal at Gate 1)** · verify: corrections folded

GGUF parse fails closed before admission, reproduced live twice this session (inspect AND plan-offload): `unsupported GGUF feature: tensor token_embd.weight has unknown or removed GGML type Unknown(21)`. Root cause: GGML type id 21 (upstream **IQ3_S** — the embedding member of the IQ3_XXS mix recipe) has no arm in `GgufTensorType::from_id` (`src/gguf/reader.rs:66-92`; the map covers 0-3, 6-15, 20, 23-28, 30, 34-35 **[corrected]**); `layout()` returns `None` for `Unknown` (`:123` **[corrected — the arm exists and returns None]**) and `tensor_nbytes` fails closed (`:530-535`). The whole IQ2/IQ3/IQ1 family (ids 16-19, 21-22, 29) is unmapped — mapping 21 alone only moves the failure. Chip consequence: the file can never reach `runtimeReady`, so the chat chip is unreachable at this head; only the static Models-page NotAnchoredRow renders.

**HOLD receipt (ready to commit):** named blocker = Camelid GGUF-reader/format gap (IQ3 family unimplemented; a fix is a format campaign — reader map + layout + dequant-parity evidence + kernels per the `wire_dequant.rs:47-51` discipline and the #455 precedent — not a patch). Oracle-side control: llama.cpp at the pin runs this exact file (PPL 2.5323, 16K residency 4,281 MiB — QUANT_QUALITY_TABLE.md:24), isolating the blocker to Camelid format coverage. Contract posture already honest (`src/api/mod.rs:2814`; COMPATIBILITY.md:35). Exit: HOLD until an IQ3 lane earns its own campaign (the recorded follow-up preference is IQ2_XXS on the #455 rails, not IQ3).

---

## Gate 1 decisions for Tim

1. **M-B4 (qwen2.5-0.5b): authorize the three-item engine bite** (QKV bias load/apply + qwen2 NEOX flip + serve-bridge work) **or HOLD it now** with the drafted receipt. This is the largest Phase 3 item on the roster; everything else about the row is easy.
2. **M-A1/M-A2 renderer path sign-off:** gemma3 = new renderer + serve-bridge widening (note the deliberate fail-closed side effect on env-OFF gemma3 chat); phi3 = pack+gate retro-fit on the existing renderer + the pre-probed NEOX flip. Raw-decode de-scope is available for phi3 (with first-of-family disclosure) but NOT for gemma3 (no completions bridge on the runnable lane).
3. **M-B2/M-B3 session budgeting:** accept the committed Q6_K CPU DNF evidence — captures may take hours per row on the generic non-Q8_0 path; confirm both stay in scope (or re-order/deprioritize).
4. **Seal M-B5 + M-B6 HOLD receipts at this gate** (wording above), leaving 6 live rows for Phases 2-6.
5. **Wave order within the live rows** (proposed): M-B1 (mechanically cleanest, pure precedent) → M-A1 → M-A2 → M-B3 → M-B2 → M-B4 (if authorized). Wave A-before-B was the default; M-B1 jumping the queue is a proposal, not a decision — strike if unwanted.

## Conductor amendments landed with this dossier (§10)

- **A-2:** §4.3's "clone-time grep found no gemma3/phi3-specific renderer" is half wrong — a phi3 renderer exists (`render_phi3_prompt`, detector-routed, one unit literal, no pack). gemma3 truly has none.
- **A-3:** §6.2 prompt-pack consumability — `raw-decode-parity.mjs --prompts-file` accepts only array-form JSON (or `{prompts:[…]}`); `qa/speed/prompts.json` is not drop-in (crashes); raw-decode rows derive and COMMIT an array-form pack before capture. Also: `--stop`/`--variant`/`--proof-chain` defaults are Llama-3/K-quant-specific — every other row passes them explicitly.
- **A-4 (no-change finding):** §7.1's four `--expect-*` flags all exist and are implemented; command valid as written; expectations assert only on the frontend leg.
