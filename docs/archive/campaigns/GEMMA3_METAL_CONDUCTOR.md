# GEMMA3 → METAL GPU-RESIDENT LANE — CONDUCTOR

Campaign branch: `feat/gemma3-metal-resident` (worktree /Volumes/Untitled/Camelid-g3metal), base
`origin/main` @ d9053ec4. Scope target: the promoted `gemma-3-1b-it-Q8_0` row (id
`gemma_3_1b_it_q8_0`, src/api/mod.rs:4044) on the M4 Mac mini (16 GB). All file:line references
are into this checkout at the base commit. Sections 1-6 are the scoping report; the Phase 0
record (decisions + reachability recon, docs-only gate) follows at the end.

---

## 1. Current state

**How gemma3 serves today.** gemma3 is served exclusively through the CPU runnable lane. The serve
router fail-closes it there by architecture name: `is_runnable_serve_arch` matches
`qwen35 | gemma2 | gemma3` (src/api/mod.rs:7163-7165), with the stated rationale that "the optimized
dense binder silently drops gemma3's QK/post norms and has no GeGLU" (src/api/mod.rs:7157-7159).
Chat requests short-circuit to the runnable bridge (src/api/mod.rs:10167-10180); the bridge is
default-on since PR #547 with opt-out `CAMELID_RUNNABLE_SERVE=0` (src/api/mod.rs:7138-7155,
8555-8559). Raw `/v1/completions` fails closed with a typed 422 keyed on the same predicate
(src/api/mod.rs:7471-7513, 9815-9820). The Metal resident lane is therefore unreachable for gemma3
regardless of kernel capability.

**The forward pass.** The only correct gemma3 forward is the runnable lane's generic parametric
pre-norm f32 decoder (src/runnable/model.rs:1, switches set at load, :602-661). Per decode step it
re-dequantizes every layer's seven weight matrices to f32 and walks the 262,144-row tied LM head
row-by-row (src/runnable/model.rs:776-903, :886-891, tied-head fallback :435-445), via sequential
`Mat::matvec` (src/runnable/model.rs:49-57) — the rayon `par_matvec` is qwen35-only (:102-109).

**Measured baseline.** ~0.2 tok/s (~5 s/token) on this M4 mini, recorded in the chat-parity evidence
bundle README (qa/evidence-bundles/gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb/)
and reflected in the support row (src/api/mod.rs:4036-4083).

**Model shape (anchored to the bundle's dense_metadata,
qa/evidence-bundles/gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb/api-webui/completion.response.json):**
26 layers, 4 attention heads / 1 KV head (4:1 GQA), head_dim = rope_dim = 256, d_model 1152
(q projection 1152→1024), ffn 6912, vocab 262,144, n_ctx_train 32,768, RMSNorm eps 1e-6, tied LM
head, 999,885,952 params, 183 Q8_0 + 157 F32 tensors.

**Correctness receipts already in hand:**
- HF-transformers greedy parity, `all_greedy_match`, max logit abs diff 1.25e-4
  (qa/runnable/gemma3-parity.json:2-9, test tests/runnable_parity.rs:61-63).
- llama.cpp (pinned acd79d603) chat-parity bundle: 4/5 prompts token-and-text identical at depths
  1/5/50; one disclosed near-tie flip at position 16 of one 50-token leg, 0.3416-nat gap
  (bundle README + manifest).
- Byte-locked chat-template pack (qa/prompt-packs/gemma3-chat-template-shapes-v1.json, in-src lock
  test src/api/mod.rs:6421-6470) and a reusable parity harness (scripts/chat-parity-gemma3.mjs,
  gate pack qa/prompt-packs/gemma3-chat-gate-pack-v1.json).

**Known correctness ceiling of the current lane.** No sliding-window mask exists anywhere on the
gemma3 path; `gemma3.attention.sliding_window=512` is never read (only gemma4 metadata at
src/model.rs:492-493 and the gait inspector src/gait/mod.rs:120 read that key). Sequences at or
beyond 512 tokens are disclosed as "mathematically wrong by construction" in the support row's
blockers (src/api/mod.rs:4051), and the row's `next_step` explicitly demands a window mask or an
optimized-lane gemma3 forward before any ≥512-token context claim (src/api/mod.rs:4083). The Metal
port is that next step.

---

## 2. Gap analysis

Requirement inventory for gemma3-1b vs the Metal GPU-resident lane at d9053ec4. "Generic lane" =
`ResidentDecodeState` + `encode_attention_block`/`encode_ffn_block` driven by
`LlamaInferenceSession`; "gemma4 lane" = the separate opt-in `Gemma4ResidentModel` stack
(src/metal.rs:6543-6735), which is the in-tree template for every gemma-shaped feature.

| # | gemma3 requirement | Metal-lane status | Receipts |
|---|---|---|---|
| 1 | RMSNorm (pre-attn, pre-FFN, final), eps from GGUF | **Supported** | Generic lane norms + GPU LogitsStage final norm (src/inference/metal_resident.rs:334-550) |
| 2 | Per-head QK RMSNorm on Q and K before RoPE | **Supported** | `rms_norm_per_head_f32` kernel (src/metal.rs:1712), wired into decode (src/metal.rs:10069-10104) and prefill (src/metal.rs:11738 area); eligibility comment src/inference.rs:2725-2727; gemma-shape unit test 8x256/8x512 (src/metal.rs:15328-15355) |
| 3 | Post-attention + post-FFN sandwich norms (before each residual add) | **Missing in generic lane; wired in gemma4 lane** | Generic blocks lack them; gemma4 `encode_gemma4_ffn` "adds the extra post_ffw_norm" (src/metal.rs:8248-8253); reference semantics src/runnable/model.rs:304-307, tensor names :597-598 |
| 4 | GeGLU (gelu-tanh(gate) \* up) | **Missing in generic lane; kernel exists** | `encode_ffn_block` hardcodes SiLU (src/metal.rs:9776); `gelu_mul_f32` kernel exists and mirrors the CPU reference exactly (src/metal.rs:1760-1777), wired only into the gemma4 lane |
| 5 | NEOX split-half RoPE pairing | **Partial** | Kernel supports both pairings via runtime flag (`rope_rotate_f32`, src/metal.rs:1813-1843), but `arch_uses_neox_rope_pairing` excludes gemma3 (src/model.rs:812-814, test assert :2124) — host must force pairing=1 as the gemma4 encode does (src/metal.rs:8525) |
| 6 | Per-layer dual RoPE theta (local base 10000 on 5-of-6 layers, global freq_base 1e6 on layers 5/11/17/23) | **Missing in generic lane; template in gemma4 lane** | Generic lane builds ONE cos/sin table per forward and ropes every layer with it (src/metal.rs:11516-11554; documented lane-wide at src/inference.rs:2693-2705; single freq_base at src/inference/rope.rs:574-629). gemma4 threads per-layer tables (src/metal.rs:6982-7009, src/gemma4_runtime.rs:2766-2801). gemma3 needs only TWO tables per token + a per-layer selector — simpler than gemma4. Reference schedule src/runnable/model.rs:607-626 |
| 7 | Sliding-window mask, window 512, 5-of-6 local layers (decode) | **Missing host wiring; kernels ready, zero new MSL** | Every decode-attention kernel takes `kv_base_offset`/strides (v1 src/metal.rs:1872,1879; kv16 :1934,1941; v2 :1994,2004; split-K :2089,2100; and tree twins :4117-4384). Generic path hardcodes offset 0 (src/metal.rs:10008-10012). gemma4 lane implements the window as `position_count = filled - window_start`, `kv_base_offset = window_start * head_dim` (src/metal.rs:8452-8453, scalar write :8537), per-layer `window_start = filled.saturating_sub(window)` (src/gemma4_runtime.rs:2789-2793). Regression test locks kernel reuse at head_dim 256 (src/metal.rs:15362-15456) |
| 8 | Sliding-window in batched prefill | **Missing everywhere (real kernel gap)** | Flash-prefill `kv_base_offset` is a uniform shift (src/metal.rs:3521,3778,3892); causal mask is upper-bound-only (src/metal.rs:3476-3489, :3663) — no per-query-row lower bound exists in any batched prefill kernel. Moot for correctness: see row 10 |
| 9 | Decode attention at head_dim 256, 4:1 GQA | **Supported via v1 kernels only** | v1 `attention_decode_f32`/`_kv16` take arbitrary head_dim + integer GQA (src/metal.rs:1859-1974, kv_head = head/group :1877). v2/split-K hard-capped at head_dim ≤ 128 and the caps are memory-safety-critical (MAX_DPL=4 :2000, sh_acc[NSG\*128] :2038, k_s/v_s[PT\*128] :2118-2119; host gates :9501, :9507-9512). gemma4 lane runs 256/512-dim heads on v1 in production (comment src/metal.rs:8410) |
| 10 | Batched GPU prefill at head_dim 256 | **Missing** | `prefill_tokens` bails on head_dim > 128 (src/metal.rs:11723-11727); flash prefill kernels require ≤ 128 (src/metal.rs:3506, :12327). gemma4's answer: token-by-token prefill through the decode path (src/gemma4_runtime.rs:4919-4949), which gets per-layer windows for free |
| 11 | Speculative batched verify | **Unavailable and must stay gated** | `verify_batch_inner` bails on head_dim > 128 (src/metal.rs:12841-12843); additionally the verify scalar layouts hardcode kv_base_offset=0 (src/metal.rs:9215, 9294), so window-correct verify would need its own plumbing |
| 12 | Embedding scale sqrt(d_model) | **Missing in generic embed path (trivial)** | Reference: src/runnable/model.rs:645-650, applied :704-711, :790-795 |
| 13 | Tied LM head, 262,144-row output GEMV | **Supported** | GPU LogitsStage + Q8_0 wire GEMV handles arbitrary row counts (ragged-row guards src/metal.rs:1154,1180); GPU sampling needs the Q8_0 wire embed table for the gather (src/inference/metal_resident.rs:480-504) |
| 14 | Q8_0 GEMV/GEMM shape fit (k ∈ {1152, 1024, 256, 6912}) | **Supported** | All k dims %32 == 0; production GEMV `q8_0_block_linear_row_ksplit_f32y_wire_nsg8` (src/metal.rs:1124-1186) requires only that |
| 15 | Eligibility dims gate | **Passes** | hidden 1152 %32, q_dim 1024 %32, head_dim 256 even, ffn 6912 %32, 4 % 1 == 0 (src/inference.rs:2883-2892) |
| 16 | No logit softcap / no logit_scale (gemma3 has neither) | **Supported** | Generic lane applies neither; reference src/runnable/model.rs:348-350, :628-630 |
| 17 | GGUF parse of window/pattern/local-base keys | **Missing on the gemma3 path** | Nothing gemma3-side reads `attention.sliding_window`/`sliding_window_pattern`; runnable lane hardcodes pattern=6, local base 10000 (src/runnable/model.rs:607-626). Template: gemma4 parse (src/model.rs:451-460, :486-494) |
| 18 | Optimized dense binder carries gemma3 tensors | **Missing (mis-bound today)** | Binder drops all 104 norm tensors (26× attn_q_norm/attn_k_norm/post_attention_norm/post_ffw_norm), no GeGLU, adjacent even/odd pairing (src/api/mod.rs:7157-7159; src/model.rs:812-814) |
| 19 | Serve routing to the resident lane | **Missing** | `is_runnable_serve_arch` diverts gemma3 before the resident engine is reachable (src/api/mod.rs:7163-7165); resident decode is library-default-OFF, enabled by the CLI fast stack (src/inference.rs:12183-12186; src/main.rs:5721-5735) |
| 20 | Execution-plan row recognition | **Missing** | No gemma3 branch in `recognized_row_level` (src/execution_plan.rs:975-1012); `is_supported_exact_q8_row` (src/execution_plan.rs:1014-1021) gates Metal-resident Q8 plan selection (src/execution_plan.rs:318-320) — without this edit the resident plan is never selected even with all kernels wired |
| 21 | KV cache with per-layer window reads | **Supported by layout; plumbing missing** | Full-length per-layer buffers `[kv_head][max_positions][head_dim]`, doubling growth, no ring buffer needed — window is purely a read-range restriction (src/metal.rs:10904-10957, :11033-11061, ensure_capacity :11096-11184); gemma4 proves per-layer window/head_dim on the same layout (src/metal.rs:6615-6623, 13769-13842) |
| 22 | GPU→CPU KV mirror-back for fallback reads | **Supported** | `ensure_cpu_kv_materialized` (src/inference/metal_resident.rs:208-257) |

Net kernel verdict (adversarially verified): a **correct decode-side gemma3 port requires zero new
MSL kernels**. All gaps are host wiring, estimated ~500-850 LOC across ~6 files (src/metal.rs
~250-400; src/inference/metal_resident.rs ~100-150; src/inference/rope.rs ~40-80; src/model.rs
~40-80; src/inference.rs ~30; src/api/mod.rs ~50-100). The only genuine new-kernel candidate is an
optional batched windowed prefill with head_dim-256 support (~200-400 MSL LOC + host + tuning),
deferrable by adopting gemma4-style token-by-token prefill.

---

## 3. Phase plan

Ordered per this repo's campaign style (qwen3 PRs #266/#275/#278; GABBRO #477-#482): recon →
binder → kernels-with-self-parity → eligibility flip in the same commit → serve reachability →
evidence bundles → promotion surfaces → perf last. Correctness gates precede all perf work.

### Phase 0 — Gates and prerequisites (hours)
Deliverables:
- Reachability recon doc: run the row with `CAMELID_RESIDENT_TRACE=1` and enumerate every bail that
  fires in `resident_decode_eligible` (src/inference.rs:2679-2893). Precedent: WIN2METAL A2
  (9ad5c9e7, docs-only) and GABBRO M1 (2ad74557, evidence-only).
- Lane-architecture decision recorded in the doc: **recommended** — extend the generic lane
  (window param on `encode_attention_block` + per-layer (window_start, theta-table) threading in
  `prepare_token`, GeGLU/post-norm encodes, per gap rows 3-7/12), with gemma4-style token-by-token
  prefill. Alternative (clone `Gemma4ResidentModel` as a gemma3 runtime driver, ~1-2k LOC borrowed
  structure, near-zero metal.rs changes) is acceptable but heavier.
- Scope pin: 1B Q8_0 row only. 4B/12B/27B are out (rope-scaling rejection
  src/runnable/model.rs:395-403; attention-scale coincidence, see Risks).
- Parity-envelope policy decision: whether the GPU lane inherits the disclosed single-flip
  0.3416-nat envelope from the runnable bundle or must match the runnable lane token-exact.
  Gate: doc committed; no code.

### Phase 1 — Config + dense binder (1-2 days)
Deliverables:
- Parse `gemma3.attention.sliding_window`, `sliding_window_pattern`, and the local freq base from
  GGUF (template: gemma4 parse at src/model.rs:451-494), replacing the hardcoded pattern=6 /
  local-base-10000 constants for the resident path (reference src/runnable/model.rs:607-626).
- Teach the optimized dense binder the gemma3 tensor set: attn_q_norm/attn_k_norm,
  post_attention_norm/post_ffw_norm (104 tensors), GeGLU flag, embed scale, forced NEOX pairing,
  per-layer rope-base schedule. Precedent: qwen3 Gate 1 (c8f886f6, tests/model_binding.rs pattern).
Parity gate: binding test asserting all 104 norm tensors bind and the schedule/window metadata
round-trips from the real GGUF.

### Phase 2 — Metal resident forward, correctness-first (2-4 days)
Sub-steps, each landing with its self-parity gate in the same commit (GABBRO M3 pattern, 6b265027):
- 2a QK-norm wiring for gemma3 (mechanical replay of d56f7da2; kernel reuse, ~90-line diff class).
- 2b Sandwich post-norms in the resident attention/FFN blocks (reuse existing RMSNorm encode).
- 2c GeGLU via existing `gelu_mul_f32` (src/metal.rs:1760-1777). Do NOT fuse gate+up — the fused
  variant was previously reverted for register spill (Metal parity campaign).
- 2d Per-layer dual-theta RoPE: build TWO cos/sin tables per token (local 10000 / global 1e6) plus
  a per-layer selector; force split-half pairing host-side (gemma4 precedent src/metal.rs:8525).
- 2e Sliding-window decode mask: add `window_start` to `encode_attention_block`, apply the two-line
  gemma4 math (src/metal.rs:8452-8453), thread per-layer window starts through `prepare_token`.
  Preserve the exact off-by-one convention: window INCLUDES the current position,
  "attend [pos+1-window ..= pos]" (src/model.rs:632-635).
- 2f Embed scale sqrt(1152) on the resident embed gather; verify the GPU sampling gather path.
- Prefill: token-by-token through the decode path (gemma4 precedent src/gemma4_runtime.rs:4919-4949)
  — no new MSL; batched prefill is out of scope here (head_dim 256 exceeds every prefill kernel cap).
Parity gate: per-kernel CPU-vs-GPU self-parity tests (extend the pattern at
src/metal.rs:15362-15456), plus a full-forward logit comparison vs the runnable lane at depths
1/5/50 under 512 tokens.

### Phase 3 — Eligibility, execution plan, serve reachability (~1 day)
Deliverables:
- Flip/condition the gemma3 disqualifiers in `resident_decode_eligible` in the SAME commit that
  completes wiring (d56f7da2 precedent).
- Add a gemma3 branch to `recognized_row_level`/`is_supported_exact_q8_row`
  (src/execution_plan.rs:975-1021) so the Metal-resident Q8 plan is selectable
  (src/execution_plan.rs:318-320). This step is absent from the historical checklist and is
  load-bearing.
- Serve routing: remove gemma3 from `is_runnable_serve_arch` (src/api/mod.rs:7163-7165) with the
  runnable lane as fallback. Handle the side effects explicitly: `/v1/completions` reopens
  automatically (same predicate via `completions_unsupported_for_arch`, src/api/mod.rs:7478); the
  "runnable-runtime" backend/health label and its test (src/api/mod.rs:2571, :15759) and the
  qualified-row comment scope (src/api/mod.rs:17357-17365) go stale.
- Keep speculative decode gated off for gemma3 (naturally blocked by src/metal.rs:12841-12843, but
  add an explicit gate + test given the kv_base_offset=0 hardcodes at src/metal.rs:9215, 9294).
Parity gate: end-to-end serve smoke on the resident lane; runnable fallback verified when
`CAMELID_METAL_RESIDENT_DECODE=0`.

### Phase 4 — Parity receipts and the ≥512-token window claim (1-2 days)
Deliverables:
- Rerun scripts/chat-parity-gemma3.mjs + scripts/raw-decode-parity.mjs on the resident lane at
  depths 1/5/50 vs pinned llama.cpp acd79d603; commit
  qa/evidence-bundles/gemma3-1b-q8-gpu-resident-parity-<date>-<head>/ (README + SHA256SUMS +
  manifest.json + parity json; pattern af928fdd/65772ff2). Pass all four bundle validators
  (check-public-scrub.sh, audit-evidence-bundle-privacy.mjs --strict,
  check-evidence-bundle-checksums.sh, check-public-evidence-claims.mjs — recorded in af928fdd).
- NEW windowed-context receipt: a ≥512-token oracle pack vs llama.cpp (none exists today; bounded
  context ladders were explicitly off for this row). This is the deliverable that unlocks the
  context claim the runnable row could not make (src/api/mod.rs:4083), with the gemma4 lane as an
  internal window-semantics cross-check.
- Determinism receipt: byte-identical decode across two serve sessions.
Parity gate: bundle committed; flips (if any) within the Phase-0 envelope policy.

### Phase 5 — Promotion surfaces (1-2 days)
Deliverables:
- Capabilities row rewrite (src/api/mod.rs:4036-4083: scope, generation_runs, readiness-gate
  string, tested_context, next_step) → MANDATORY ledger regen (scripts/check-ledger-drift.mjs
  Check A; CI gate .github/workflows/ci.yml:146-150) → capabilities test → DECISIONS.md entry.
- Fix the live id mismatch: catalog_id "gemma3_1b_it_q8_0" (src/api/mod.rs:20813) vs row id
  "gemma_3_1b_it_q8_0" (src/api/mod.rs:4044) breaks `filename_is_supported_exact_row`
  (src/api/mod.rs:22103-22111) and drift Check C; precedent c2c33fb5 (qwen3-4B).
- Frontend surfaces (absent from the historical checklist): smoke fixtures
  (frontend/scripts/model-state-smoke.mjs:165-178, capability-readiness-smoke.mjs,
  model-lanes-smoke.mjs — all CI-run at .github/workflows/ci.yml:362-406,428), catalog entry
  (frontend/src/lib/supportedModels.js; gemma4 precedent at :89-116), and executionPlan.js backend
  sets if a new backend label is minted. Use real /api/capabilities row ids — the lane-gate
  fixture-drift incident came from fixtures diverging from production ids.
- Docs sweep across README.md, COMPATIBILITY.md, SUPPORT_MATRIX_v0.1.md, STATUS.md,
  CAPABILITY_MATRIX.md, DOCS.md + architecture note, honoring drift Checks D/E (full-sha256
  mentions must state the ledger sha b205840c…; new bundle indexed ONLY in STATUS.md). Do the
  sweep twice (GABBRO needed 1df8c791 as a second pass).
- Validate with `cargo test --all-targets` (integration tests in tests/ are missed by --lib/--bins).
Parity gate: CI green including ledger schema/drift, scrub, frontend smokes.

### Phase 6 — Performance (after all correctness gates; 2-5 days, open-ended)
Deliverables, in order of expected yield:
- Measured baseline of the correct resident lane (decode tok/s at depths 64/512/2048; prefill
  tok/s), plus a same-hardware llama.cpp Metal measurement for a neutral comparison.
- Free win already banked: window masking caps the v1 kernel's serial position walk at 512 for
  20 of 26 layers, so depth scaling improves vs full-causal.
- Token-by-token prefill cost assessment; if prompts dominate, the batched windowed prefill kernel
  with head_dim-256 support (~200-400 MSL + host + tuning) is the single largest kernel item.
- Decode-attention utilization: at 4 heads the v1 dispatch is 4 threadgroups × 32 lanes with each
  lane streaming 1 KB K rows — a 256-dim-capable fast variant (wider staging or dims-per-lane
  restructure of v2/split-K) is real kernel work with self-parity gates; never relax the ≤128 host
  gates without it (they prevent out-of-bounds writes).
Parity gate for every perf commit: byte-identical decode vs the Phase-4 receipts.

Total estimate: ~8-14 working days to a promoted, receipt-backed resident row (Phases 0-5), with
Phase 6 perf work incremental on top.

---

## 4. Expected performance

- **Baseline (measured):** ~0.2 tok/s (~5 s/token), runnable CPU lane, f32 with per-token
  re-dequantization and a row-walked 262k tied head (bundle README; src/runnable/model.rs:776-903).
- **Roofline ceiling:** decode is weight-bandwidth-bound. 999,885,952 params at Q8_0 wire density
  (34 bytes / 32 weights = 1.0625 B/param) ≈ 1.06 GB of weight traffic per token, tied head
  included; KV traffic is negligible beside it at ≤512-window depths. At the M4's ~120 GB/s
  unified-memory bandwidth the ceiling is ≈ 110 tok/s.
- **Reference points:** no in-repo measured Metal number exists for gemma3 (the lane does not run
  it today). The nearest structural precedent is the gemma4 resident lane, which runs the same v1
  attention kernel at head_dim 256/512 in production (src/metal.rs:8410). A same-hardware llama.cpp
  Metal measurement should be taken in Phase 6 as the neutral external reference.
- **Defensible target range: 25-60 tok/s decode** (roughly 125-300x over baseline). The discount
  from the ~110 tok/s roofline reflects: v1 attention underutilization at 4 heads (4 threadgroups
  of 32 lanes, uncoalesced 1 KB K-row streams per lane), per-token encoder overhead at 26 layers,
  and imperfect GEMV bandwidth utilization on the 262k-row head. Anything in this range is
  transformative vs 0.2 tok/s; the upper half likely requires the Phase 6 attention work.
- **Prefill:** token-by-token prefill runs at roughly decode speed, so long prompts will be
  noticeably slower than batched-prefill architectures until the optional head_dim-256 windowed
  flash prefill kernel lands (gap row 8/10).

---

## 5. Risks & landmines

1. **head_dim 256 excludes every fast kernel, and the ≤128 caps are memory-safety-critical.**
   v2/split-K would corrupt memory at 256 (fixed per-lane arrays MAX_DPL=4 src/metal.rs:2000,
   128-float staging :2038, :2118-2119); the host gates (:9501, :9507-9512, :11723-11727,
   :12841-12843) are what prevent it. Mitigation: ship on v1 (proven at 256 by the gemma4 lane),
   never relax gates without new 256-dim variants carrying self-parity tests.
2. **No batched GPU prefill at head_dim 256.** Mitigation: gemma4-style token-by-token prefill for
   correctness (src/gemma4_runtime.rs:4919-4949); treat the batched windowed prefill kernel as a
   scoped Phase 6 item, not a blocker.
3. **Window semantics have no in-repo oracle at ≥512 tokens.** The runnable reference lane has no
   mask at all (src/api/mod.rs:4051), and the off-by-one convention (window includes current
   position, src/model.rs:632-635) is easy to get subtly wrong. Mitigation: llama.cpp ≥512-token
   parity pack as the gate (Phase 4), gemma4 lane as internal cross-check, reuse the exact
   window_start math (src/metal.rs:8452-8453).
4. **Speculative-verify paths hardcode kv_base_offset=0** (src/metal.rs:9215, 9294) even though the
   tree kernels accept the offset. Mitigation: explicit gemma3 spec-decode gate + test in Phase 3
   (head_dim already blocks it, but belt-and-braces against future kernel work).
5. **Hardcoded schedule constants and the attention-scale coincidence.** pattern=6 / local base
   10000 are hardcoded (src/runnable/model.rs:607-626) and 1/sqrt(head_dim) equals gemma3-1B's
   query_pre_attn_scalar only because head_dim=256; larger sizes differ, and 4B+ carry rope scaling
   the loader rejects (src/runnable/model.rs:395-403). Mitigation: parse the GGUF keys in Phase 1;
   pin scope to the 1B row; re-scope explicitly before any multi-size claim.
6. **Promotion-surface landmines (each has bitten before):** execution-plan row gate missing
   (src/execution_plan.rs:975-1021 — resident plan never selected without it); live
   catalog_id/row-id mismatch (src/api/mod.rs:20813 vs :4044, precedent c2c33fb5); mandatory ledger
   regen on readiness-gate string edits (GABBRO fix bb8ee0f2); frontend fixtures must use real
   /api/capabilities ids (lane-gate fixture-drift incident); /v1/completions silently reopens when
   gemma3 leaves the runnable predicate (src/api/mod.rs:7478); docs drift Checks D/E wording rules.
   Mitigation: all are enumerated as explicit Phase 3/5 deliverables above.
7. **GeGLU fusion regression precedent.** The fused gate+up kernel was reverted for register spill.
   Mitigation: separate gate/up GEMVs + `gelu_mul_f32`, matching the gemma4 lane.
8. **Parity-envelope ambiguity.** The existing receipt carries one disclosed 0.3416-nat near-tie
   flip; without a pinned policy, the GPU lane's comparison result is unadjudicable. Mitigation:
   Phase 0 policy decision, recorded before any comparison runs.

---

## 6. Go/No-Go recommendation

**GO**, scoped to the gemma-3-1b-it-Q8_0 row, decode-side first with token-by-token prefill.
Verified kernel-level analysis shows a correct port needs zero new MSL — every gemma3 requirement
(QK-norm, sandwich norms, GeGLU, dual-theta RoPE, kv_base_offset windowing, head_dim-256 v1
attention) already exists in-tree with the gemma4 lane as a production-proven template, leaving
~500-850 LOC of host wiring plus the standard promotion overhead. The payoff is large and
receipt-backed on both ends: a measured 0.2 tok/s baseline, a ~110 tok/s roofline, a reusable
llama.cpp parity harness, and the port simultaneously retires the row's disclosed ≥512-token
correctness blocker by adding the sliding-window mask the CPU lane never had.

---

## 7. Phase 0 record (2026-07-29)

Docs-only gate, executed on this branch at base d9053ec4 with the prebuilt release binary of the
same main commit. Recon model file: the desktop app's `gemma-3-1b-it-Q8_0.gguf`
(1,069,306,368 bytes). No engine code was changed; no server was started (the recon vehicle is
the one-shot `bench-generate` subcommand).

### 7a. Reachability recon results

Mechanism verified before running: `CAMELID_RESIDENT_TRACE` (any value) makes
`resident_decode_eligible` print `[resident-eligible] no: <gate>` on stderr for the first gate
that declines (bail macro, src/inference.rs:2680-2689). The trace is evaluated inside the
library, but the resident lane itself is CLI-armed: `apply_default_fast_stack`
(src/main.rs:5721-5735) sets `CAMELID_METAL_RESIDENT_DECODE=1` (plus wire/NSG8/attn2/prefill/MM
defaults) for every non-deterministic subcommand, so `bench-generate` is a faithful direct-session
probe of the lane.

**Run 1 — default fast stack, trace on** (6-token greedy prompt, 8 generated tokens):

- **Zero eligibility bails fired.** Full trace-relevant stderr, verbatim:
  - `[resident-dispatch] cuda_enabled=false metal_enabled=true` (src/inference.rs:3611-3621)
  - `[resident] pos=5 layers=26 ...` through `[resident] pos=12 layers=26 ...` — the generic
    Metal resident decode lane admitted gemma3-1b and decoded every generated token on the GPU.
- Measured: 38.09 tok/s decode at trivial depth (positions 5-12), peak RSS 0.57 GB (wire pages).
- Output is garbage, as the gap analysis predicts for the mis-bound forward:
  `讖Compliance по bowels по切りごφό` (token ids 251392, 70408, 1311, 143805, 1311, 49874,
  237790, 137586).
- Every gate in `resident_decode_eligible` was walked without firing: session disable
  (src/inference.rs:2690), NoPE (:2700), runnable-tier parity verdict (:2711 — vacuous here: no
  cache key is set on this path, and the GPU-runnable tier is CUDA-only with gemma3 excluded by
  `is_gpu_runnable_arch`, src/execution_plan.rs:837-848), backend-enabled (:2716), MoE (:2719),
  logit_scale (:2722), diagnostic defaults (:2728-2732), the per-layer Q8_0 loop (:2828-2857; all
  26 layers carry wire-page Q8_0 projections, no attention biases), tied output projection
  (:2864-2866), output_norm dim (:2868-2879), and the dims gate (:2883-2892; 1152%32, q_dim
  1024%32, head_dim 256 even, 6912%32, 4%1).

**Run 2 — control, resident decode+prefill forced off** (`CAMELID_METAL_RESIDENT_DECODE=0`,
`CAMELID_METAL_RESIDENT_PREFILL=0`): confirms the trace mechanism works. One disqualifier fired,
15 times (once per prefill/decode/speculative eligibility call), verbatim:

> `[resident-eligible] no: neither CAMELID_METAL_RESIDENT_DECODE nor CAMELID_CUDA_RESIDENT_DECODE enabled`

emitted from src/inference.rs:2716-2717. CPU-lane output is also garbage
(`讖Compliance по切り میر마다 ラя`; ids 251392, 70408, 1311, 49874, 43344, 108003, 37646,
236895) and diverges from the GPU lane at generated index 3 — the wrongness is binder-level and
shared by both lanes, with lane-numeric drift on top of the already-wrong graph. Confirmed
in-source: the dense binder classifies gemma3 as neither `expects_qk_norm` nor `forbids_qk_norm`
(src/model.rs:1008-1009), so `attn_q_norm`/`attn_k_norm` bind `(None, None)` silently and the
post_attention/post_ffw sandwich norms are never requested — the mis-binding disclosed at
src/api/mod.rs:7157-7159 is live, not hypothetical.

**The one resident-side decline that did fire was silent.** Batched GPU prefill declined at
head_dim 256: `try_metal_resident_prefill` (src/inference/metal_resident.rs:67) passes
eligibility (:75), then `prefill_tokens` returns `None` on its guard `self.head_dim > 128`
(src/metal.rs:11714-11731, offending term :11727), and the host returns `Ok(false)` with no
trace line (src/inference/metal_resident.rs:154-159). Evidence: resident decode telemetry starts
at pos=5 (the last prompt token), so positions 0-4 were CPU-prefilled. This is gap rows 8/10
observed live, and it is invisible to `CAMELID_RESIDENT_TRACE`.

**Consequences recorded for the plan (amendments to sections 1-3):**

1. Section 1's "the Metal resident lane is therefore unreachable for gemma3" is true for
   **serve only**. On the CLI direct-session path (bench-generate today; any future non-serve
   session running the default fast stack), reachability is already OPEN: nothing stands between
   gemma3 and a mathematically wrong resident forward, and it decodes silently at speed. The
   serve router divert (src/api/mod.rs:7163-7165) is the only correctness guard in production.
2. Phase 3's "flip/condition the gemma3 disqualifiers in `resident_decode_eligible`" has nothing
   to flip — **no gemma3 disqualifier exists**. The work is inverted: Phase 1 must ADD a
   fail-closed arch-keyed disqualifier (gemma3 declines the resident path until the wiring is
   complete), and Phase 3 removes it in the same commit that lands the last correctness encode.
   This closes the silent-garbage CLI path for the duration of the campaign instead of only at
   its end.
3. The execution plan's fail-closed safe path does not protect this: gemma3 gets the
   "non-validated row or quant" safe plan, but plan `env_updates` never unset the CLI fast-stack
   variables, so the resident lane still engages. Also noted in passing: the startup line
   `[hw] GPU: none detected — CPU backend is the inference path` printed while the Metal lane
   decoded every token; the hardware-probe log line is CUDA-oriented and cosmetically wrong on
   this path (not a Phase 0 work item).
4. Corroborating perf datum: the (incorrect) resident forward at head_dim 256 on the v1 kernels
   already sustains ~38 tok/s at trivial depth on this M4, consistent with the section 4 target
   range (25-60 tok/s) for the corrected lane once the added encodes (QK-norm, sandwich norms,
   GeGLU, dual-theta RoPE, window) take their share.

### 7b. Lane-architecture decision

**DECIDED: extend the generic resident lane.** Concretely: a window parameter on
`encode_attention_block` plus per-layer `(window_start, theta-table)` threading in
`prepare_token`, GeGLU and sandwich post-norm encodes in the generic blocks, forced split-half
RoPE pairing host-side, and the embed scale on the resident gather — per gap rows 3-7/12 — with
gemma4-style token-by-token prefill through the decode path (no new MSL, per gap rows 8/10).

Alternative considered and **rejected for weight**: cloning `Gemma4ResidentModel`
(src/metal.rs:6543-6735) as a standalone gemma3 runtime driver. It would borrow ~1-2k LOC of
structure and leave metal.rs nearly untouched, but it duplicates the generic lane's
session/KV/dispatch machinery for one row, doubles the surface future kernel work must keep in
parity, and forfeits the generic lane's existing self-parity test pattern — the per-feature
deltas on the generic lane are each small, kernel-reusing, and individually gateable.

### 7c. Scope pin

**gemma-3-1b-it-Q8_0 row only.** 4B/12B/27B are explicitly OUT pending a re-scope: the loader
rejects their rope scaling (src/runnable/model.rs:395-403), and 1/sqrt(head_dim) equals gemma3's
query_pre_attn_scalar only at the 1B's head_dim=256 — the larger sizes break that coincidence
(risk 5). No multi-size claim, docs row, or fixture may reference them without a new scoping
pass.

### 7d. Parity-envelope policy

**The GPU lane inherits the existing receipt's envelope.** Token-exact vs the reference is the
target; disclosed near-ties are the tolerance. Specifically: near-tie flips are permitted only
if disclosed in the evidence-bundle README with their measured nat gap (precedent: the runnable
bundle's single position-16 flip at 0.3416 nats); the new ≥512-token windowed pack must be clean
— any flip in it is individually adjudicated before the bundle lands rather than waved through
under the envelope; and no undisclosed divergence of any size is acceptable. This pins the
Phase 4 adjudication rule before any comparison runs (risk 8).

---

## 8. Phase 1 record (2026-07-29)

Landed on this branch after rebasing onto origin/main @ bce31c2c (clean rebase; PR #553 merged
the fail-closed CLI/resident guard with the shared `model::is_runnable_only_arch` predicate and
the new `LlamaModelConfig.architecture` field — that PR IS amendment §7a-2's "Phase 1 must ADD a
fail-closed disqualifier" deliverable, landed ahead of this branch, so Phase 1 here keeps and
tests it rather than re-adding it).

**Config metadata (gap row 17).** New `Gemma3Metadata` on `LlamaModelConfig` (`config.gemma3`,
parsed in `from_gguf` for gemma3 only; src/model.rs): `sliding_window`,
`sliding_window_pattern`, `rope_freq_base_global`, `rope_freq_base_local`,
`layer_is_sliding` (schedule: layer i global iff (i+1) % pattern == 0 — NO forced-global final
layer, unlike gemma4), `embed_scale` = sqrt(d_model), `ffn_geglu`, `rope_neox_pairing`, plus
accessors `is_sliding_layer`/`rope_freq_base_at`/`layer_window` (window INCLUDES the current
position, same convention as `Gemma4LayerPlan::window`). Phase 2 consumes this struct for the
resident encodes.

**Key-name verification finding (deviation from the section-3 sketch).** The real row
(gemma-3-1b-it-Q8_0.gguf, 38 metadata keys dumped raw) carries ONLY two window/rope keys:
`gemma3.attention.sliding_window = 512` (u32) and `gemma3.rope.freq_base = 1e6` (f32). There is
NO sliding-window-pattern key and NO local-freq-base key in the file — no gemma3 conversion
writes them; the reference implementations hardcode pattern 6 / local base 10000 (the same
no-GGUF-key situation as smollm3's `no_rope_layer_step`). Resolution, honoring "no silent
defaults" as far as the file format allows: the two keys that exist are REQUIRED (absent or
malformed → typed parse error; a gemma3 GGUF without `attention.sliding_window` or
`rope.freq_base` no longer loads anywhere, including the runnable lane, which shares
`from_gguf`); pattern and local base are reference-pinned constants
(`Gemma3Metadata::REFERENCE_SLIDING_WINDOW_PATTERN = 6`,
`REFERENCE_LOCAL_ROPE_FREQ_BASE = 10000.0`) disclosed in the struct docs, with explicit
override keys (`gemma3.attention.sliding_window_pattern` scalar,
`gemma3.rope.freq_base_swa`) honored if present and hard-erroring if present-but-malformed —
never a silent fallback over an explicit key. The runnable lane's hardcoded schedule
(src/runnable/model.rs:607-626) is untouched; it remains the CPU reference.

**Dense binder (gap rows 5/6/18).** gemma3 moved from unclassified to `expects_qk_norm`
(alongside qwen3/command-r, with the key_length==value_length gate now arch-labeled), and a new
`expects_post_norms` (gemma3-only) requirement binds `post_attention_norm`/`post_ffw_norm` —
new `Option` fields on `LlamaLayerTensors`, shape-validated `[embedding_length]` as a
must-be-paired set. All 26×4 = 104 norm tensors now bind non-None from the real file, and a
gemma3 row missing ANY of the four fails closed at bind — mis-binding to `(None, None)` is
impossible. `arch_uses_neox_rope_pairing` (and `LlamaModelConfig::rope_neox_pairing`) are
deliberately UNCHANGED for gemma3 per the §7b lane decision: the resident lane forces
split-half pairing host-side in Phase 2 from `Gemma3Metadata.rope_neox_pairing` (gemma4-encode
precedent), leaving the guarded-off CPU dense path unperturbed.

**Safety invariant (unchanged, now co-tested with binding).** `is_runnable_only_arch` still
matches gemma3; the serve divert, the CLI direct-session guard, and the resident-eligibility
arch disqualifier (src/inference.rs, PR #553) are untouched, and PR #553's
`runnable_only_arch_disqualifies_the_resident_gpu_path` still passes. The new binding tests
additionally assert the predicate fires AFTER a successful bind — tensors available, lanes
unreachable. No serve routing, gemma2/qwen35, or Metal encode change (Phase 2/3 scope).

**Parity gate tests** (tests/model_binding.rs; real-row test env-keyed on `CAMELID_GEMMA3_GGUF`
per the `CAMELID_GEMMA4_GGUF` convention, run PASS against the real file):
- `gemma3_real_row_binds_all_104_norm_tensors_and_window_schedule` — (a) 104/104 norms bind
  with real shapes ([256] QK, [1152] sandwich), (b) window 512 / pattern 6 / globals at
  5/11/17/23 / layer 25 local / local 10000 / global 1e6 round-trip, (c) GeGLU + sqrt(1152)
  embed scale + pairing flags set, plus the guard-still-fires assertion.
- `gemma3_binds_qk_and_sandwich_norms_with_window_metadata` (synthetic twin),
  `gemma3_without_qk_norm_fails_closed`, `gemma3_without_sandwich_norms_fails_closed`,
  `gemma3_without_sliding_window_key_fails_closed`,
  `gemma3_explicit_pattern_and_local_base_keys_override_reference_constants`,
  `gemma3_malformed_pattern_key_fails_closed`.
- `model::gemma3_tests::one_b_schedule_globals_at_5_11_17_23_and_no_forced_global_final_layer`
  (unit, schedule/accessor semantics).

Gates: cargo fmt clean, clippy --all-targets -D warnings clean, cargo test --all-targets green
(with the real-row test exercised via CAMELID_GEMMA3_GGUF), check-public-scrub.sh clean.

### 8a. Phase 1b review record (2026-07-29)

Five confirmed adversarial-review findings against the Phase 1 landing, all fixed in one
commit on this branch:

**R1 (major) — swapped sandwich-norm bindings were test-invisible.** `post_attention_norm`
and `post_ffw_norm` are both `[1152]` and every Phase 1 test asserted only `.dimensions`, so
transposing the `find_tensor` lookups passed the whole suite. Fixed by NAME-pinning: the
synthetic binding test and the real-row test now assert the bound descriptor's `.name`
(`blk.{i}.post_attention_norm.weight` on the `post_attention_norm` field, and likewise for
`post_ffw_norm` and — same blindness, both `[256]` — `attn_q_norm`/`attn_k_norm`) on every
layer. Verified by temporarily transposing the lookups (sandwich pair AND QK pair): both the
synthetic and the real-row test fail on the name assertion in each case; restored, both green.

**R2 (major) — schedule derivation was only CI-exercised at block_count=1, and the 26-layer
unit test duplicated the production expression (tautology).** The fixture writer now takes a
`block_count` option (per-block tensors follow it), and two new `from_gguf`-driven tests
assert THROUGH the parsed metadata's accessors with literal expected lists (never the
`(i+1) % pattern` formula): (a) 26 layers, no override → globals exactly at 5/11/17/23,
layer 25 local; (b) 12 layers with an explicit `sliding_window_pattern = 4` override →
globals at 3/7/11, plus a `freq_base_swa = 50000` override reaching `rope_freq_base_at` —
proving the resolved (possibly overridden) pattern, not `REFERENCE_SLIDING_WINDOW_PATTERN`,
drives the derivation. The unit test's hand-built fixture now uses a literal 26-entry
schedule list instead of the formula; it remains the accessor-semantics test.

**R3 (minor, design) — override keys honored by `Gemma3Metadata` but not by the runnable
lane.** CHOICE: single source of truth (the preferred option), not the fail-closed fallback.
The runnable lane's hardcoded pattern-6/local-10000 schedule (src/runnable/model.rs) now
derives `layer_rope_base` from the SAME parsed `Gemma3Metadata` (`cfg.gemma3`, shared
`from_gguf`) via `rope_freq_base_at`, so an override-carrying row can no longer make the
runnable (CPU parity oracle) and resident lanes compute different schedules for one file.
Bit-identity for the real 1B row proven by a new env-gated test
(`runnable::model::gemma3_schedule_tests::gemma3_real_row_runnable_rope_schedule_is_the_reference_schedule`,
literal expected base list + forward-logits fingerprint over a short prompt): fingerprint
`sum_bits=0x0002eec61740012f` identical before and after the rewiring, and the Phase 1
real-row binding/schedule test still passes. Note the runnable lane still implements no
window mask (documented full-support blocker) — R3 unifies the schedule/rope-base inputs,
not the mask.

**R4 (minor) — stale "silently drops the norms" safety comments.** With Phase 1 binding the
norms (and the dense forward path applying QK norms where bound), five comments describing
the pre-Phase-1 binder as present-tense fact were rewritten to the current rationale (binds,
but does not APPLY the sandwich norms; no GeGLU, dual-theta RoPE, or sliding-window mask;
gemma2's sandwich norms still dropped at bind): src/model.rs (`is_runnable_only_arch` doc),
src/inference/tests.rs (resident-disqualifier test doc), src/main.rs
(`ensure_arch_has_direct_dense_session` doc + its bail message), src/api/mod.rs (M-A1 compat
row comment, `is_runnable_serve_arch` doc). A grep sweep caught three more the list missed:
src/inference.rs (resident disqualifier comment + bail message), src/model.rs
(`runnable_only_arch_set_is_exactly_the_serve_bridge_set` comment), src/api/mod.rs
(`completions_unsupported_for_arch` doc).

**R5 (minor) — untested fail-closed branches in `Gemma3Metadata::from_gguf`.** Fixture
options added for each; five new tests assert the typed `InvalidModelMetadata` error (not a
silent reference-constant fallback): `sliding_window == 0`, missing `rope.freq_base`,
non-positive `rope.freq_base`, wrong-typed `rope.freq_base_swa`, non-positive
`rope.freq_base_swa`.

Gates re-run after the fixes: cargo fmt clean, clippy --all-targets -D warnings clean,
cargo test --all-targets green, real-row + runnable-schedule tests green under
CAMELID_GEMMA3_GGUF, check-public-scrub.sh clean.

---

## 9. Phase 2 record (2026-07-30)

Phase 2 landed the correctness-first Metal resident forward for gemma3 in the six sub-steps
sketched in §3, each with its self-parity gate in the same commit (`8b9247d1` 2a QK-norm,
`94ae0263` 2b sandwich post-norms, `55a2e961` 2c GeGLU, `8c476e45` 2d dual-theta RoPE schedule,
`9dad6544` 2e sliding-window decode mask, `462bedec` 2f embed scale), then merged `origin/main`
@ `e28f0f76` underneath them and re-proved the whole stack against the real row. The lane
decision from §7b held: zero new MSL kernels, all host wiring in the generic resident lane.

### 9a. The merge (`origin/main` @ e28f0f76 → the branch)

Merged, not rebased. Five of the six Phase 2 commits touch the same hunks in src/metal.rs, so a
rebase would have replayed the same three-way weave five times against five different
intermediate states; a merge resolves each region once against the final state. Nine conflict
regions in exactly two files (src/metal.rs ×7, src/inference/metal_resident.rs ×2). Main's side
is PR #556 (`ResidentLinearWeight` GEMV dispatch, Q8/F16 primary KV formats, format-dispatched
embed gather, appliance-mode encode-ahead gating) plus PR #557 (prompt-prefix cache).

Three regions have a resolution that **compiles and is silently wrong**, and they are the reason
this is recorded rather than left in the commit message:

1. **FFN f32y GEMV.** Keeping the campaign's `encode_q8_matmul_f32y` and satisfying the type
   checker with `&gate_w.buffer` compiles and pushes Q4_K/Q6_K FFN weights through the Q8_0
   GEMV — garbage on exactly the K-quant rows #556 exists to serve. Resolution takes main's
   `encode_resident_matmul_f32` call *shape* and appends the campaign's GeGLU/SiLU
   `act_pipeline` binding.
2. **Attention scalar byte 28 (`kv_base_offset`).** After #556 the shared encode computes a
   conditional `kv_position_stride` — BYTES on the Q8 primary, elements on f32/f16 — and
   `kv_base_offset` shares those units. Neither side's text is acceptable: main pins byte 28 to
   0, which does not merely revert to full-causal (the caller still passes the narrowed
   `position_count`, so the kernel reads the OLDEST rows and never the current position); the
   campaign reverts bytes 20/24 to element strides, breaking the Q8 primary KV lane for every
   `head_dim <= 128` row. The correct weave is `window_start * kv_position_stride`.
3. **`prepare_token` gather scalar.** The 8-byte allocation and the shader's `buffer(4)` read
   auto-merge from the campaign, and `pool_get` classes by `bytes.max(32).next_power_of_two()`
   and never zeroes, so main's format-derived bytes-per-row alone leaves bytes 4..8 unwritten
   while the kernel reads them — every GPU-sampled token's embedding multiplied by a recycled
   stale float, on ALL resident rows, not just gemma3. The weave writes both fields.

Two more were loud-but-easy to get wrong: `ResidentDecodeState::new` needed BOTH prologues
(main's text alone silently deletes the fail-closed schedule-length check, turning a clean
`None` into an out-of-bounds panic on the first decode token), and the decode encode-ahead
tables had to be NESTED inside `resident_encode_ahead_enabled` rather than replaced (taking the
campaign hunk wholesale compiles and reinstates unconditional encode-ahead, undoing #556's
cooperative-batching head-of-line-blocking fix).

**Hardening taken while the attention region was open (was §5 landmine material, now closed):**
`encode_attention_block` no longer re-reads the process-global KV-format gates. It takes the
session's `kv16`/`kvq8` as parameters, because the call site, the KV readback and the KV seed
already use the per-session fields and the window offset now rides in the same scalar. The three
standalone (non-session) helpers pass the globals, so their behaviour is byte-identical. Residual,
recorded: the inner `encode_attention` helper still selects its pipeline from the globals — it is
shared with the gemma4 lane and the speculative-verify path, so threading it is a separate change.

### 9b. Phase 2 final gate — real-row parity

`gemma3_real_row_resident_forward_matches_runnable_oracle` drives the resident machinery directly
(the production arch disqualifier stays up until Phase 3) with every Phase 2 encode live, and
requires a token-identical greedy continuation to the runnable lane — the CPU oracle pinned to HF
transformers by qa/runnable/gemma3-parity.json. Run targeted and in `--release` with the
production GEMV configuration (f32y + wire + NSG8), because those gates are process-latched and a
full `--lib` run silently SKIPs the test:

```
CAMELID_METAL_F32Y=1 CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 \
CAMELID_GEMMA3_GGUF=<gemma-3-1b-it-Q8_0.gguf> \
  cargo test --release --lib gemma3_real_row_resident_forward -- --nocapture
```

Measured on the M4 mini against the real `gemma-3-1b-it-Q8_0` row, no SKIP line, 590.94 s:

| depth | resident argmax | oracle argmax | max abs logit diff |
|---|---|---|---|
| 1 | 108 | 108 | 6.247e-5 |
| 5 | 1077 | 1077 | 7.820e-5 |
| 50 | 578 | 578 | 9.584e-5 |

**PASS: 50/50 greedy tokens identical; overall max abs logit diff 2.122e-4.** Zero flips, so the
§7d envelope is not drawn on at all — this receipt is clean, not disclosed-flip. Per §5 landmine
below, the slot count matters: this run is the `active_slots <= 1` equivalent with encode-ahead
OFF (the test passes `next_rope: None`), and 9c's MR2 regression proves encode-ahead ON is
bit-identical to it.

The gate asserts `total < 512`, so the whole comparison sits inside the window and it cannot
distinguish a correct window base from one pinned to 0. That is 9c's job.

**Landmine found while re-applying this test: it must not arm its own gates.** As written it
opened with `std::env::set_var("CAMELID_METAL_F32Y"/"..._WIRE"/"..._WIRE_NSG8", "1")` *before*
its own SKIP checks, so it mutated the process environment on every run, including a plain
`cargo test` with no gemma3 GGUF present. Those gates are process-wide `OnceLock`s read by every
other Metal test in the binary; whichever siblings had not read them yet then latched onto the
wire path, where the standalone block helpers' 36-byte uploads are read as 34-byte wire blocks
and come back NaN. Measured: `cargo test --lib metal::tests` is green twice at the merge commit
and fails five resident/standalone tests with the gate test added; a full `cargo test
--all-targets` failed those five plus `metal_gemma4_layer_matches_cpu`. Whether a given sibling
is hit is a race, so an earlier full run happening to pass proves nothing. The test now checks
`CAMELID_GEMMA3_GGUF` first, never sets the gates, and SKIPs with the full armed invocation in
its message. **Three pre-existing tests on main still use the in-test `set_var` pattern**
(`metal_verify_gemv_batched_bit_identical`, `metal_spec_verify_bit_identical`,
`metal_tree_verify_bit_identical`) — same hazard, untouched here, worth the same treatment.

### 9c. Post-merge regressions added

- `metal_resident_window_start_beyond_512_matches_seeded_window_oracle` — pins a windowed decode
  at `filled` 256/512/513/561/600 (the row's real window of 512, `window_start` up to 88)
  bit-for-bit against a full-causal oracle seeded with exactly the window's rows. head_dim 64 is
  deliberate: it admits the v2/split-K attention geometry the 1B's head_dim 256 can never reach,
  and it makes the Q8 primary KV reachable, where `kv_base_offset` is a byte offset. Verified
  sensitive in both directions by temporarily breaking the packed word — byte 28 pinned to 0
  fails this test AND the Phase 2e self-parity; element units fail this test under
  `CAMELID_METAL_KV_DTYPE=q8`. Green on all three primaries (f32 default, q8, f16).
  The history is SEEDED rather than decoded, and that is load-bearing: the first cut walked 600
  real tokens and destabilised the whole suite. `MetalLinearKernel` owns ONE shared serial
  command queue (the `Drop` impl for `ResidentDecodeState` already warns that a gated pending
  graph "would block every future commit on the shared serial queue"), so a test holding it for
  hundreds of gated command buffers starves the others — observed as unrelated one-dispatch
  kernel tests (`metal_rms_norm_matches_cpu`, `metal_silu_mul_matches_cpu`,
  `metal_rope_rotate_matches_reference`, `metal_soft_cap_matches_cpu`, `metal_residual_add…`)
  returning their untouched input, and as NaN in the resident/standalone comparisons. Two
  command buffers per depth is the same proof at 1/300th of the occupancy (0.34 s vs 10 s).
  **Rule for future Metal tests on this lane: budget command buffers, not wall time.**
- `metal_resident_gemma3_decode_is_identical_with_encode_ahead_off` — 12/12 tokens bit-identical
  on a gemma3-shaped session (dual theta + sliding window + QK/sandwich norms) with the next
  token's tables supplied and withheld. This is the claim that makes the appliance-mode
  `(None, None)` arm safe, and it is the coverage the Phase 2 gate cannot give (that test is
  already the encode-ahead-off configuration). Note this is the **only** in-suite exercise of the
  encode-ahead pipeline: before it, every `forward_token` call in `mod tests` passed
  `next_rope: None`. The two configurations run SEQUENTIALLY, each session dropped before the
  next starts; the first cut interleaved them and deterministically broke the five gemma3
  self-parity tests plus `metal_gemma4_layer_matches_cpu` — a pre-encoded graph is committed and
  gated, so with two live sessions the second one's work (and every concurrent Metal test's)
  queues behind a command buffer that only unblocks on the next loop iteration.
- `metal_kquant_embed_gather_drops_embed_scale_so_gpu_sampling_fails_closed` — proves on the
  device that binding `buffer(4)` on `embed_row_gather_q4k` is legal and INERT, and pins the new
  host fail-closed (`gpu_sampling_tail_is_scale_safe`) that refuses the device-side sampling tail
  when a non-unit `embed_scale` meets a non-Q8_0 embedding table. Note the production caller
  already requires a Q8_0 token embedding before it builds the stage at all, so this is
  defence-in-depth at the enforcement point rather than a live bug fix.
- `resident_session_construction_sets_the_kquant_lane_at_both_sites` — `ResidentDecodeState::new`
  reads the global that `set_resident_kquant_lane` writes, and both call sites sit in
  merge-conflicting hunks; dropping either is silent (an F32 primary where F16 was intended, no
  failing assertion anywhere). A source-level count is crude but it is the only thing that fails
  when a merge quietly deletes one.

### 9d. Gates

cargo fmt clean; `clippy --all-targets -D warnings` clean (load-bearing here: it is what turns a
dead `window_start` parameter into a red build — do not silence it by underscoring the
parameter); `cargo test --all-targets` green twice in a row (1734 passed / 0 failed, against
1730/0 at the commit before these four tests); `cargo test --lib metal::tests` green three times in a row
(the filtered run is the sensitive one for queue starvation); the real-row gate green with the
numbers in 9b; check-public-scrub.sh clean.

**Standing note for anyone adding Metal tests here.** The three failure modes hit in this pass
all came from process-wide or device-wide sharing, never from the maths: (1) a test that sets a
gate env var latches sibling tests onto a different kernel; (2) a test that holds the single
shared serial command queue for hundreds of command buffers starves siblings until they read
back unwritten buffers; (3) two live resident sessions with encode-ahead park a gated command
buffer in front of each other. All three present as "unrelated test returns 0 or NaN", and (2)
and (3) look like flakes until you re-run the filtered subset. Budget command buffers, keep one
resident session live at a time, and arm gates from the shell.

### 9e. Amendments to sections 3-6 (recorded, not yet applied to those sections)

1. **§3 Phase 3 — a fourth eligibility surface.** Main added `is_gpu_runnable_arch`
   (src/execution_plan.rs), an allow-list of `llama | qwen2 | qwen3 | mistral` consumed by the Q8
   GPU-runnable tier and by K-quant plan selection, whose comment explicitly names gemma3 as a
   mirror of `resident_decode_eligible`. Without it the Metal-resident K-quant plan is never
   advertised and the Q8 GPU-runnable tier stays closed. Add it beside
   `resident_decode_eligible`, `recognized_row_level`/`is_supported_exact_q8_row` and
   `is_runnable_serve_arch`. The §3/§5 line citations for all of these are stale after main's
   +2775 lines in src/metal.rs alone — re-derive rather than trusting them.
2. **§3 Phase 3 — the prompt-prefix cache is a first-class blocker.** PR #557 did not exist when
   Phase 3 was scoped. On a non-exact cache hit the resume path rolls back to `k` and re-prefills
   the divergent suffix at `kv_position = k > 0`; the only GPU-prefill hook refuses any non-zero
   start, so the suffix is evaluated by the CPU dense forward — which has none of gemma3's
   structure and no window at all. Partial hits are admitted from 16 tokens. The failing case is
   ordinary multi-turn chat. The bypass must be a new explicit windowed-arch predicate at the
   lookup sites and the store site, NOT inside `try_metal_resident_prefill` (unreachable at
   position > 0) and NOT `CAMELID_PREFIX_CACHE_RESIDENT=0` (`prepare_for_prompt_prefix_cache`
   returns before consulting it). Related: the campaign's "token-by-token prefill through the
   decode path" bullet needs a location — it must be at the session level in
   `generate_next_token_with_history_diagnostics`, forcing a single-token prefill chunk.
3. **§3 Phase 3 — pin the flip to the Q8_0 row in the mechanism, not only the risk register.**
   The disqualifier is arch-keyed (`matches!(architecture, "gemma2" | "gemma3")`), and main has
   since opened the Metal resident lane to Q4_K/Q6_K weights. A gemma3 Q4_K_M GGUF would reach
   the resident lane, take an F16 primary, and activate the whole mirror/store/partial-hit path.
   Scope the flip to the Q8_0 exact row, or exclude gemma3 from the K-quant Metal admission,
   until a windowed K-quant lane has its own receipt.
4. **§3 Phase 2 — the "two-line gemma4 math" is no longer two lines.** After #556 the shared
   `encode_attention_block` computes `kv_position_stride` conditionally, so the window base is
   `window_start * kv_position_stride`. The gemma4 lane keeps element units legitimately (f32 KV
   only) and is no longer a copyable precedent for the shared encode.
5. **§4 / §6 — state the prefix-cache exclusion with any throughput claim.** On the Q8_0 row
   gemma3 gets ZERO prompt-prefix reuse: `prepare_for_prompt_prefix_cache` requires
   `kv_roundtrips_through_cpu_exactly()`, which is literally the F16-primary flag. Multi-turn
   chat therefore pays a full token-by-token prefill every turn. The §4 "0.2 tok/s baseline /
   ~110 tok/s roofline" framing predates #557 and must carry this caveat.
6. **§5 — two new landmines.** (a) The process-global vs per-session KV-format split inside the
   shared attention encode; closed for `encode_attention_block` in 9a, still open for the inner
   `encode_attention`. (b) Appliance mode drops encode-ahead at 2+ active slots, which makes
   `SampleStage.embed_scale` dormant — the sqrt(d_model) scale is applied twice by design (CPU
   input and GPU gather) and the two paths are never both exercised in one run. Every gemma3
   parity receipt must state its slot count; the Phase 4 bundle should carry both.
   Also: a window-aware KV mirror is tempting (25 of 26 layers can only read the trailing 512
   positions, yet the mirror copies `[0, position)` per layer — ~53 KiB/position, ~3.4 GB at the
   row's 32,768-token context) but it changes the round-trip exactness argument and must not be
   attempted before the Phase 3 blockers.
7. **§3 Phase 4 — the ≥512-token receipt has a second job.** Because the Phase 2 gate asserts
   `total < 512`, the windowed pack is the only external artifact that can distinguish a correct
   window base from a zeroed one. It is merge-correctness evidence, not only a context claim.

### 9f. Phase 3 is NOT open

The blockers in 9e-2/9e-3 (prompt-prefix-cache routing, arch-vs-quantization scope) plus the
requirement that an explicit gemma3 fail-closed on the CPU dense fallback lands in the SAME
commit that removes `is_runnable_only_arch` / the `resident_decode_eligible` disqualifier are
gates on Phase 3, not Phase 3 work. Until they are closed, the ≥512-token correctness claim this
campaign exists to retire can be re-broken by any fallback, and the serve router divert remains
the only production correctness guard.

## 10. Phase 3a record (2026-07-30)

Phase 3a closed the three blocking hazards from §9e-2/§9f so the Phase 3b routing flip
(removing the arch disqualifiers) becomes a safety-neutral change. gemma3 stayed FAIL-CLOSED
throughout: nothing here changes production routing for any arch — this is the safety plumbing
the flip will stand on. Three commits, one per hazard, each gated by
fmt / clippy --all-targets -D warnings / cargo test --all-targets.

### 10a. H1 — prompt-prefix cache bypass for windowed archs

New predicate `crate::model::arch_has_windowed_attention(&LlamaModelConfig)`
(src/model.rs:167, beside `is_runnable_only_arch` at :150): keyed on the PARSED metadata
(`config.gemma3.is_some()`), not the arch string, so gemma3-4B and any future windowed arch
inherit every guard that consults it.

Enforced at the three prompt-prefix-cache decision sites in src/api/mod.rs:

- STORE (`store_prompt_prefix_cache`, :13053): a windowed arch never stores an entry — checked
  before the position check and before `prepare_for_prompt_prefix_cache`, so no mirror cost is
  ever paid on the refusal.
- Both PARTIAL-RESUME sites now share one decision point, `resume_partial_prefix_hit` (:13116,
  extracted so the non-streaming handler and `stream_prompt_cache_prologue` cannot drift),
  which refuses a windowed arch: the divergent suffix would be re-prefilled at
  `kv_position > 0`, the resident prefill hook refuses any non-zero start, and the CPU dense
  forward has no window — the H1 failing case is ordinary multi-turn chat. Declining costs one
  cold full prefill: slower, never wrong.
- EXACT hits stay allowed: no forward runs on that path, and with the store site refusing, no
  windowed entry can exist outside a stale pool — which the resume guard also covers.

Tests: `windowed_arch_never_stores_a_prompt_prefix_entry` (store site; a non-windowed control
run proves the bypass is the thing that fired), `windowed_arch_never_takes_a_partial_prefix_resume`
(the shared resume decision point, with control), and
`stream_prologue_windowed_arch_partial_hit_falls_back_to_cold_prefill` (the streaming site
end-to-end through `CooperativeStreamDecodeJob::new` against a hand-inserted stale entry).

### 10b. H2 — session-level token-by-token prefill for windowed archs

`session_prefill_chunk_tokens(config, prefill_count)` (src/inference.rs:5510) is now the
prefill routing decision consumed by `generate_next_token_with_history_diagnostics`: a windowed
arch forces the single-token lane (chunk = 1), so every prompt token flows through
`forward_single_token_timed_internal` → `try_resident_decode_forward` — the only lane whose
forward carries the sliding-window / dual-theta schedule once the arch is admitted (the gemma4
runtime's token-by-token prefill is the semantic precedent, per §9e-2). Every other arch keeps
`prefill_chunk_token_count` verbatim — byte-identical routing, pinned by
`non_windowed_arch_prefill_chunking_is_byte_identical` next to
`windowed_arch_prefill_forces_the_single_token_lane`.

The production arch disqualifier (src/inference.rs:2779) stays up. A cfg(test)-only seam,
`TEST_ADMIT_WINDOWED_ARCH_RESIDENT` (src/inference.rs:12303, compiled out of production builds
entirely), admits the arch for the duration of one targeted test so the routing could be proven
BEFORE the flip: `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle`
(src/metal.rs) drives the PRODUCTION session entry over the real 1B row with a multi-token
prompt. Measured (M4 mini, release, f32y+wire+NSG8 armed, CAMELID_METAL_RESIDENT_DECODE=1):

| depth | session argmax | oracle argmax | max abs logit diff |
|---|---|---|---|
| 1 | 108 | 108 | 6.247e-5 |
| 2 | 584 | 584 | 6.676e-5 |
| 3 | 568 | 568 | 5.627e-5 |
| 4 | 2364 | 2364 | 5.913e-5 |
| 5 | 1077 | 1077 | 7.820e-5 |

5/5 greedy tokens identical, overall max abs logit diff 7.820e-5; depth 1 matches the Phase 2
gate bit-for-bit-in-report (108 / 6.247e-5) — same forward, now reached through the session.
The routing itself is pinned by `!session.cpu_kv_authoritative()` at the end: a CPU dense
prefill of any flavor materializes the CPU KV as it goes; the resident lane leaves it hollow.

### 10c. H3 — the cache kill switch is real (arch-independent live-main bug)

`prepare_for_prompt_prefix_cache` returned `true` for a CPU-authoritative session BEFORE
consulting `CAMELID_PREFIX_CACHE_RESIDENT`, so the documented opt-out did nothing on any
CPU-authoritative session — which today is every windowed-arch session (H2) and the entire
ordinary CPU lane. The gate is now consulted FIRST
(`prepare_for_prompt_prefix_cache_gated`, src/inference/metal_resident.rs:428): with the
variable set to `0`/`false`, preparation refuses every session, making the variable a real
kill switch for cache storage (`store_prompt_prefix_cache` refuses on `false`).

Tests: `prompt_prefix_cache_preparation_env_opt_out_is_a_kill_switch` drives the parameterized
seam on a session that caches under `true`;
`prefix_cache_env_setting_parses_the_documented_opt_out` covers the env-value parse
(`prefix_cache_setting_enables`, split pure from the OnceLock). Deliberately NOT an in-test
`set_var`: the gate is a process-wide OnceLock and §9d's standing note applies — gates are
armed from the shell, never latched from inside a test.

### 10d. Gates

Per commit: cargo fmt clean; clippy --all-targets -D warnings clean; cargo test --all-targets
green (H1 commit: pipeline exit 0 under pipefail, lib suite 1387 tests started, no failures —
the tally lines were lost to an output filter; H2 commit: 1367 lib passed / 0 failed / 23
ignored plus every integration suite green, exit 0; H3 commit: recorded below in this section's
final battery). check-public-scrub.sh clean. Env-keyed battery at phase end (release,
production GEMV configuration armed from the shell): the H2 session-level gate above, and the
Phase 2 real-row final gate re-run:

H3-commit full battery: cargo test --all-targets exit 0, lib 1369 passed / 0 failed / 23
ignored, every integration suite green (60 green tallies). Phase 2 real-row final gate re-run
(release, no SKIP, 570.08 s): depth 1 argmax 108 = oracle, max abs logit diff 6.247e-5; depth 5
argmax 1077 = oracle, 7.820e-5; depth 50 argmax 578 = oracle, 9.584e-5 — 50/50 greedy tokens
identical, overall max abs logit diff 2.122e-4, bit-for-bit the §9b record. The whole Phase 2
stack is therefore proven UNCHANGED under the Phase 3a plumbing.

### 10e. What Phase 3b still owes (restated against current line numbers)

- The flip commit must remove/condition BOTH `is_runnable_only_arch` (src/model.rs:150) and
  the `resident_decode_eligible` arch disqualifier (src/inference.rs:2779) AND land the
  explicit gemma3 fail-closed on the CPU dense fallback in the SAME commit (H4, §9f).
- Scope the flip to the Q8_0 exact row in the MECHANISM (H5): main's Metal K-quant admission
  (`is_resident_quant` / `metal_only`, src/inference.rs:2894-2903) would otherwise admit a
  gemma3 Q4_K_M to an F16-primary resident lane whose K-quant gather drops `embed_scale` (the
  H6 fail-closed covers only the GPU sampling tail).
- Eligibility surfaces, current locations: `is_gpu_runnable_arch` (src/execution_plan.rs:863;
  consumed at :353 Q8 GPU-runnable tier and :375 K-quant plan selection),
  `recognized_row_level` (src/execution_plan.rs:1027) / `is_supported_exact_q8_row` (:1066),
  `is_runnable_serve_arch` (src/api/mod.rs:7337, a delegate to `is_runnable_only_arch`).
- NEW since the §3 checklist was written (the #549/#554 merges):
  - `prepare_generation` (src/api/mod.rs:11523) carries the raw-completions choke-point gate
    (`completions_unsupported_for_arch` at :11587, delegating to `is_runnable_serve_arch`)
    covering `/completion`, `/v1/completions` n>1 fan-out, `/api/generation/preflight`,
    `/api/generation/sessions`, and receipt replay; `/v1/completions` itself also gates via
    `reject_completions_for_runnable_arch` (:7684, applied at :10065). Flipping
    `is_runnable_only_arch` membership REOPENS all of these for gemma3 in the same motion —
    the flip commit must decide whether that is intended and cover it with tests (the #554
    test module `runnable_completions_gate_api_tests` pins today's behavior).
  - `/v1/responses` delegates to `chat_completions` (src/api/responses.rs:169) and so inherits
    whatever chat routing the flip leaves behind.
  - The runnable serve lane's tools threading (`runnable_request_tools`, src/api/mod.rs:14066,
    consumed by `runnable_chat_nonstreaming` :7731 and `runnable_chat_streaming` :7854) serves
    gemma3 chat today and goes dormant for gemma3 when the arch leaves
    `is_runnable_serve_arch` — tool-calling parity on the dense lane is NOT covered by any
    existing gemma3 test.
- H1's bypass keys on `arch_has_windowed_attention`, which is INDEPENDENT of the runnable
  predicates: the flip does not reopen the prompt-prefix cache for gemma3. Reopening it later
  is §9e-5/H11 territory (F32 primary never qualifies; a window-aware mirror changes the
  round-trip exactness argument) and stays out of 3b.

## 11. Phase 3b record (2026-07-30)

Phase 3b is the routing flip: gemma3 is servable on the Metal GPU-resident lane. One commit
(ba6de7f7), standing entirely on the 3a plumbing; nothing in it touches kernels or the forward.

### 11a. The capability-aware predicate

The flip is NOT a bare list edit. gemma3 left `is_runnable_only_arch` (src/model.rs, now
`qwen35 | gemma2` only), and routing keys on a new pair beside it:

- `model::arch_requires_runnable_bridge(arch)` — the live predicate serve and the CLI direct
  lanes consult. True for qwen35/gemma2 always; true for gemma3 only where the resident lane
  cannot serve it.
- `model::arch_requires_runnable_bridge_given(arch, capable)` — the pure half, so the split is
  unit-testable without env or a device.
- `inference::windowed_arch_resident_host_available()` — the host probe:
  macOS build AND `resident_decode_metal_enabled()` (live env; deterministic mode force-off)
  AND NOT `resident_decode_cuda_enabled()` (the CUDA engine has no windowed forward)
  AND a real Metal device (`detect_metal_device().available`, cached in a OnceLock).

Consumers rewired: `api::is_runnable_serve_arch` (serve router + runnable-runtime load +
`completions_unsupported_for_arch`) and `main::ensure_arch_has_direct_dense_session` both
delegate to `arch_requires_runnable_bridge`, so serve and the CLI cannot disagree. Outcome:
on a resident-capable host gemma3 chat falls through the runnable short-circuit onto the dense
lane and the resident engine serves it; on every other host (non-macOS CI legs, resident decode
opted out, deterministic mode, CUDA-resident, no device) gemma3 loads the runnable runtime and
serves exactly as before the flip — never the CPU dense forward.

### 11b. H4 — the CPU dense forward fails closed for windowed archs (same commit)

`LlamaInferenceSession::ensure_windowed_arch_off_cpu_dense` (src/inference.rs), keyed on
`arch_has_windowed_attention`, returns a typed `BackendError::UnsupportedModelArchitecture`
naming the hazard and both correct lanes. Guarded at ALL THREE CPU dense forward dispatches:
the single-token decode fallback (the else-branch after `try_resident_decode_forward`
declines), `forward_prefill_chunk_timed_fast`, and
`forward_prefill_layer_major_timed_fast_inner`. No routing mistake can silently run gemma3
full-causal. A second cfg(test)-only seam (`TEST_ADMIT_WINDOWED_ARCH_CPU_DENSE`, drop-guarded,
armed only under `env_lock`) lets the 3a prompt-prefix-cache decision tests keep driving tiny
synthetic gemma3 configs through the CPU forward mechanically; the 3a resident seam is
unchanged and now effectively covers only gemma2. Pinned by
`windowed_arch_cpu_dense_forward_fails_closed` (with a non-windowed control proving causality).

### 11c. H5 — resident admission pinned to the Q8_0 exact row (same commit)

In `resident_decode_eligible`: the arch disqualifier is now gemma2-only; windowed archs gained
(a) a CUDA-resident bail (the CUDA engine would run the window full-causal) and (b) a Q8_0 pin
— `is_resident_quant` returns `is_q8` only for windowed archs, plus an explicit pre-loop typed
decline when any layer linear is non-Q8_0 (a gemma3 Q4_K_M would otherwise ride the Metal
K-quant admission onto an F16-primary lane whose gather drops `embed_scale`, with no windowed
receipt). Serve falls back to the runnable bridge for such files.

### 11d. Execution plan

- `recognized_row_level`: gemma-3-1b-it row added at a NEW honest level string
  `supported_exact_row_smoke_sub512` (the ≥512 receipt is Phase 4's), included in
  `is_supported_exact_q8_row`; `support_level` already gates it to Q8_0 files only.
- Plan selection is platform-split for windowed archs: macOS+Metal+resident →
  `metal_resident_q8_runtime` (the load-bearing selection §3 called out); anywhere the Metal
  selection cannot fire (non-macOS, resident unset, `CAMELID_MAC_Q8_METAL_PLAN=0`) the plan
  FAILS CLOSED to `safe_q8_plan` with a windowed-arch reason instead of advertising a CPU
  dense lane H4 forbids (`select_macos_q8_plan` gained a `windowed_attention_arch` param;
  the x86 arm is bypassed via `is_windowed_attention_arch`).
- `is_gpu_runnable_arch`: gemma3 deliberately NOT added. Both consumers are non-Q8-exact
  tiers H5 forbids — the Q8 GPU-runnable tier is CUDA-resident (no windowed CUDA forward) and
  the K-quant plan selection would advertise the Metal K-quant lane. Decision recorded in the
  function comment; pinned by `gemma3_kquant_never_takes_the_metal_resident_kquant_plan`.
- New plan tests: `gemma3_q8_row_selects_metal_resident_plan_on_a_resident_mac`,
  `gemma3_q8_row_fails_closed_to_safe_plan_where_metal_resident_cannot_run`, and the K-quant
  pin above.

### 11e. Raw-completions surfaces (#554) and the dense chat renderer

The #554 chokepoints (`prepare_generation` dense gate, `reject_completions_for_runnable_arch`)
key on the capability-aware predicate, so gemma3's raw surfaces (`/completion`,
`/v1/completions` + n>1 fan-out, `/api/generation/preflight`, `/api/generation/sessions`,
receipt replay) REOPEN exactly where the resident lane serves and stay 422-gated on the
runnable fallback. `api::runnable_completions_gate_api_tests` pins the split: the always-gated
tests moved to qwen35/gemma2, and two new tests pin gemma3 both ways
(`completions_gate_stays_closed_for_gemma3_on_a_runnable_fallback_host` — env=0 under
env_lock, restores the caller's value; `completions_gate_reopens_for_gemma3_where_the_
resident_lane_serves` — macOS+device gated).

The dense chat lane gained gemma3's prompt renderer: without it the fallback renderer dropped
gemma3 chats onto the role-colon prompt. `is_gemma3_chat_template` (`<start_of_turn>` +
`<end_of_turn>` + `first_user_prefix`) routes to the SAME byte-faithful `render_gemma3_prompt`
the runnable bridge uses, with the identical encode contract (no BOS in the string,
add_special=true, parse_special=true). Pinned by
`gemma3_template_renders_through_the_shared_gemma3_renderer_on_the_dense_lane`.

### 11f. Tool calling (#549) — decision

gemma3 tool calling has never been supported on ANY lane: the row is `tool_capable: false` and
the runnable bridge returns a typed 422 (`unsupported_tools` — "no tools branch, no certified
grammar") by design. "Fixing dense-lane tool threading" would mean inventing an uncertified
tool grammar for a template that has none — the opposite of this repo's fail-closed policy.
The flip therefore PRESERVES the explicit refusal contract on the dense lane:
`render_chat_prompt_for_tokenization_with_tools` declines gemma3's template with the same
row-accurate reason (surfaced as 422 `unsupported_chat_template`), pinned by test. Tools are
never silently dropped from the prompt, and `tool_choice:"none"` still renders plain chat
(verified live). This is behavior-IDENTICAL to pre-flip from the API user's perspective.

### 11g. Serve smokes (release, this M4 mini 16 GB, no special env vars)

Resident smoke (`camelid serve --model gemma-3-1b-it-Q8_0.gguf --no-open`):
- /v1/health: `generation_ready:true`, `selected_backend:"metal_resident_q8_runtime"`,
  `decode_path:"q8_0_metal_resident_decode"`, `support_level:"supported_exact_row_smoke_sub512"`,
  backend `"llama"` (dense serve lane; NO runnable runtime loaded).
- Greedy chat ("Why is the sky blue? Answer in one sentence.", 20 prompt tok): coherent
  Rayleigh-scattering sentence, finish stop, 26 tokens. Run 1 wall 0.825 s, run 2 (warm)
  0.788 s — byte-identical token ids across runs. Warm timings: prefill 19 tok / 302.2 ms
  = 62.9 tok/s (token-by-token per H2), first token 40.4 ms, decode 25 tok / 389.4 ms
  = **64.2 tok/s decode** at short depth.
- Long greedy (33 prompt / 256 completion, two runs, byte-identical): prefill 81.5 / 86.1
  tok/s; decode 255 tok in 5662.0 / 5543.3 ms = **45.0 / 46.0 tok/s decode** at depth ~289.
  Within the §4 25-60 target band; ~0.4-0.6x of the ~110 tok/s roofline.
- Oracle check: first 8 greedy token ids [818, 7217, 7412, 3730, 1547, 529, 496, 20284]
  IDENTICAL 8/8 to the runnable oracle's (same prompt, fallback server below). No envelope
  flip needed.
- Tools (tools + tool_choice auto): typed 422 `unsupported_chat_template` — "the gemma3 chat
  template has no tools branch and no tool-call grammar is certified for this row; tool
  requests fail closed on the dense lane exactly as on the runnable bridge" (§11f).
- Raw `/v1/completions` REOPENED: "The capital of France is" → " Paris.\n\nThe largest city
  in France" (200).
- Response `lane` discloses `"experimental"`: `filename_is_supported_exact_row` still fails on
  the catalog/row id mismatch (`gemma3_1b_it_q8_0` vs `gemma_3_1b_it_q8_0`) — the KNOWN
  Phase 5 deliverable (§3 Phase 5, precedent c2c33fb5), pre-existing, not new breakage.

Fallback smoke (`CAMELID_METAL_RESIDENT_DECODE=0`, same command):
- /v1/health: backend `"runnable-runtime"`; plan fails closed to `cpu_reference` /
  `safe_cpu_decode` with the windowed-arch reason string.
- Greedy chat 8 tok: 27.07 s wall (the known ~0.2-0.3 tok/s runnable lane), token ids above.
- `/v1/completions`: 422 `unsupported_completions_lane` (gate intact verbatim).
- Tools: 422 `unsupported_tools` (runnable lane refusal unchanged).
Both servers killed by saved PID only.

### 11h. Gates

Flip commit: cargo fmt clean; clippy --all-targets -D warnings clean; cargo test --all-targets
exit 0 under pipefail (lib 1376 passed / 0 failed / 23 ignored; every integration suite green).
check-public-scrub.sh clean; scripts/check-ledger-drift.mjs passed (no capability-row or
readiness-gate string touched — the row rewrite is Phase 5's). Env-keyed battery (release,
production GEMV gates + resident decode armed from the shell, targeted names so the
env-mutating gate tests never run alongside):
- Phase 2 real-row final gate re-run: depth 1 argmax 108 = oracle, max abs logit diff
  6.247e-5; depth 5 argmax 1077 = oracle, 7.820e-5; depth 50 argmax 578 = oracle, 9.584e-5 —
  50/50 greedy tokens identical, overall max abs logit diff 2.122e-4, 564.04 s. Bit-for-bit
  the §9b/§10d record: the whole Phase 2 stack is UNCHANGED under the flip.
- 3a session-level prefill gate re-run: 5/5 greedy identical (108/584/568/2364/1077),
  overall max abs logit diff 7.820e-5, 21.92 s — bit-for-bit the §10b record.
- `gemma3_real_row_runnable_rope_schedule_is_the_reference_schedule` (release): PASS,
  fingerprint sum_bits 0x0002eec61740012f unchanged.
- `gemma3_real_row_binds_all_104_norm_tensors_and_window_schedule` (env-keyed, with the
  updated Phase 3b invariants): PASS.

### 11i. What Phase 4/5 inherit

- The ≥512-token windowed receipt (Phase 4) is now reachable over the SERVED lane — depth >512
  decode measured working here at speed (45-46 tok/s at depth ~289; §9e-7's merge-correctness
  role stands).
- Phase 5 owes: the capabilities-row rewrite (it still describes the runnable lane) + ledger
  regen; the catalog/row id mismatch fix that currently makes dense responses disclose
  `lane:"experimental"` and keeps `filename_is_supported_exact_row` false for this row;
  frontend fixtures; docs sweep. The plan's new `supported_exact_row_smoke_sub512` level
  string is deliberately scoped until the Phase 4 receipt lands.
- Multi-turn chat pays a full token-by-token prefill every turn (prefix cache stays closed for
  windowed archs, H1/§9e-5) — at 60-80 tok/s prefill this is now noticeable but not painful;
  reopening the cache stays H11 territory.
- Speculative decode: opt-in env only; on gemma3 the CPU verify path now H4-errors (typed)
  and no explicit spec gate was added — flagged for Phase 5/6 if spec decode is ever pointed
  at this row.

## 12. Phase 3c record — adversarial review of the Phase 3b flip (2026-07-30)

Phase 3b (`ba6de7f7`) was put through an adversarial review. Four findings were confirmed
blocking/major before this phase started; eight more were reported by the review's lenses but
its verification pass died on infrastructure, so each was re-verified here against the code
before being fixed or dismissed. **Every one of the twelve turned out to be real.** Four
commits, each gated by fmt / clippy --all-targets -D warnings / cargo test --all-targets.

The stakes framing that shaped every fix: on these paths a routing mistake is SILENT WRONG
OUTPUT, not a crash. The CPU dense forward has no sliding window, no sandwich norms, no GeGLU
and no dual-theta RoPE, so gemma3 on it decodes fluent-looking garbage under a supported label.

### 12a. F1 (BLOCKER) — the fail-closed moved to a choke point

H4 guarded three CPU dense entry points. The flip made four more reachable for gemma3, all
unguarded: `forward_greedy_verify_chunk` (the spec-decode CPU verify walk, live from
`src/api/mod.rs`), `forward_layer_range_from_hidden` (distribute master / activation replay),
`forward_worker_layers` (`src/distributed.rs`), `ghost_forward_one_layer` (`src/main.rs`).

Guarding those four would have left the fifth. **The choke point:
`ensure_windowed_arch_off_cpu_dense_layer` (src/inference.rs), called from
`forward_layer_timed` and `forward_prefill_layer_chunk_timed`** — the only two per-layer dense
forwards in the file. A dense walk cannot be written that skips it, because it cannot compute a
layer without one of them.

**Completeness enumeration.** Every CPU dense layer loop in src/inference.rs, and the choke
point each now hits:

Line numbers are at `b156e92c`; `ensure_windowed_arch_off_cpu_dense_layer` is
src/inference.rs:7114, called from `forward_layer_timed` (:7130) and
`forward_prefill_layer_chunk_timed` (:7898).

| # | entry point (src/inference.rs) | layer loop | per-layer forward | guard reached |
|---|---|---|---|---|
| 1 | `forward_worker_layers` :4076 (prefill arm) | :4114 | `forward_prefill_layer_chunk_timed` | choke point |
| 2 | `forward_worker_layers` :4076 (decode arm) | :4165 | `forward_layer_timed` | choke point |
| 3 | `forward_layer_range_from_hidden` :4218 | :4250 | `forward_prefill_layer_chunk_timed` | choke point |
| 4 | `ghost_forward_one_layer` :4284 | (single layer) | `forward_prefill_layer_chunk_timed` | choke point |
| 5 | `forward_prefill_chunk_timed_fast` :4353 | :4401 | `forward_prefill_layer_chunk_timed` | early guard "batch prefill", then choke point |
| 6 | `forward_greedy_verify_chunk` :4461 | :4521 | `forward_prefill_layer_chunk_timed` | choke point |
| 7 | `forward_prefill_layer_major_timed_fast_inner` :4591 | :4648 | `forward_prefill_layer_chunk_timed` | early guard "layer-major prefill", then choke point |
| 8 | `forward_single_token_timed_internal` :4786 | :4903 | `forward_layer_timed` | early guard "decode forward", then choke point |

There is no ninth: the only other `layers.iter()` sites in the file are binding/merge code, not
forwards. The three session-level guards are KEPT, demoted to an early leg that names the lane
before the walk starts; they delegate to the same free function so the message cannot drift.

The review also found all three existing guard sites were **revert-invisible** — the single H4
test asserts two substrings the choke point also emits, so deleting any guard failed nothing.
Ten tests now cover this (src/inference/tests.rs), each named for what it catches:

| test | fails when |
|---|---|
| `windowed_arch_choke_point_refuses_the_per_layer_decode_forward` | the `forward_layer_timed` choke-point call is deleted |
| `windowed_arch_choke_point_refuses_the_per_layer_prefill_chunk` | the `forward_prefill_layer_chunk_timed` choke-point call is deleted |
| `non_windowed_arch_still_computes_a_cpu_dense_layer` | the choke point starts refusing everything (causality control) |
| `windowed_arch_batch_prefill_names_its_lane_before_the_layer_walk` | the `"batch prefill"` early guard is deleted |
| `windowed_arch_layer_major_prefill_names_its_lane_before_the_layer_walk` | the `"layer-major prefill"` early guard is deleted |
| `windowed_arch_decode_fallback_names_its_lane_before_the_layer_walk` | the `"decode forward"` early guard is deleted |
| `windowed_arch_speculative_verify_chunk_inherits_the_choke_point` | entry point 6 stops being covered |
| `windowed_arch_worker_layer_shard_inherits_the_choke_point` | entry point 1/2 stops being covered |
| `windowed_arch_layer_range_replay_inherits_the_choke_point` | entry point 3 stops being covered |
| `windowed_arch_ghost_layer_probe_inherits_the_choke_point` | entry point 4 stops being covered |

### 12b. F2 (BLOCKER) — CLI lane admission is honest again

`ensure_arch_has_direct_dense_session` (src/main.rs) went capability-aware for gemma3, which
admitted it to all ten lanes it guards. Five of them walk the CPU dense layer loop directly and
can never run a windowed forward on ANY host. They now refuse before weights load, with an
actionable error naming `camelid serve`, via a `DenseLaneWindowedForward` lane class:

- `CpuDenseOnly` (refuse windowed on every host): distribute worker role in `main`,
  `run_ghost`, `load_model_drafter`, `run_bench_speculative`, `run_distribute_worker`,
  `run_distribute_master`.
- `ViaSessionDecode` (capability-aware, unchanged): `BenchAllocGate`, `gait_profile_trial`,
  `run_bench_owner_sweep`, `run_bench_generate` — all of which generate through
  `generate_next_token_with_history_diagnostics` and therefore inherit the resident lane (H2).

Tests: `cpu_dense_only_cli_lanes_refuse_a_windowed_arch_on_every_host` (with a llama control on
the same lane class, so the refusal is caused by the window schedule and not the lane),
`a_kquant_windowed_row_is_refused_even_on_a_session_decode_lane`,
`runnable_only_archs_stay_refused_on_both_lane_classes`.

### 12c. F3 (MAJOR) — routing is keyed on the FILE, not the arch string

`arch_requires_runnable_bridge_given` keyed on (arch, host) while resident admission is Q8_0-
pinned (H5). A gemma3 Q4_K_M on a resident-capable Mac routed to the dense lane, loaded no
runnable runtime, was declined by H5, and died on the H4 error for every request — a hard
regression (the bridge served every gemma3 quant pre-flip), and both the H5 decline text and
the flip's commit message promised a fallback no code implemented.

- `model::file_requires_runnable_bridge(gguf)` is the live predicate. The arch-only entry point
  is REMOVED, not defaulted: every live caller has the `GgufFile` and must pass it, so
  quant-blindness is structurally impossible rather than merely fixed.
- `model::windowed_arch_resident_quant_admissible(gguf)` decides the quant half from GGUF
  metadata pre-load, mirroring the engine pin exactly (all seven per-layer linears Q8_0; a file
  with no recognizable layer linears fails closed to the bridge).
- `arch_requires_runnable_bridge_given(arch, host_available, quant_admissible)` is the pure
  core; a windowed arch needs BOTH to take the resident lane.
- Consumers renamed in lockstep: `api::is_runnable_serve_file`,
  `api::completions_unsupported_for_file`.

**Verification method, stated because it is not a live-file proof:** no non-Q8_0 gemma3 GGUF
exists under `/Volumes/Untitled/models` (or the desktop model dir) — only
`gemma-3-1b-it-Q8_0.gguf`. F3 is therefore proven by unit tests on the predicate
(`a_non_q8_windowed_row_requires_the_runnable_bridge_on_every_host`,
`windowed_quant_admission_requires_every_layer_linear_to_be_q8_0`) plus an HTTP-level test with
a synthetic Q4_K fixture (`a_kquant_gemma3_keeps_the_runnable_bridge_on_a_resident_capable_host`,
identical to the reopen test but for the tensor type, so quantization is provably what decides
it), and the CLI test above.

### 12d. F4 (MINOR) — speculation is declined for the arch, not gated per mode

`serve --spec-decode` with the resident lane armed made every gemma3 request H4-error: CPU
speculation pins the target off the resident paths so chunk-verify rollback has
CPU-authoritative KV. Of the two options offered, neither was taken as written. Teaching
routing about spec mode couples two unrelated decisions and leaves routing env-order-sensitive;
exempting windowed archs from the spec pin is worse, because the CPU verify walk is itself the
window-less dense forward. **Decision: a windowed arch never speculates, on either verify
lane.** The GPU batched verify is no better than the CPU one here — `encode_attention`
(src/metal.rs) takes no `window_start` at all and reads `[0, position_count)` full-causal; it
is shared with the gemma4 lane and was never threaded with the window (§9a records this as an
open residual). Since lossless speculation only ever adds throughput, declining it costs
correctness nothing, and `--spec-decode` now SERVES gemma3 on the plain resident lane instead
of failing. The disqualifier set moved into `api::speculation_admissible`, which also gives the
four pre-existing disqualifiers their first test
(`speculation_is_declined_for_a_windowed_arch_on_both_verify_lanes`). This supersedes §11i's
"no explicit spec gate was added" flag.

### 12e. Triage of the eight unverified leads — all eight CONFIRMED, none dismissed

1. **Dense-lane renderer keyed on TEMPLATE SHAPE while routing admits by ARCH.** CONFIRMED.
   `is_gemma3_chat_template` tests three substrings of the template text; routing keys on
   `config.architecture`. A gemma3 GGUF whose template lacks them fell through
   `render_chat_prompt_for_tokenization_fallback` to `render_role_colon_prompt` — a `"user: …"`
   prompt the model was never trained on, served silently under a supported-row label — while
   the runnable bridge, which keys the same decision on `runtime.architecture`, rendered it
   correctly. The two lanes disagreed for exactly the files most likely to be mis-converted.
   FIXED: `reject_windowed_arch_with_unrecognized_template` fails the arch/template mismatch
   closed with a typed 422 naming the missing markers. Test
   `gemma3_arch_with_an_unrecognized_template_fails_closed_on_the_dense_lane` covers the
   near-miss (turn markers present, `first_user_prefix` absent), the role-colon template, and
   a missing template, with a recognized-template control and non-windowed controls.
2. **H5's Q8_0 admission has ZERO CI coverage.** CONFIRMED, and the mechanism is worse than
   reported: the test claiming to pin it does not short-circuit, it is *vacuous* — its own doc
   comment says the refusal may come from the backend-enabled gate instead, and its fixture's
   f32 weights are rejected by the generic layer loop first, so deleting both H5 lines fails
   nothing on any host, armed or not. The blocker is ordering: the backend-enabled bail sits
   ~120 lines earlier and fires on every ordinary `cargo test` (the flag is default-off), and
   arming it from inside a test is forbidden by §9d. FIXED by extraction:
   `windowed_arch_layers_violate_q8_pin` is the pure decision, pinned per-linear by
   `windowed_arch_q8_pin_rejects_any_non_q8_layer_linear` with an all-Q8_0 control.
3. **No gemma3 chat test on a runnable-FALLBACK host.** CONFIRMED and worse: there was no
   HTTP-level gemma3 *chat* test on ANY host — the only gemma3 HTTP tests hit `/completion`.
   FIXED: `gemma3_chat_routes_to_the_runnable_bridge_on_a_fallback_host` (plain `#[test]`, no
   cfg gate, so it runs on every non-macOS leg — the legs where it matters).
4. **Two of three H4 guard sites revert-invisible.** CONFIRMED and worse — all three were, and
   so was the choke point once it absorbed them. FIXED by the ten tests in 12a; the three early
   guards are now pinned BY LANE NAME, which is what makes their removal visible rather than
   silently absorbed.
5. **catalog_id/row_id mismatch.** CONFIRMED. `catalog_id: "gemma3_1b_it_q8_0"` vs row id
   `"gemma_3_1b_it_q8_0"` made `filename_is_supported_exact_row` false for the one promoted
   gemma3 row, so `classify_model_lane` returned `ExperimentalImplemented` — which both emitted
   `lane:"experimental"` AND whitespace-trimmed the generated text. For a supported exact row
   that text is the parity-claimed artifact and must stay byte-identical, so this was a live
   wrongness on the served surface, not only a label. FIXED here (one line, matching the rule
   the neighbouring qwen3 entry states verbatim). It needs NO ledger regen: the ledger is
   derived from the capabilities literal + API_CONFORMANCE_CASES, and `catalog_id` appears in
   neither; `check-ledger-drift.mjs` passes. The capabilities-ROW rewrite stays Phase 5's.
6. **`completions_fail_closed_for_runnable_serve_archs` reads the live predicate twice.**
   CONFIRMED: tautological (it asserted the gate equals itself) and genuinely racy on a Metal
   host, where seven sibling tests mutate `CAMELID_METAL_RESIDENT_DECODE` under a lock this
   test did not hold. FIXED by driving the pure core with explicit capability booleans — no
   env, no device, no lock needed, and it now pins behaviour instead of restating the
   implementation.
7. **Plan-vs-routing disagreement.** CONFIRMED for `CAMELID_MAC_Q8_METAL_PLAN=0`; CONFIRMED but
   differently-worded for the Safe profile (there the plan disclosed a bare "safe profile"
   reason next to CPU dense labels that for this arch fail closed at every dispatch — arguably
   a worse disclosure). FIXED: `execution_plan::macos_q8_metal_plan_selectable()` is now
   consulted by `inference::windowed_arch_resident_host_available()`, so the two agree, and the
   windowed "serve chats via the runnable bridge" reason is pushed alongside every early return
   that yields `safe_q8_plan()` rather than only the one after the Metal arm.
   **Landmine found while writing this fix, worth the record:** the first cut also consulted
   `CAMELID_MAC_Q8_REPACK`, which is a genuine third early return in `select_macos_q8_plan` —
   and a self-defeating latch. `CAMELID_MAC_Q8_REPACK` is a MANAGED_ENV_KEY that a successful
   Metal-resident selection WRITES to `"off"`, and `env_flag_disabled` reads `"off"` as
   disabled. Routing would therefore have decided the Metal plan was unselectable the instant
   `PlannerEnv::apply` ran, sending gemma3 to the bridge and killing the lane the plan had just
   selected — the whole Phase 3b result, undone by a "consistency" fix. The rule this yields:
   **a plan OUTPUT can never be a routing INPUT.** Only the two operator opt-outs
   (`CAMELID_PROFILE`, `CAMELID_MAC_Q8_METAL_PLAN`, neither plan-managed) are consulted, and
   `macos_q8_metal_plan_selectability_tracks_every_early_return` now pins that
   `CAMELID_MAC_Q8_REPACK=off` does NOT disarm routing, so it is not helpfully added back.
   Residual, OPEN: an operator who PRE-sets `CAMELID_MAC_Q8_REPACK=0` still gets a safe plan
   with resident routing. Closing it needs the plan to stop overloading one variable for both
   "operator opt-out" and "plan output".
8. **gemma3 tool refusal returns a different code/param across hosts.** CONFIRMED. Bridge: 422
   `unsupported_tools`, no param. Dense lane: raised inside template rendering and surfaced
   wrapped as `unsupported_chat_template` / param `"messages"`. Same status, different
   machine-readable identity, decided purely by host capability — a client switching on
   `error.code` degraded correctly on Linux and fell into a generic branch on the M4 mini.
   FIXED: one shared constructor (`gemma_runnable_lane_tools_rejection`), raised arch-keyed at
   the dense chokepoint ahead of tokenizer/template work.
   **§11f correction:** its claim that the dense-lane refusal is "behavior-IDENTICAL to pre-flip
   from the API user's perspective" was an overclaim — the refusal SEMANTICS were identical
   (422 class, tools never silently dropped, row stays `tool_capable: false`), the IDENTIFIERS
   were not. They are now.

### 12f. Nothing dismissed

No finding was dismissed. The one place a reported framing was corrected rather than accepted
is §12e-4 (the review said two of three guard sites were revert-invisible; all three were), and
§12e-2/3, where the confirmed defect is larger than described. F4's two suggested remedies were
both rejected in favour of a third, for the reason in 12d.

### 12g. Gates

Per commit: cargo fmt clean; `clippy --all-targets -D warnings` clean; `cargo test --all-targets`
exit 0 under pipefail. Final battery at `34bf672e`: **lib 1398 passed / 0 failed / 23 ignored**
(against 1376/0/23 at the flip commit `ba6de7f7` — +22 lib tests), 60 green integration tallies,
every suite green. `check-public-scrub.sh` clean. `scripts/check-ledger-drift.mjs` passed
(ledger == code contract; the `catalog_id` fix in 12e-5 touches neither the capabilities literal
nor API_CONFORMANCE_CASES, so no regen was needed — confirmed by the check, not assumed).

One pre-existing test needed a mechanical update: `tests/invariant_matrix_binding.rs`'s
I-cache-quant na anchor `"fn is_runnable_serve_arch"` became `"fn is_runnable_serve_file"`. The
anchor did its job — it caught the rename rather than letting the na verdict go stale.

Env-keyed battery (release, production GEMV configuration + resident decode armed FROM THE SHELL
per §9d, targeted names so the env-mutating gate tests never run alongside):

- **Phase 2 real-row final gate** (`gemma3_real_row_resident_forward_matches_runnable_oracle`),
  no SKIP, 585.18 s: depth 1 argmax 108 = oracle, max abs logit diff 6.247e-5; depth 5 argmax
  1077 = oracle, 7.820e-5; depth 50 argmax 578 = oracle, 9.584e-5 — **50/50 greedy tokens
  identical, overall max abs logit diff 2.122e-4**. Bit-for-bit the §9b / §10d / §11h record.
- **3a session-level prefill gate**
  (`gemma3_session_level_token_by_token_prefill_matches_runnable_oracle`), 22.20 s: depths 1-5
  argmax 108 / 584 / 568 / 2364 / 1077, all = oracle; **5/5 identical, overall max abs logit
  diff 7.820e-5**. Bit-for-bit the §10b / §11h record.

The whole Phase 2/3a stack is therefore proven UNCHANGED under the Phase 3c fixes.

### 12h. Serve smokes (release, this M4 mini 16 GB, no special env vars)

Resident (`camelid serve --model gemma-3-1b-it-Q8_0.gguf --no-open`):

- `/v1/health`: `generation_ready:true`, backend `"llama"` (dense serve lane, no runnable
  runtime), plan `metal_resident_q8_runtime` / `q8_0_metal_resident_decode` /
  `supported_exact_row_smoke_sub512`.
- Greedy chat, 20 prompt tok: coherent Rayleigh-scattering sentence, finish `stop`, 26 tokens.
  Wall 795.3 ms cold-ish / **684.0 ms warm** (§11g: 825 / 788 ms), token ids byte-identical
  across runs.
- Long greedy (26 prompt / 240 completion, two runs, byte-identical): wall 5486.7 / 5445.5 ms =
  43.7 / 44.1 tok/s end-to-end INCLUDING prefill; netting the ~26-token prefill leaves
  **≈46-47 tok/s decode** at depth ~266 — the §11g band (45.0 / 46.0 tok/s at depth ~289) with
  no regression.
- First 8 greedy token ids `[818, 7217, 7412, 3730, 1547, 529, 496, 20284]` — **IDENTICAL to
  the §11g oracle-verified head**, and identical to the fallback server's below.
- Tools: 422 **`unsupported_tools`, param absent** — the 12e-8 fix, verified live. §11g recorded
  `unsupported_chat_template` here; the two lanes now return the same object.
- `lane` field: **ABSENT** — the 12e-5 catalog_id fix, verified live. §11g recorded
  `lane:"experimental"`, which also meant the generated text was whitespace-trimmed; it is now
  served byte-identical, as a parity-claimed row must be.
- Raw `/v1/completions`: 200, `"The capital of France is"` → `" Paris.\n\nThe largest city in
  France is Paris.\n\n"` (the §11g continuation).

Fallback (`CAMELID_METAL_RESIDENT_DECODE=0`, same command):

- `/v1/health`: backend `"runnable-runtime"`; plan `cpu_reference` / `safe_cpu_decode` carrying
  the windowed-arch reason verbatim.
- Greedy chat 8 tok: 27.54 s (§11g: 27.07 s — the known ~0.2-0.3 tok/s runnable lane), token ids
  `[818, 7217, 7412, 3730, 1547, 529, 496, 20284]` — byte-identical to the resident lane's head,
  which is the cross-lane identity claim this campaign exists to hold.
- Tools: 422 `unsupported_tools`, param absent (unchanged; the dense lane moved TO this, not the
  other way).
- `/v1/completions`: 422 `unsupported_completions_lane` (gate intact verbatim).

Both servers killed by saved PID only.

### 12i. Still open after Phase 3c

- **Operator-set `CAMELID_MAC_Q8_REPACK=0`** still yields a safe plan with resident routing
  (12e-7). Needs the plan to stop overloading one variable for both operator opt-out and plan
  output.
- **`encode_attention` carries no `window_start`** (§9a residual). It is why speculation is
  declined for windowed archs (12d) rather than supported; threading it is the prerequisite for
  ever pointing spec decode at this row.
- **`scripts/chat-parity-gemma3.mjs` defaults `--row-id` to the old `gemma3_1b_it_q8_0`
  spelling**, so a receipt generated without an explicit flag carries a row id that matches no
  compatibility row. One line, deliberately left to Phase 5's docs/scripts sweep rather than
  widened into here.
- **`camelid pull` alias changed** as a consequence of 12e-5: the catalog id is now
  `gemma_3_1b_it_q8_0`. `README.md:227` already advertised a third spelling (`gemma3_1b`) that
  resolved to neither, so the README row is no more wrong than before — but it is wrong, and it
  belongs to the Phase 5 docs sweep.
- Phase 5's own inheritance from §11i is unchanged: the capabilities-row rewrite + ledger regen,
  frontend fixtures, docs sweep. The `lane`/trim half of that list is now CLOSED by 12e-5.

---

## 13. Phase 4 record (2026-07-30)

Phase 4 is the receipt phase: the SERVED resident lane compared against the pinned EXTERNAL
oracle, not against the runnable lane — whose oracle role expires at the 512-token window.
Bundle: `qa/evidence-bundles/gemma3-1b-q8-gpu-resident-parity-20260730-head-6eaf9053/`
(20 artifacts, README + manifest + SHA256SUMS).

### 13a. The oracle exists and is the pinned one

`llama-server` **version 9632 (`acd79d603`)** at
`acd79d603cb2e1c84c0886137b80f1ad649b6857`, clean tree, Release build; binary sha256
`382096b1dc10da68c2bf0a97e1f0dd36db90531cdea1434760d8c1a70fea1310`. Run on its **CPU** backend
(`-ngl 0 -ctk f32 -ctv f32 -fa off --no-repack -c 4096`) — the same comparator configuration the
frozen runnable bundle used, so the two gemma3 receipts share an oracle. Two-phase discipline
held for every capture, comparison and probe: the two engines were never resident together.

### 13b. Sub-512 receipts

**Chat (`scripts/chat-parity-gemma3.mjs`, committed gate pack, depths 1/5/50): CLEAN.**
Cross-engine prompt tokenization identical 5/5 (16/16/16/19/17 tokens); **15/15 generation legs
token-AND-text identical**; `all_pass: true`, zero flips. Note the contrast, stated carefully:
the frozen runnable bundle recorded one flip on this same pack (0.3416 nat); this bundle does
not re-adjudicate it — different host, different lane, and the runnable lane was not re-run on
the sub-512 pack here.

**Raw decode (`scripts/raw-decode-parity.mjs`, harness default 4 prompts, depths 1/5/50):
committed as-is with `all_pass: false`, 6/12 legs token-AND-text identical.** Two distinct
findings are inside that number:

1. A **harness re-encode artifact**, not an engine divergence. The harness scores camelid by
   re-encoding its TEXT; on this 262k SPM vocab a run of spaces re-encodes as single-space
   tokens (`236743`) where llama.cpp emitted merged whitespace tokens (`138`, `140`). Reading
   camelid's ACTUAL ids (`camelid.generated_token_ids`, `camelid-raw-probe.json`) removes it:
   `"Q: What is 2+2? A:"` is token-identical for all 43 generated tokens plus the `106` stop.
   The harness was NOT changed to hide this.
2. **Three real flips, all at depth 50**, each probed from both sides
   (`near-tie-analysis.json`):

| prompt | idx | ref | camelid | camelid top-2 gap | oracle top-2 gap (no-repack / repack) | oracle rank of camelid tok |
|---|---|---|---|---|---|---|
| `The capital of France is` | 44 | 9639 | 32219 | 0.0032 | 0.0431 / 0.0285 | **1** |
| `Once upon a time,` | 37 | 4658 | 11207 | 0.0173 | 0.4471 / 0.1398 | 2 |
| `def fibonacci(n):` | 5 | 2094 | 22304 | 0.0353 | 0.4402 / 0.4402 | **1** |

Two of the three are **oracle-side flips** — the same pinned binary and flags emit camelid's
token when the position is scored from a re-fed prefix and the reference token when it decodes
continuously from the raw prompt (`probe-oracle-continuous.json`); on prompt 0 the prefix-fed
oracle also flips back when only its repack kernel changes. The third (`Once upon a time,`) is
the weakest attribution: the oracle is rank-1-stable across all four kernel/thread controls AND
the continuous control, and camelid's token is oracle rank 2 at **0.4471 nat** — ABOVE the
0.33-nat Ornith line and above the frozen bundle's 0.3416. Disclosed as such, with the two
mitigating measurements stated rather than argued: camelid's own gap there is 0.0173 nat, and
the oracle's own gap moves 0.31 nat (0.4471 → 0.1398) when only its repack kernel changes.

Consequence for Phase 5, and it is a real one: **there is no token-exact depth-50 claim for raw
`/v1/completions` on this row.** Depths 1 and 5 are clean on 4/4.

### 13c. The >=512 windowed receipt — the deliverable this campaign exists for

New pack `qa/prompt-packs/gemma3-windowed-context-pack-v1.json`: three plain-English prompts
whose rendered gemma3 turns tokenize to **606 / 1205 / 2403 tokens** = 1.18x / 2.35x / 4.69x the
file's own `gemma3.attention.sliding_window = 512`, each answerable only from its first sentence.

**Resident lane vs the pinned oracle: 3/3 prompt tokenization identical, 9/9 generation legs
token-AND-text identical, ZERO flips, `all_pass: true`.** §7d required this pack to be clean and
it is — the envelope is not drawn on at all.

**The runnable lane on the same prompt, same oracle capture, same depth** (resident off,
`cpu_reference` / `safe_cpu_decode`): diverges at generated index 2 and never resynchronises —
oracle and resident emit `[818,103708,563,...]` "The Willow is the name of the river that runs
past the town."; the runnable lane emits `[818,103708,7940,236761]` "The Willow River.".
Prompt tokenization was identical (606/606) on both lanes, so they saw the byte-same input.
**And it is not a near-tie**: re-fed the identical prefix, the oracle ranks its own/resident
token `563` at logprob -0.2125 and the runnable lane's `7940` second at **1.667 nats** behind
(`probe-window-divergence.json`) — four times the largest disclosed near-tie in the bundle.
Attribution is stated as attribution: the two lanes are token-identical BELOW the window (13e's
two gates), and the one documented architectural difference is the window mask the runnable lane
does not implement.

Cost bound, recorded: the runnable lane prefills 606 tokens in ~10.8 min (~0.2 tok/s), so only
the shortest windowed prompt was run on it, at the single deepest leg. 1205/2403 on that lane
were not attempted.

### 13d. Determinism

Two fresh `camelid serve` processes, full stop/start between them, no env overrides, greedy.
Each session records 6 chat legs (incl. the 2403-token windowed prompt) and 5 raw-completion
legs carrying ACTUAL generated token ids (incl. a 2395-token windowed prompt).
`det-run1.json` and `det-run2.json` are **byte-identical files**, sha256
`632992c609941494905650a186ec255bf7d545950f4490aedc6a3c7158bf64d3`.

### 13e. Gates

- `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean;
  `cargo test --all-targets` **exit 0** under pipefail (every suite green, 0 failed).
- `scripts/check-public-scrub.sh` clean.
- All four bundle validators pass with the new bundle present:
  `check-public-scrub.sh`, `audit-evidence-bundle-privacy.mjs --strict` (0 findings),
  `check-evidence-bundle-checksums.sh`, `check-public-evidence-claims.mjs`
  (159 manifests, up from 158).
- Env-keyed gemma3 gates re-run at this head, targeted, release, production GEMV gates armed
  from the shell:
  - `gemma3_real_row_resident_forward_matches_runnable_oracle` (release, f32y+wire+NSG8):
    depth 1 argmax 108 = oracle, max |logit diff| 6.247e-5; depth 5 argmax 1077, 7.820e-5;
    depth 50 argmax 578, 9.584e-5 — **50/50 greedy tokens identical, overall max |logit diff|
    2.122e-4**, 565.79 s. **Bit-for-bit the §9b/§11h record.**
  - `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle` (same gates plus
    `CAMELID_METAL_RESIDENT_DECODE=1`): 108 / 584 / 568 / 2364 / 1077, overall max |logit diff|
    **7.820e-5**, 5/5 identical, 22.05 s. **Bit-for-bit the §10b record.**

### 13f. Script fixes carried in this phase (receipt-label only, no engine behaviour)

`scripts/chat-parity-gemma3.mjs`: (1) `--row-id` now defaults to the real row id
`gemma_3_1b_it_q8_0` — this closes the §12i one-liner; (2) new `--lane-label` so the emitted
receipt names the lane it actually certified instead of hardcoding
`gemma3_marker_chat_greedy_runnable_serve`; (3) `postJson` moved from the global `fetch` to
`node:http` with a `--request-timeout-ms` flag, mirroring `scripts/raw-decode-parity.mjs` — the
undici ~5-minute header timeout aborted the client mid-request while the CPU runnable lane was
still legitimately prefilling a 606-token prompt, which is exactly the leg 13c needs.

### 13g. What Phase 5 MAY and MAY NOT claim

**MAY:**
- The row runs on the Metal GPU-resident serve lane by default and is token-AND-text identical
  to llama.cpp `acd79d603` on the committed chat gate pack at depths 1/5/50 (15/15), with 5/5
  cross-engine prompt tokenization.
- **Correct sliding-window behaviour above 512 tokens**, proven against the external oracle at
  606 / 1205 / 2403 prompt tokens, depths 1/5/50, 9/9 legs, zero flips. `tested_context` may
  move from "well under the 512-token sliding window" to a bounded chat claim of **2,403 prompt
  + 50 generated tokens**.
- Determinism: byte-identical decode across two fresh serve processes, including past the
  window.
- That the runnable CPU lane is measurably the wrong reference above the window (1.667-nat
  disagreement with the oracle at the divergence position) — i.e. the `full_support_blockers`
  sentence "context above the 512-token sliding window ... mathematically wrong by construction"
  now applies to the RUNNABLE lane only, and is CLOSED for the resident lane up to 2,403 tokens.

**MAY NOT:**
- No perf or throughput claim, and no speed comparison with llama.cpp.
- No token-exact claim for raw `/v1/completions` at depth 50 (three disclosed flips, one of them
  a 0.4471-nat stable-oracle near-tie). The clean raw-decode claim stops at depth 5.
- No context claim above 2,403 prompt tokens; the file's native 32,768 is UNMEASURED here, as is
  everything between ~2.4k and 32k.
- No claim for any other gemma3 row or quant (§7c scope pin holds).
- No multi-turn, streaming, tool-calling, speculative-decode or prefix-cache claim.
- The runnable-lane divergence leg is ONE prompt at ONE depth; it is a demonstration, not a
  runnable-lane receipt, and it must not be cited as a general runnable-lane characterization.
- The frozen runnable bundle's sub-512 0.3416-nat flip is NOT re-adjudicated by this bundle.

### 13h. Still open after Phase 4

- The raw-decode depth-50 near-ties are a live limit on the row's claimable surface; closing
  them would need either a chat-shaped raw pack or an oracle-side reduction-order study, and
  neither is scoped.
- §12i's other open items are unchanged except the `--row-id` default, which is CLOSED here.
- The runnable lane's cost makes a full windowed cross-lane matrix impractical; if a broader
  divergence table is ever wanted it needs a faster CPU reference, not more patience.

---

## 14. Phase 5 record (2026-07-30)

Phase 5 is the promotion-surface phase: the row, the ledger, the frontend, the docs, and a
decision-log entry. No engine behaviour changed in this phase — the only `src/` edits are the
capabilities literal, four stale comments, and the execution-plan comment that Phase 3b left
pointing at a future that has now arrived.

### 14a. The capabilities row

`gemma_3_1b_it_q8_0` (src/api/mod.rs) rewritten against §13g and nothing wider.

- `family` `gemma3_runnable_decoder` → **`gemma3_windowed_decoder`**. The old name described a
  lane the row no longer defaults to; the new one describes the architecture.
- `support_scope` → **`exact_row_metal_gpu_resident_windowed_chat_smoke_only`** (new literal;
  `ledger/camelid-ledger.schema.json` `supportScopeVocabulary` extended in the same commit, which
  `check-ledger-schema.mjs` enforces as a superset check).
- `generation_runs` names both halves of the split:
  `metal_gpu_resident_serve_chat_greedy_with_eog_stop_default_lane_on_metal_hosts_runnable_bridge_fallback_elsewhere`.
- `tested_context` moves from "well under the 512-token sliding window" to
  `gemma3_chat_greedy_1_5_50_at_606_1205_and_2403_prompt_tokens_plus_50_generated_all_above_the_512_token_sliding_window_metal_resident_lane_only`.
- `parity_audited` states 15/15 sub-512 and 9/9 windowed, and that raw completions are clean
  through depth 5 only.
- `performance_measured` → `not_claimed_resident_lane_throughput_is_a_separate_unshipped_measurement_phase`.
  The old `observed_about_5_s_per_token_...` value was DELETED and not replaced. Note the
  side effect this buys: the frontend's `hasExactRowBoundedPerformanceEvidence` rejects any value
  containing `not_`, so the row cannot advertise throughput readiness even by accident.
- `frontend_readiness_gate` names the resident backend/decode path and states plainly that off
  that lane only the sub-512 envelope is green.
- `full_support_blockers` keeps the old ">512 is wrong by construction" sentence but re-attributes
  it to the RUNNABLE lane, and adds the raw-decode depth-50 limit, the >2,403 limit, the
  un-run bounded-context ladder, and the fail-closed surfaces.
- `latest_checked_*` and `evidence` point at the Phase 4 bundle; the frozen July bundle is named
  as history for the other lane and explicitly not re-adjudicated.

### 14b. The execution-plan level string STAYS `sub512`, deliberately

§11i left this open. The answer is not to widen it. `recognized_row_level` is keyed on the row
NAME only and is platform-blind — the same string is reported on hosts where the unmasked
runnable bridge serves the row. `supported_exact_row_smoke_sub512` is the envelope that holds on
every host that recognizes this row; a `..._windowed_2403` string would over-claim on the
fallback. The stale "this level string stays sub-512 until [Phase 4] lands" comment is replaced
with that reasoning, and `/api/capabilities` carries the lane-aware claim as the support source
of truth. Recorded in DECISIONS D20's closing section.

### 14c. Ledger

`node scripts/extract-capabilities-to-ledger.mjs` → 37 model rows, 40 contract fields each.
`check-ledger-schema.mjs` passed (2 documents, coverage ⊇ code enums). `check-ledger-drift.mjs`
passed all five checks: A freshness ok; B 22/24 support-claim rows mapped (the 2 unmapped are
pre-existing DiffusionGemma/Ornith label mismatches, logged not failed); **C 14/14 catalog ids
resolve** (13/13 before — the gemma3 catalog entry is the new one, and its resolution is the
Phase 3c catalog_id/row_id fix holding); D 14 correct filename+sha co-occurrences; E anchors
single-home.

### 14d. Frontend

The row had NO frontend presence at all before this phase.

- `frontend/src/lib/supportedModels.js`: Gemma 3 1B-It added with `catalog_id`
  `gemma_3_1b_it_q8_0` — the same id as the contract row and the backend catalog.
- `model-lanes-smoke.mjs`: real row id + two lane checks (resolves to `supported` from the
  filename alone with no backend `lane_class` hint; a Q4_K_M gemma3 file does NOT inherit it).
- `model-state-smoke.mjs`: the row copied field-for-field OUT OF THE GENERATED LEDGER, plus the
  counter-case to the gemma2 planning row above it, the Q8_0 pin, and a throughput-readiness
  guard assertion.
- `capability-readiness-smoke.mjs`: the shipped id resolves; the historical `gemma3_1b_it_q8_0`
  spelling must NOT; the bare family string must NOT.

No `executionPlan.js` change was needed — `metal_resident_q8_runtime` was already a known Metal
backend and `runnable-runtime` a known specialized one, so both sides of the split already render.

**Fixture verification was not by inspection.** A release `camelid serve` was run on this M4
against the real row and `/api/capabilities` fetched live; the gemma3 row is **field-for-field
identical to the regenerated ledger (40/40)**, and all 20 fields of the model-state fixture are
byte-identical to the live payload. The catalog endpoint returns `catalog_id`
`gemma_3_1b_it_q8_0`, and a live chat returns "Paris" with `finish_reason: stop` and NO
`lane:"experimental"` disclosure — the join the Phase 3c fix was for, confirmed end to end rather
than assumed.

### 14e. Docs — two passes, and the second one was necessary

Pass 1: README (pull id `gemma3_1b` → `gemma_3_1b`, matching the catalog-id-prefix convention
every other row follows, closing the §12i three-spelling item; plus the serve-lane table row),
COMPATIBILITY, SUPPORT_MATRIX, STATUS (campaign note, the July promotion marked as history rather
than edited, and the new bundle added to Durable evidence anchors — its only index home, per
drift Check E), DOCS, and DECISIONS D20.

Pass 2 swept the whole tree, not just the files pass 1 touched, and found eight real things.
The worst was mine: **D20.2 as first written documented the design Phase 3c explicitly rejected**
— it named `ensure_windowed_arch_off_cpu_dense`, claimed "all three dispatch sites", tabulated
three entry points and told future readers to add a fourth row when a fourth appeared. That is
the whack-a-mole F1 exists to end. Rewritten around the shipped choke point
(`ensure_windowed_arch_off_cpu_dense_layer`, two per-layer forwards, three session-level guards
demoted to courtesy legs, ten pinning tests). The same stale wording is fixed in the
capabilities-row comment. Also in pass 2:

- `scripts/chat-parity-gemma3.mjs` still DEFAULTED `--lane-label` to `..._runnable_serve`. The
  harness only speaks HTTP and cannot observe its lane, so a resident run would emit a receipt
  stamped runnable — the same class of bug as the `--row-id` default closed in §13f. Now fails to
  `..._serve_lane_unspecified`.
- `src/inference.rs` claimed "gemma3 itself remains behind the arch disqualifier", contradicting
  that disqualifier's own comment 55 lines above.
- `src/model.rs` said gemma3 "serves via the runnable lane" and its resident lane is "guarded off".
- `qa/invariant_lanes.json` cited `fn is_runnable_serve_arch`, a symbol that no longer exists —
  the meta-test checks the anchor array but not the reason prose, so CI was green on a dead symbol.
- `BACKEND_ASKS.md` still listed gemma3's dense-path dual-rope-base hazard as open; it is closed.
- The gate pack's `purpose` asserted the runnable lane and a sub-512 ceiling. The pre-registered
  sentence is kept VERBATIM (MUSTER 6.1) with a dated amendment appended; prompts untouched.
- COMPATIBILITY was the one promoted row on that table citing no evidence bundle; and the bare
  "clean 4/4" read as a silent drop from the 5-prompt chat pack — raw completions are a separate
  4-prompt harness, now said so in the row, STATUS and SUPPORT_MATRIX.

### 14f. Deliberately NOT done

- gemma3 is NOT added to `qa/evidence-bundles/README.md` (STATUS.md is the canonical anchor
  index), `docs/benchmarks/PARITY.md` (frozen to the four-row era — Mistral, Qwen3, gemma4 and
  the K-quant rows are all absent; adding one row would imply a completeness the file lacks),
  `docs/benchmarks/BENCHMARKS.md` or `docs/perf-deep-dive/LANE_STATUS_LEDGER.md` (throughput
  surfaces, and this row has no throughput claim), or `CAPABILITY_MATRIX.md` (receipt-backed
  capability axis; gemma3 has no capability receipts).
- The `bounded_context_*` pack fields stay `not_promoted`. The windowed pack is a different
  artifact from the repo's bounded-context ladder, and the ladder was not run.

### 14g. Gates

Per commit: `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean;
`cargo test --all-targets` **exit 0** under pipefail (debug — the configuration CI runs).

- `scripts/check-public-scrub.sh` exit 0.
- All 27 `scripts/test-*.mjs` PASS.
- `check-ledger-schema.mjs` and `check-ledger-drift.mjs` pass (§14c).
- All four bundle validators pass: `check-public-scrub.sh`,
  `audit-evidence-bundle-privacy.mjs --strict` (0 findings, 0 bundles with findings),
  `check-evidence-bundle-checksums.sh`, `check-public-evidence-claims.mjs`
  (159 manifests, 50 summary files).
- Frontend: `npm run build` PASS. Node-only smokes PASS: model-state, model-lanes,
  catalog-activation, first-run, catalog-browse, model-deletion, 3b-closure,
  capability-readiness, integration, workspace, streaming, ui.

**Env-keyed gemma3 battery re-run at this head** (release, production GEMV gates armed from the
shell per §9d, targeted names):

- `gemma3_real_row_resident_forward_matches_runnable_oracle`, no SKIP, **553.09 s**: depth 1
  argmax resident=oracle=108, max |logit diff| 6.247e-5; depth 5 argmax 1077, 7.820e-5; depth 50
  argmax 578, 9.584e-5 — **50/50 greedy tokens identical, overall max |logit diff| 2.122e-4**.
  **Bit-for-bit the §9b / §11h / §13e record.**
- `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle` (same gates plus
  `CAMELID_METAL_RESIDENT_DECODE=1`), **21.98 s**: depths 1-5 argmax 108 / 584 / 568 / 2364 /
  1077, all = oracle; **5/5 identical, overall max |logit diff| 7.820e-5**. **Bit-for-bit the
  §10b / §11h / §13e record.**

The Phase 2/3a stack is therefore proven unchanged under everything Phase 5 did — as expected,
since Phase 5 changed no engine behaviour.

**Live serve verification** (release, this M4 mini, no special env vars): `/v1/health` reports
`selected_backend: metal_resident_q8_runtime`, `decode_path: q8_0_metal_resident_decode`,
`prefill_path: q8_0_metal_resident_prefill`, `generation_ready: true`. `/api/capabilities`'s
gemma3 row is field-for-field identical to the regenerated ledger (40/40). The catalog endpoint
returns `catalog_id: gemma_3_1b_it_q8_0`. A greedy chat returns "Paris" with
`finish_reason: stop` and **no `lane:"experimental"` disclosure**.

**Two pre-existing failures, disclosed rather than absorbed into this campaign's record:**

1. `tensor::nvfp4_tests::encode_vectors_reproduce_pin_wire_bytes_and_dequant` FAILS in a
   `--release --lib` run (1397 passed / 1 failed / 23 ignored) on the `path-snan-first` case,
   and PASSES in debug. This branch does not touch `src/tensor/` at any commit
   (`git diff origin/main..HEAD -- src/tensor/` is empty), and CI runs the debug configuration,
   which is green. It looks like sNaN handling being folded away under optimization; since
   NaN-sentinel NVFP4 files are supposed to fail closed, it may be a real defect in shipped
   release builds rather than a test bug. Raised as separate work, NOT fixed here.
2. `npm run smoke:observatory` fails one check ("honest no-traffic state renders with no
   requests / title missing"). Verified pre-existing by checking `frontend/` out at the
   pre-Phase-5 head `5b047a3e` and re-running: identical failure. That smoke is not in the CI
   workflow's frontend job.

Two frontend smokes (`first-run-card`, `offline-banner`) could not run on this host — they need
Chrome or Edge, which is not installed. CI's frontend job supplies a browser.

### 14h. Still open after Phase 5

- Perf is Phase 6 and is now the ONLY thing the row's `next_step` leads with. Nothing in the
  repo claims a gemma3 throughput number.
- The three depth-50 raw-completion near-ties (§13h) are unchanged and still bound the row's
  raw surface to depth 5.
- Context above 2,403 prompt tokens, and the bounded-context ladder packs, remain unrun.
- The runnable CPU bridge still has no window mask. That is now stated as a lane-scoped blocker
  rather than a row-wide one, but it is the reason the windowed claim cannot travel off Metal.
- `encode_attention` still carries no `window_start` (§12i), which is why speculation stays
  declined for windowed archs.

---

## 15. Phase 6 record — performance (2026-07-31)

Phase 6 is the throughput phase, and it runs under one hard rule inherited from §13e:
**every perf commit must leave decode byte-identical to the Phase 4 receipts.** Section
15c explains why that rule, not the kernel catalogue, is what actually determined the
shape of this phase.

Branch `feat/gemma3-metal-perf`, based on the PR #560 head `1f5430f0`. Host: this M4 Mac
mini, 16 GB, 10 GPU cores. No GPU profiler exists here (Apple perf counters are
entitlement-gated; the headless route is a confirmed dead end), so everything below is
wall-clock through the serve path, the engine's own `CAMELID_RESIDENT_TRACE` per-token
timers (which read the command buffer's GPU-side timestamps), and A/B env toggles.

### 15a. Measured baseline

One `camelid serve` alive at a time, started and killed by saved PID. Prompts supplied as
EXACT token ids via `camelid_prompt_token_ids`, streamed `/v1/completions`, `temperature:0`,
every row run at least twice and reported warm. `depth` is the mean context over the
generated span. `/v1/health` confirmed `metal_resident_q8_runtime` /
`q8_0_metal_resident_decode` for every run rather than assuming it.

| depth | ms/token | decode tok/s | % of the ~110 tok/s roofline |
|---|---|---|---|
| ~64 | 16.19 | 61.8 | 56% |
| ~262 | 28.63 | 34.9 | 32% |
| ~510 | 45.42 | 22.0 | 20% |
| ~1020 | 49.20 | 20.3 | 18% |
| ~2030 | 57.87 | 17.3 | 16% |

Prefill (token-by-token through the decode path — no batched GPU prefill exists at
head_dim 256): 64 tokens 0.85 s / 74.9 tok/s; 512 13.98 s / 36.6; 1200 45.80 s / 26.2;
2400 112.68 s / 21.3. Load with a warm page cache is 2.6-3.1 s (`CAMELID_METAL_NOCOPY` is
active, so the weights are mmap'd wire pages and the first generation pays the fault-in);
a genuinely cold load was NOT measured — dropping the page cache needs sudo on this box.

The five decode points obey one law, fitted against the per-layer attention work
`sum_l position_count_l = 22*min(d,512) + 4*d`:

> **ms/token = 12.9 + 2.372e-3 * sum_l position_count_l**

Residuals 0-4% across all five depths. Note the layer split is **22 windowed / 4 global**,
not the 20/6 the campaign scoping assumed: with a `sliding_window_pattern` of 6 the globals
are layers 5/11/17/23. A 20/6 model over-predicts the d=2030 point by 14%; 22/4 lands
within 1.7%. Cross-check: the "~46-47 tok/s at depth ~266" figure this phase started from
is reproduced by the law at a mean depth of ~140, i.e. it was a FINAL depth of 266 behind a
short prompt, not a mean depth of 266.

### 15b. Attribution

**CPU vs GPU, measured not inferred.** `CAMELID_RESIDENT_TRACE=1`, decode phase:

| position | encode (CPU) | commit_wait | gpu_busy |
|---|---|---|---|
| 70 | 1.2-2.1 ms | 16.8 ms | 17.0-18.3 ms |
| 490 | 1.2-2.1 ms | 42.8 ms | 43.1 ms |
| 2004 | 1.1-2.4 ms | 57.5 ms | 57.8 ms |

The GPU is >=97% of the token at every depth, and CPU encode is flat at ~1.5 ms and fully
overlapped (encode-ahead is live — `next_encode` is non-zero). **Every CPU-side lever is
therefore worth zero**, including the one that looks most attractive on inspection: the
per-layer `write_buffer_f32` re-upload of the norm and RoPE tables, ~560 KB of memcpy plus
~800 `pool_get` calls per token. Recorded so nobody spends a day on it.

The GPU timestamps also confirm the law independently of the fit: the gpu_busy slope is
2.37e-3 and 2.33e-3 ms per layer-position over the two intervals.

**Inside the 12.9 ms fixed term.** Prefill runs the same graph without the logits stage,
which is a free A/B: prefill gpu_busy is 15.5 ms at position 100 against a decode
prediction of 19.0, and 34.3 ms at position 400 against 37.5. So the tied 262,144-row LM
head plus sampling tail is **3.2-3.5 ms/token** (~80% of its own 2.68 ms roofline), leaving
~9.6 ms for 26 layers of GEMVs, norms, RoPE and dispatch (~64% of their 6.18 ms roofline).
Stated with its uncertainty: that split assumes prefill and decode differ ONLY by the
logits stage. Nothing in the tree proves that directly and individual dispatches cannot be
timed here; the +/-0.3 ms between the two probes is the visible error bar, and a systematic
error would move the split but not the 12.9 ms total, which is measured. The 9.6 ms was NOT
decomposed further — a ~470-dispatch graph at a plausible 3-8 us per dispatch would account
for the 3.4 ms gap to roofline, but that is dispatch counting, not measurement.

**The bottleneck.** At 4 query heads / 1 KV head / head_dim 256 the lane falls to the v1
`attention_decode_f32` kernel, and that kernel dispatches ONE threadgroup per query head of
ONE 32-lane simdgroup — **4 threadgroups, 128 threads, for a whole layer's attention on a
10-core GPU**, with each lane streaming a full 1 KB K row so consecutive lanes touch
addresses 1 KB apart. Measured 2.372 us per layer-position = 0.86 GB/s counting unique K+V
bytes, 3.5 GB/s counting the 4x GQA re-read: **0.7-3% of this machine's bandwidth**. Share
of the token: 24% at d=64, 70% at d=510, 80% at d=2030.

**Not missing anything the other resident rows get:** weight residency, NOCOPY wire-page
fast load, encode-ahead, the GPU sampling tail, and the production
`q8_0_block_linear_row_ksplit_f32y_wire_nsg8` GEMV (256 threads/TG, `rows/2` threadgroups —
131,072 threadgroups on the 262k head, so not launch-starved, consistent with its measured
~80% of roofline) are all live, and `apply_default_fast_stack` arms the same F32Y/WIRE/NSG8
configuration the parity receipts were taken in. Two genuine exclusions, both pre-existing
and both correctness-scoped rather than perf bugs: no prompt-prefix reuse on this row
(§9e-5 — `prepare_for_prompt_prefix_cache` requires the F16-primary flag, and gemma3 Q8_0 is
F32-primary), and no batched GPU prefill at head_dim > 128. Any throughput statement about
multi-turn chat must carry the first.

Measured aside worth keeping: sending `logit_bias` takes a request OFF the GPU sampling
fast lane and costs ~13% decode (61.8 -> 52.4 tok/s at d=64). It is not a neutral benchmark
knob, and the harness does not use it.

### 15c. The parity rule is what decided this phase

Metal compiles with fast-math ON, so it is free to re-associate reductions. Byte-identical
decode therefore excludes **every** restructuring that moves a floating-point reduction
order: flash / online-softmax rewrites, split-K over positions, reading the f16 KV mirrors
(which the split-K path already does for head_dim <= 128 and which would halve KV traffic),
and float4 vectorised dot products. What survives is only work whose individual arithmetic
chains are ALREADY independent and can be spread over more threads without touching any
order. That is a much smaller design space than the kernel catalogue suggests, and it is
the reason this phase shipped one change rather than five.

### 15d. What shipped

**`gemma3-metal: Phase 6 — split decode attention into scores/softmax/context`**
(`15d0b039`). The v1 kernel becomes three dispatches — `attention_decode_scores_f32`
(n_blocks threadgroups per head; each score an independent sequential dot over head_dim),
`attention_decode_softmax_f32` (unchanged shape and unchanged lane->position striding, so
`simd_max`/`simd_sum` see the same partials in the same order), and
`attention_decode_context_f32` (n_blocks threadgroups per head; each output dim an
independent sequential sum over positions, consecutive threads owning consecutive dims so
the V reads coalesce). Default on, `CAMELID_METAL_ATTN_SPLIT3=0` restores the single-kernel
encode. Nothing in it is gemma3-specific: any f32-KV row that lands on the v1 fallback gets
the same, bit-identically.

**Two codegen traps, found by the test and not by reasoning.** Both are documented at the
kernels because both look like noise and are not:

1. Written the obvious way — one thread per (head, position) — the outputs differed in the
   last ulp: 829/1024 elements, max abs diff 3.05e-8. Fast-math picks its reassociation
   from the shape of the loop nest, so the kernels keep v1's nest verbatim (strided outer
   loop, scalar inner accumulation) and only widen the stride.
2. The softmax kernel publishes the DENOMINATOR and the context kernel takes `1.0 / denom`
   itself. Publishing the reciprocal instead hides the `1.0/x` from the phase-3 multiply
   that fast-math folds it against, and reproduced the identical 3.05e-8 divergence — which
   is how it was diagnosed: three structurally different rewrites all produced byte-identical
   "wrong" output, so the cause could not be the part being rewritten.

### 15e. Result

Decode, warm, two runs each; the table shows the FIRST (slower) run, confirmation run in
brackets:

| depth | before | after | speedup |
|---|---|---|---|
| ~64 | 61.8 | 73.9 [77.0] | 1.20x |
| ~262 | 34.9 | 68.1 [70.9] | 1.95x |
| ~510 | 22.0 | 60.3 [63.0] | 2.74x |
| ~1020 | 20.3 | 58.8 [58.7] | 2.90x |
| ~2030 | 17.3 | 55.7 [56.1] | 3.22x |

Prefill: 64 tokens 0.85 s -> 0.73 s (74.9 -> 87.3 tok/s); 512 13.98 -> 6.37 s (36.6 -> 80.4);
1200 45.80 -> 16.18 s (26.2 -> 74.2); **2400 112.68 -> 34.77 s (21.3 -> 69.0 tok/s, 3.24x)**.

A same-binary control with `CAMELID_METAL_ATTN_SPLIT3=0` in its own server process returns
55.1 / 32.7 / 22.2 / 20.4 / 16.6 tok/s, reproducing the pre-change binary — so the gain is
attributable to this change rather than to drift between builds.

Refitted law: **ms/token = 13.2 + 2.25e-4 * sum_l position_count_l**. Attention per
layer-position fell **10.5x** (2.372 us -> 0.225 us) and is now 3% of the token at d=64 and
8% at d=2030; the +0.3 ms constant is exactly the two extra dispatches per layer.

### 15f. Gates

- `metal_attention_decode_split3_is_bit_identical_to_v1`: raw f32 BIT equality (not a
  tolerance) across six geometries — the production 4x1x256, a windowed read with non-zero
  `kv_base_offset` (600-position cache, `window_start` 88, `position_count` 512), a
  single-position cache, a non-GQA 2x2x64, an 8x2x128, and a position count that is not a
  multiple of the 32-lane stride.
- `gemma3_real_row_resident_forward_matches_runnable_oracle` (release, F32Y+WIRE+NSG8 armed
  from the shell, 567.73 s): depth 1 argmax 108, 6.247e-5; depth 5 argmax 1077, 7.820e-5;
  depth 50 argmax 578, 9.584e-5 — **50/50 greedy tokens identical, overall max |logit diff|
  2.122e-4. Bit-for-bit the §9b / §11h / §13e / §14g record.**
- `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle` (same gates plus
  `CAMELID_METAL_RESIDENT_DECODE=1`, 21.58 s): 108 / 584 / 568 / 2364 / 1077, **5/5
  identical, overall max |logit diff| 7.820e-5. Bit-for-bit the §10b record.**
- `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean;
  `cargo test --all-targets` **exit 0 under pipefail, 1775 passed / 0 failed** (debug — the
  configuration CI runs); `scripts/check-public-scrub.sh` exit 0.

### 15g. Stopping rule, and the honest envelope

The rule this phase stopped on, stated before the work: stop when the remaining gap is
either smaller than this box's run-to-run spread, or reachable only by changing decode
output. After the split both are true.

- Attention is now 3-8% of the token. A *perfect* attention kernel is worth at most 8% from
  here, and it could not be bit-identical anyway.
- The fixed 13.2 ms is 67% of the 8.85 ms weight-traffic roofline. The remaining 4.3 ms is
  per-dispatch overhead and GEMV efficiency spread across ~470 dispatches per token; no
  single bit-identical lever addresses more than ~1% of it, and there is no profiler here to
  find one.
- Run-to-run spread on this host is 2-10%. Below ~10% nothing here is distinguishable from
  thermal drift.

**The envelope.** Decode on this row is weight-bandwidth-bound: 1.06 GB of Q8_0 weight
traffic per token against ~120 GB/s of unified memory is ~110 tok/s, and no software change
moves that number. This lane now delivers 74-77 tok/s short (67-70% of roofline) and
56 tok/s at 2k context (51%). The residual is per-dispatch overhead over a 26-layer graph
and imperfect GEMV bandwidth utilisation, both of which need GPU performance counters to
attack, and those are entitlement-gated on this machine with a confirmed-dead headless
route. **This row is done at the envelope this machine can see. Reopening it needs a
profiler, not more patience.**

### 15h. GO/NO-GO on the batched windowed prefill kernel: NO-GO

The campaign's one big kernel item (§3 Phase 6, §5.2) is declined, for four independent
reasons any one of which is sufficient:

1. **It cannot pass this phase's gate.** Batched attention plus batched GEMM changes
   reduction order versus the token-by-token GEMV path. Byte-identical decode is the stated
   pass condition, so this item is unsatisfiable under it; it would need its own parity
   envelope and a fresh external-oracle receipt (a re-run of the >=512 windowed pack), i.e.
   a Phase 4 repeat, not a Phase 6 commit.
2. **Its addressable share collapsed when the split landed.** Batching amortises the fixed
   ~13 ms/token of weight streaming. At 2400 prompt tokens that ceiling is now 2400 x 13 ms
   ~= 31 s of a **34.8 s** prefill, where before the split it was 31 s of 113 s.
3. **Cost.** 200-400 lines of new MSL needing a per-query-row LOWER mask bound that exists
   in no batched prefill kernel in the tree (the causal masks are upper-bound only), plus
   host plumbing, tuning, and new self-parity tests — the largest single item in the
   campaign against the smallest remaining multiple.
4. **It cannot be tuned here.** Tiling and occupancy work without GPU counters is guesswork,
   and the <=128 host gates it must relax are memory-safety gates over fixed per-lane arrays
   (§5.1). Relaxing those on an unprofilable kernel is precisely the risk §5.1 warns about.

### 15i. Deliberately NOT done

- **No throughput claim was published anywhere.** The row's `performance_measured` stays
  `not_claimed_...`; no README, COMPATIBILITY, STATUS, BENCHMARKS or LANE_STATUS_LEDGER
  edit was made, and no evidence bundle was cut. These numbers are a serving measurement
  taken on one host, not a receipt: promoting them is a separate deliberate act with its own
  surfaces and its own ledger regeneration, and it should be decided on rather than
  inherited from a perf commit. §14a's side effect (the frontend's
  `hasExactRowBoundedPerformanceEvidence` rejecting any value containing `not_`) is left
  intact on purpose.
- **No llama.cpp speed comparison** appears in any commit, doc or comment.
- The f16-mirror KV read, a genuine 256-dim v2/split-K attention kernel, and K/V projection
  dispatch fusion were all evaluated and declined in §15c / the plan doc, on the gate or on
  yield. The fused gate+up remains forbidden (implemented, proven bit-correct, measured to
  regress on register spill, reverted).
- The prompt-prefix-cache exclusion (§9e-5) is a real multi-turn cost and was NOT addressed:
  it is a KV-format and correctness change, not a perf commit, and the Phase 4 gates cannot
  validate it.

---

## 16. Long-prompt TTFT campaign, Phase 1 record — the parity harness (2026-07-31)

A **new campaign** starts here, on branch `feat/gemma3-batched-prefill` off `main` at
`a5945f8a`. Its target is TTFT on long prompts: prefill is token-by-token today
(`session_prefill_chunk_tokens` hard-returns `1` for any windowed arch), so a 2 400-token
prompt costs 2 400 sequential 26-layer command buffers. Its plan has two tiers — Tier A,
batched weight streaming with a raw-bit gate; Tier B, windowed attention-as-matmul at
head_dim 256.

**Phase 1 builds the gate and nothing else. No kernel work was started, and none should be
until this record's numbers are read.** The campaign's own recon found the existing windowed
evidence has near-zero power against the exact error class the coming kernels introduce, so
every downstream receipt would have been unfalsifiable. §16a explains what was wrong with
it; §16b-§16f are the replacements; §16g is what the mutation run actually found, including
the parts that should change the Tier A/B plan.

### 16a. What the existing windowed evidence could not see

Read off the committed files, not inferred:

1. **The windowed pack cycles its content.** `qa/prompt-packs/gemma3-windowed-context-pack-v1.json`
   builds its three prompts from a pool of **30 unique sentences cycled up to 8x** — 211
   sentences, 30 distinct, at N=2403. A window-boundary error moves the edge across
   verbatim-duplicate text, so the same information stays reachable from the duplicate.
2. **Its load-bearing fact is out of every window.** "Willow" occurs only at character 48,
   ~token 12. No query past position 523 has it in a 512-window at all, so at N=2403 it is
   reachable **only through the 4 global layers** — which no window mutation touches. The
   pack is a strong test of global-layer reach and almost no test of the 22 sliding layers.
3. **No length is a boundary.** 606/1205/2403 are 30/53/35 mod 64 and 94/53/35 mod 128 — not
   a tile multiple, not a tile edge, not a window edge.
4. **The windowed sample is 31 content tokens**, not 9 legs: depths 1/5/50 are strict
   prefixes of one greedy continuation, 14/8/9 content tokens across the three prompts.
5. **Token identity was measured on re-encoded text**, not on the engine's ids
   (`scripts/chat-parity-gemma3.mjs`), which is lossy in both directions on this 262k SPM
   vocab — §16d.
6. **No margins were recorded**, so the sensitivity of a passing leg could not be estimated
   afterwards.

The measured dead zone this leaves: reduction-order noise on this row is **2.122e-4** max
|logit diff| (§9), and the smallest observed argmax flip sits at a **0.0032-nat** top-2 gap
(§13). A perturbation between those bounds is invisible to any argmax-only gate.

### 16b. The window convention, pinned once — `src/window_ref.rs`

Three sites encoded `[max(0, p+1-w) ..= p]` separately (the Metal resident decode encode,
the gemma4 CPU runtime, and the doc on `Gemma3Metadata::layer_window`), and **two of them
cited a `src/model.rs is_position_visible` that does not exist**. `window_bounds()` is now
the single reference, and its unit tests assert it against verbatim re-statements of the
other two expressions over 0..2600 positions and six window values.

`is_visible(query, key, window)` is written as `key + window > query`, never
`key >= query - window`. The subtraction underflows for `query < window` on unsigned types —
exactly the bug class that appears when the bound moves from host Rust (`saturating_sub`)
into MSL, which has no such thing. The `>` rather than `>=` is where the
INCLUDES-current-position convention lives: at `key = query + 1 - window` the predicate is
`query + 1 > query` (visible, the oldest in-window position), and at `key = query - window`
it is `query > query` (hidden, the first position outside).

### 16c. The KV-equivalence invariant — `src/kv_equivalence.rs` + `ResidentDecodeState::kv_snapshot`

End-to-end token identity is the right *outer* gate and a poor *inner* one: one argmax per
step, after 26 layers of mixing. The direct claim is cheaper and far sharper — **a batched
prefill of n tokens must leave the KV cache in the same state as n token-by-token
forwards** — a per-(layer, position, head, dim) comparison with no softmax between the
defect and the observable.

- **Tier A** asserts `assert_bit_identical`. Batching over the token dimension re-associates
  no reduction, so there is no tolerance to negotiate; a differing bit is a bug.
- **Tier B** asserts `meets_bound(published bound, outlier factor)`. The scalar bound alone
  is satisfiable by a kernel that is uniformly slightly wrong; the **outlier half** is what
  has power — a uniform small delta is reduction noise, one position 10x above the
  per-position median is a mask or stride defect.

Two properties decide where a defect becomes visible, and both are asserted rather than
assumed:

- **Layer 0's cache is mask-independent.** Its K/V come from the token embedding, never from
  attention, so a mask defect first appears in layer 1's cache. A RoPE defect, by contrast,
  *must* move layer 0 — K is rotated before the cache write — and the synthetic gate asserts
  that direction too.
- **The last layer's attention reaches no KV cache at all.** A defect confined to it moves
  **zero** cache elements. This is not hypothetical: building the gate produced exactly that
  shape (§16e), and it found a hole in the invariant's own bound check, which is the whole
  argument for writing the harness before the kernels.

### 16d. The chat-parity harness now reads the engine's ids

`scripts/chat-parity-gemma3.mjs` computed camelid's "tokens" by re-encoding camelid's output
**string** with camelid's tokenizer. That round-trip is lossy in both directions:

- it **manufactured** divergences — a run of spaces re-encodes as single-space tokens
  (236743) where the engine emitted the merged whitespace tokens (138, 140), which is why
  the Phase 4 bundle had to add `camelid-raw-probe.json` to unpick the artifact;
- it **masked** them — any two id sequences that decode to the same string, or that the
  tokenizer re-merges to the same canonical ids, compare EQUAL. A batched-prefill defect
  that changes an id without changing the rendered text is invisible to it. That is the
  defect class this campaign exists to catch.

`token_match` is now scored on `camelid.generated_token_ids`, with trailing EOG ids (1, 106)
stripped identically from both sides. A lane returning no diagnostics block makes the harness
**throw**, not fall back. The old re-encode survives as `text_reencode_token_match`, reported
and never scored, with a `text_reencode_artifact` flag that fires when the ids agree and the
re-encode does not. `--top-logprobs N` (default 0, i.e. the request shape the frozen bundles
used) records per-position top-2 margins on both engines.

### 16e. The synthetic gate — and the hole it found in the invariant

`metal_kv_snapshot_equivalence_catches_window_and_rope_mutations` runs in the DEFAULT suite:
a 3-layer synthetic fixture, no GGUF, no model load. It establishes a positive control (two
independent sessions, same schedule, same inputs -> bit-identical KV and equal digest) and
then requires six schedule mutations to turn the invariant red. Measured, all six caught:

| mutant | differing elements | kv max abs | hidden max abs | first difference |
|---|---:|---:|---:|---|
| `window_minus_one` | 7 296 / 13 824 | 1.083e1 | 4.073e0 | K layer 1 position 3 |
| `window_plus_one` | 6 272 / 13 824 | 4.550e0 | 4.930e-1 | K layer 1 position 4 |
| `no_lower_bound` (full causal) | 6 272 / 13 824 | 1.020e1 | 3.260e0 | K layer 1 position 4 |
| `window_on_all_layers` | **1 152 / 13 824** | **0.000e0** | 2.356e0 | **final_hidden[0]** |
| `window_on_the_wrong_layers` | 6 272 / 13 824 | 1.020e1 | 4.668e0 | K layer 1 position 4 |
| `rope_tables_swapped` | 11 376 / 13 824 | 2.083e0 | 4.315e-1 | K layer **0** position 1 |

Two rows carry findings:

- **`window_on_all_layers` moves ZERO cache bits.** The mutation is confined to the last
  layer, whose attention output projects no K/V, so every one of its 1 152 differing
  elements is in the final hidden. An earlier draft of `meets_bound` checked only the caches
  and **passed this mutant**. It now bounds the final hidden separately. Tier A/B harnesses
  must therefore always capture a final-hidden (or logits) vector alongside the caches, or
  a last-layer mask defect ships silently.
- **`rope_tables_swapped` is the only mutant that reaches layer 0**, because RoPE is applied
  to K *before* the cache write. The mask-only mutants are asserted to leave layer 0 exactly
  untouched, and do.

The gate also asserts the invariant's **blind spot**, so it stays documented rather than
being rediscovered: with `filled <= window`, widening the window (or dropping it entirely) is
a **bit-exact no-op**. Prompts must exceed the window to have any power over its bound.

### 16f. The new pack — `qa/prompt-packs/gemma3-window-edge-pack-v1.json`

24 items, built by the committed generator `scripts/build-gemma3-window-edge-pack.mjs`
against the pinned oracle's tokenizer (`/tokenize`, `add_special=true`), which is what makes
the token positions *measured* rather than estimated. Every item carries its
`prompt_token_ids`, so the in-src mutation harness needs no tokenizer and the positions are
checkable without re-running the generator.

| item | tokens | anchors (offset from q) | window power |
|---|---:|---|---|
| `w-edge-q-510` | 1 024 | q-510 `crimson` (inside) | high |
| `w-edge-q-511` | 1 024 | q-511 `amber` — the OLDEST visible position | high |
| `w-edge-q-512` | 1 024 | q-512 `indigo` — the FIRST invisible position | high |
| `w-edge-q-513` | 1 024 | q-513 `emerald` (outside, control) | high |
| `w-multi-1536` | 1 536 | q-1024, q-552, q-511, q-64 | high |
| `w-multi-2400` | 2 400 | q-2048, q-1024, q-552, q-511, q-64 | high |
| `w-len-{63,64,65}` | 63/64/65 | — | **none** |
| `w-len-{127,128,129}` | 127/128/129 | — | **none** |
| `w-len-{255,256,257}` | 255/256/257 | — | **none** |
| `w-len-511` | 511 | — | **none** |
| `w-len-{512,513}` | 512/513 | — | minimal |
| `w-len-{1023,1024,1025}` | 1023/1024/1025 | q-511 | high |
| `w-len-{2400,2432,2433}` | 2400/2432/2433 | q-511 | high |

Design points, each answering a specific defect in §16a:

- **Content is non-repeating and it is asserted, not assumed.** No body sentence appears
  twice anywhere in the pack; no anchor gate or colour is reused. The generator filters the
  anchor vocabulary against every filler word list, which caught `copper` clashing with the
  filler place "copper works" — a hand audit had already missed it.
- **The four `w-edge-*` items are identical in construction and differ only in anchor
  placement**, so a change in answer between them is attributable to the mask and nothing
  else. Their answer word occurs exactly once per prompt, enforced at build time.
- **Lengths are exact** and sit on the 512-window, the 64-wide NR0/NR1 attention tiles, and
  the `n_pad = next_multiple_of(128)` boundary (2432 is n_pad; 2433 pushes n_pad to 2560).

Three limits are recorded in the pack itself rather than discovered later:

1. **`window_power: none` below 512 tokens is a fact about the arithmetic, not a weakness of
   the content.** Below saturation `filled.saturating_sub(w)` is 0 for every `w`. Those nine
   items exist for the Tier B TILE geometry and each says so.
2. **At N=512 and N=513 the window edge lands on the TEMPLATE PREFIX, not on text** — the
   position that moves in or out is 0 (BOS) and 1 (`<start_of_turn>`). No content anchor is
   possible there, so those two items probe the attention-sink tokens instead.
3. **A q-511/q-512 anchor PAIR inside one prompt is geometrically impossible**: an anchor
   sentence is ~9 tokens long. Consecutive anchors in the multi-depth items are kept >=40
   tokens apart, and the exact one-token pair is carried by the dedicated single-anchor
   items. This is why there are four `w-edge-*` items rather than one.

**And the honest framing of what the pack proves.** A sliding-window model can still recall a
fact from outside a single layer's window by STACKING — 22 sliding layers give an effective
receptive field of ~22x511 positions. The pack therefore does **not** claim that a correct
implementation fails to recall a q-512 fact, and any harness built on that assumption would
be wrong. Its power comes from **output sensitivity to the mask**: the token stream must
change when the bound moves, scored against a reference run, never against a notion of the
right answer.

### 16g. The mutation run — and the finding that changes the Tier A/B plan

`gemma3_real_row_window_mutation_harness` on the real row, 9-item mutation subset, 12 greedy
tokens per leg, 8 schedules (production + 7 defects), 1 761 s in one process. Positive
control first: two baseline runs of `w-len-513` gave identical tokens and **bit-identical KV**
(digest `9c51bfa9b0c9eef9`).

The defects are applied as `ResidentLayerSchedule` perturbations, not as scratch edits to the
window arithmetic, and that is exact rather than convenient: `window_start =
filled.saturating_sub(w)` means a schedule of `w-1` **is** "window_start off by +1 wherever
the window is saturated", and `w+1` is the off-by-one the other way. The tree is left clean.

**All seven mutants caught — `survivors` is empty — and the gate PASSES.** Which observable
caught them is the result:

| mutant | caught by TOKEN identity | caught by KV equivalence |
|---|---:|---:|
| `window_minus_one` (w-1) | **0 / 9** | 9 / 9 |
| `window_plus_one` (w+1) | **0 / 9** | 9 / 9 |
| `window_on_all_layers` | 6 / 9 | 9 / 9 |
| `layer_pattern_shift_by_one` | 8 / 9 | 9 / 9 |
| `no_lower_bound` (full causal) | 9 / 9 | 9 / 9 |
| `window_on_wrong_layers` | 9 / 9 | 9 / 9 |
| `rope_tables_swapped` | 9 / 9 | 9 / 9 |

**A one-position window error changed not a single generated token on any of the nine items —
including the four built specifically to make it visible, at 1 024 tokens with the answer word
planted at exactly q-510 / q-511 / q-512 / q-513.** It is invisible to argmax at every length
tested, 513 through 2 400.

Its KV signature, by contrast, is unmistakable. Per item, for the two off-by-one mutants:

| item | tokens | max abs ΔKV (w-1 / w+1) | per-position median (w-1 / w+1) | differing elements |
|---|---:|---|---|---|
| `w-edge-q-510` | 1 024 | 1.157 / 3.419 | 9.87e-3 / 6.37e-3 | ~6.8M / 13.9M |
| `w-edge-q-511` | 1 024 | 0.996 / 8.598 | 1.42e-2 / 5.64e-3 | ~6.8M / 13.9M |
| `w-edge-q-512` | 1 024 | 2.140 / 0.848 | 1.14e-2 / 6.06e-3 | ~6.8M / 13.9M |
| `w-edge-q-513` | 1 024 | 0.771 / 1.002 | 1.37e-2 / 5.25e-3 | ~6.8M / 13.9M |
| `w-multi-1536` | 1 536 | 11.502 / 10.885 | 3.50e-2 / 3.87e-2 | ~13.4M / 20.7M |
| `w-multi-2400` | 2 400 | 3.107 / 1.788 | 4.19e-2 / 4.37e-2 | ~24.4M / 32.2M |
| `w-len-513` | 513 | 0.583 / 0.429 | **0.0 / 0.0** | 287 743 / 7.1M |
| `w-len-1024` | 1 024 | 1.637 / 1.416 | 1.33e-2 / 2.80e-3 | ~6.8M / 13.9M |
| `w-len-2400` | 2 400 | 4.414 / 24.357 | 3.01e-2 / 3.33e-2 | ~24.4M / 32.2M |

Against a reduction-noise floor of 2.122e-4, every one of those is 3-5 orders of magnitude
clear. `w-len-513` is the sharpest case for the **outlier** half of the Tier B bound: its
per-position median is exactly **0.0** (only 1-2 of 513 query positions clip, so the median
across positions is zero) while 287 743 elements differ. A scalar bound alone would have to be
set absurdly tight to see that; the per-position outlier test sees it immediately.

A consistency check the data produced without being asked: at N=513, `window_plus_one` and
`no_lower_bound` give **identical** KV numbers (274 944 differing, max 4.2887e-1, hidden
1.4681). They must — a 513-wide window never clips 513 positions, so w+1 *is* full causal
there.

**Consequences, and they are not small:**

1. **Gate G9 (external oracle, token identity) cannot carry Tier B on its own.** It has
   measured zero power against the campaign's headline defect. It stays as the outer gate;
   it is not the gate.
2. **G1/G6 (KV equivalence) are promoted from "cheap direct check" to MANDATORY for both
   tiers.** Tier A already had a raw-bit gate; Tier B must publish a bound AND the
   per-position outlier factor, and the outlier half is the half with the power.
3. **A Tier B harness must capture the first-token logits (or a final hidden) alongside the
   caches.** The last layer's attention reaches no KV cache; §16e showed a mutant that moves
   zero cache bits and is visible only there.
4. **The pack's own high-power items did not save token identity.** Planting the answer word
   at exactly q-511 and q-512 was the strongest content design available and it still did not
   flip an argmax. This is not a defect of the pack — it is the measured ceiling of what
   argmax observation can do on this row, and it should be quoted whenever someone proposes
   token identity as a sufficient gate.

### 16h. Baseline on the current (token-by-token) lane

`scripts/chat-parity-gemma3.mjs` against the Metal GPU-resident serve lane, new pack, depths
1/5/50, margins armed. Bundle
`qa/evidence-bundles/gemma3-1b-q8-window-edge-harness-20260731-head-a82dd41a/`.

- **Cross-engine prompt tokenization identical 24/24** — the pack's own `prompt_tokens`
  matched the oracle's `/tokenize` on every item, so the exact lengths are real.
- **70/72 generation legs token-AND-text identical**, `all_pass: false`.
- **48/48 depth-1 and depth-5 legs clean.** Both failures are depth-50.
- **All six anchored window items clean at every depth**, per-leg minimum top-2 margins
  3.45-7.81 nats. Those are the items the window gates rest on.
- The two failures, disclosed with their margins at the divergence position:
  `w-len-256` depth 50, index 13, camelid gap **0.468** nat / oracle 0.235 nat; `w-len-513`
  depth 50, index 5, camelid gap **0.0696** nat / oracle 0.314 nat. Both are unanchored
  ladder items carrying the open-ended "name one item mentioned above" question, both sit
  inside the near-tie band this row already discloses (§13: flips at 0.0032 / 0.0173 /
  0.0353 nat, largest disclosed oracle-side gap 0.447), and neither is a window item.
  Neither is excused: the receipt ships with `all_pass: false`.

**One negative finding worth keeping.** `text_reencode_artifact` fired on **0/72** legs, and
the old text round-trip disagreed with the engine ids on 0 legs where the ids matched. On
this pack the fixed harness reaches the same verdict the old one would have. The fix is a
*power* fix for the defect class ahead — an id change that does not change the rendered text
— not a correction of a wrong result here, and it should not be advertised as one.

### 16i. Gates and what was deliberately not done

`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --all-targets`
and `scripts/check-public-scrub.sh` all clean. The two real-row tests are env-gated and skip
in the default suite; the synthetic KV-equivalence gate runs in it.

Deliberately not done:

- **No kernel work.** Not one line of MSL, no admission-predicate change, no env flag. Phase
  1 is the harness.
- **No pack item was softened after seeing a result.** The two failing depth-50 legs stay in
  the pack and in the receipt with their margins.
- **No re-run of the mutation harness to tidy a denominator.** `differing_elements` in the
  committed receipt is counted against a caches-only `compared_elements`; the code was
  corrected afterwards so future receipts share one denominator, and the bundle README says
  so rather than the numbers being regenerated to look neater.
- **No promotion of any surface.** No README, COMPATIBILITY, STATUS or ledger edit; the row's
  claims are unchanged. This phase adds a gate, not a claim.

### 16j. Handover to Phase 2

The GO condition on Tier B ("Phase 1's harness detects all five mutants") is **met, with all
seven**, but the plan's gate list must be amended by §16g before Phase 2 starts:

- G1 (raw `to_bits` KV equality, batched vs token-by-token, over the length sweep) is
  implementable today: `ResidentDecodeState::kv_snapshot` + `kv_equivalence::compare` +
  `assert_bit_identical`. Tier A should assert it at the pack's exact lengths, which now
  include the tile and n_pad edges the old evidence never touched.
- G6's bound must be published **with** its outlier factor, and the harness must capture the
  first-token logits.
- G9 stays, and its measured limitation is now on record.
- The prompt-prefix cache remains closed on this row, so nothing here interacts with it.

## 17. Long-prompt TTFT campaign, Phase 2 record — Tier A, batched weight streaming (2026-07-31)

Branch `feat/gemma3-batched-prefill`, off `main` at `a5945f8a`. Phase 2 builds Tier A: batch
the weight-streaming half of prefill, opt-in, and hold it to raw bit-identity against the
shipped token-by-token lane. **The gate passed on the first run and every campaign gate stayed
green. The performance thesis did not: the measured win is 1.07-1.13x, against the plan's
projected 2.0-3.6x, and §17e explains why with a mechanism rather than a shrug.** Read §17e
before planning Tier B — it changes what the remaining prize is.

### 17a. What shipped

Everything is behind `CAMELID_GEMMA3_BATCH_PREFILL`, **default OFF this phase**. A non-gemma3
row cannot reach it: the arming predicate is a conjunction with `config.gemma3.is_some()`.

| piece | where |
|---|---|
| `prefill_tokens_windowed` / `_hidden` / `_inner` | `src/metal.rs` |
| `schedule_window_bounds` — the window expression, written once, shared with `prepare_token` | `src/metal.rs` |
| `gemma3_batch_prefill_enabled` / `gemma3_batch_prefill_rows` | `src/metal.rs` |
| `gemma3_batched_prefill_armed`, dual arming gate, per-position dual-theta tables, G11 assert | `src/inference/metal_resident.rs` |

The shape is `verify_batch_inner`'s, not `prefill_tokens`'. Inside a chunk, every stage whose
output depends on one row runs the EXACT single-token kernel at a row byte offset (per-head
QK-norm, RoPE, K/V scatter, attention); every stage that streams a weight matrix runs the
proven byte-exact batched twin (`rms_norm_batch_f32`, the C0 batched-column GEMV); the
elementwise stages run the same kernels with a wider `n`.

### 17b. The four blockers, and how each was handled

`prefill_tokens` refuses gemma3 on four independent gates. The campaign plan's §2.5 is
confirmed: head_dim is only one of them, and relaxing it alone gets nothing.

1. **head_dim > 128.** NOT relaxed. `MAX_DPL` / `MAX_DOCT` and the `<= 128` host gates on
   `prefill_tokens`, `verify_batch` and the split-K route are untouched — they are
   memory-safety gates shared by eleven kernels. Tier A is a *separate admission path* with no
   fixed per-lane array anywhere in its chain, so 256 is admissible there and nowhere else.
2. **The dual-theta schedule.** Carried: the ALT/primary table pair is selected per layer from
   `schedule.use_alt_rope[l]`, mirroring `prepare_token` verbatim, and the caller must supply
   the ALT tables for every position or the path fails closed. The tables are built per
   position with `rope::gemma3_rope_tables` for BOTH thetas — the runnable-oracle frequency
   form the decode path uses, NOT the generic negated-exponent form the existing prefill
   builds. That was not cosmetic: the generic form's last-ULP drift would have shown up
   directly as a G1 failure.
3. **The sandwich norms.** Carried as two extra `rms_norm_batch_f32` dispatches per layer,
   applied to the attention and FFN outputs BEFORE their residual adds, exactly as
   `encode_attention_block` / `encode_ffn_block` do.
4. **GeGLU.** Carried by dispatching the same `gelu_mul_f32` the decode encode uses with
   `n = rows * ffn_dim`. Zero new MSL, as the plan predicted.

**`use_attn_mm` is not on this path at all.** Its `!has_qk_norm` clause would refuse this
QK-norm arch, and its `head_dim <= 128` clause would refuse this row — but attention-as-matmul
is Tier B, and Tier A routes every row's attention through the unchanged `encode_attention`,
which at head_dim 256 lands on `encode_attention_split3`, the exact kernel the shipped lane
already runs. The sliding-window mask therefore needs no new kernel and no new MSL: the
per-row `(window_start, position_count)` ride in through the attention scalar's
`kv_base_offset` word, the same mechanism the token-by-token lane uses.

**The window expression is now written once.** Three sites used to encode it separately.
`schedule_window_bounds` is the shipped `filled.saturating_sub(w)` verbatim — deliberately NOT
a call to `window_ref::window_bounds`, because the two disagree on the degenerate `Some(0)`,
and bit-identity with the shipped lane is the gate. A test pins their agreement for every
`w > 0` over 2 600 positions and records the one input where they differ.

**G11 is asserted, not assumed.** `filled()` must equal `n` after the prefill or the caller
declines and falls back losslessly. A short `filled` would re-seed from a CPU KV cache this
lane deliberately leaves hollow, decline at `history_materialized`, and drop the whole prompt
onto a CPU path that fails closed for windowed archs.

### 17c. Gate G1 — bit-identity, measured

Both sides are fed the SAME embedding array, so an embedding-lookup difference can neither
mask nor manufacture a result. The final hidden is compared alongside the caches, because the
last layer's attention reaches no KV cache at all (§16e found a mutant that moves zero cache
bits).

**Real row** (`gemma3_real_row_batched_prefill_kv_bit_identical`), 26 layers, head_dim 256,
the shipped 22-sliding/4-global schedule, chunk 256:

| n | elements compared | bit-identical |
|---:|---:|---|
| 5 | 72 320 | **yes** |
| 256 | 3 702 784 | **yes** |
| 257 | 3 717 248 | **yes** |
| 513 | 7 420 032 | **yes** |
| 1 024 | 14 811 136 | **yes** |

**Synthetic** (`metal_batched_windowed_prefill_is_bit_identical_to_token_by_token`), 4 layers,
head_dim 256, chunk 4, window 6, at n = 1/3/4/5/6/7/8/9/13/17 — chunk edges, exact multiples,
ragged tails and both sides of window saturation. Bit-identical at every length, for the
gemma3 shape (dual-theta + sandwich norms + GeGLU + QK-norm) **and** for a plain Llama shape
run through the same entry point.

No tolerance was negotiated and none was needed. Batching over the token dimension
re-associates no reduction, so this was the expected outcome; it is recorded because the
campaign's stopping rule 3 says a G1 failure is a bug to fix, not a bar to lower.

### 17d. The other campaign gates — all unchanged

| gate | result |
|---|---|
| `gemma3_real_row_resident_forward_matches_runnable_oracle` | 50/50 greedy tokens identical, overall max abs logit diff **2.122e-4** — the pinned constant, unchanged |
| `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle` | 5/5 identical, **7.820e-5** — unchanged |
| `gemma3_real_row_window_mutation_harness` (G7) | all **7/7** mutants caught, re-run with Tier A in the tree (1 683 s, one process). Per-mutant token/KV: `window_minus_one` 0/9 and 9/9, `window_plus_one` 0/9 and 9/9, `no_lower_bound` 9/9 and 9/9, `window_on_all_layers` 6/9 and 9/9, `window_on_wrong_layers` 9/9 and 9/9, `layer_pattern_shift_by_one` 8/9 and 9/9, `rope_tables_swapped` 9/9 and 9/9 — reproducing §16g exactly, including the finding that a one-position window error changes NO generated token on any of the nine items |
| window-edge pack vs pinned llama.cpp `acd79d603`, **flag ON** | **70/72** legs token-AND-text identical, prompt tokenization 24/24 — the SAME two pre-existing depth-50 failures (`w-len-256` at generated index 13, `w-len-513` at index 5), same indices, same margins. No regression, and the two are not "fixed" |
| `cargo fmt --check`, `clippy --all-targets -D warnings`, `cargo test --all-targets`, `scripts/check-public-scrub.sh` | clean |

Non-gemma3 archs see zero behaviour change, asserted rather than argued:
`batched_windowed_prefill_never_arms_for_a_non_gemma3_arch` holds in EITHER state of the
(process-latched) flag, and the synthetic G1 runs a Llama-shaped fixture through the new entry
point to show the added machinery is inert when every `Option` is `None`.

### 17e. THE MEASUREMENT — and why Tier A's thesis does not survive it

Same binary, flag on vs off, one server at a time, streamed SSE timing request-start to first
content token, exact token-id prompts (tokenizer out of the loop), a distinct prompt window per
request so no two share a prefix, `prompt_cache_hit false` on all 32. Round 0 is reported as
the cold column and excluded from the mean; the mean is rounds 1-3.

| N | flag OFF warm mean | sd | flag ON warm mean | sd | speedup | load (off / on) |
|---:|---:|---:|---:|---:|---:|---|
| 600 | 7.772 s | 0.116 | **7.231 s** | 0.105 | **1.07x** | 3.67-4.53 / 3.25-3.92 |
| 1 200 | 16.314 s | 0.148 | **14.952 s** | 0.361 | **1.09x** | 3.72-4.41 / 3.05-4.17 |
| 2 366 | 34.138 s | 0.218 | **30.604 s** | 0.180 | **1.12x** | 3.95-4.24 / 3.43-4.44 |
| 2 400 | 35.162 s | 0.602 | **31.232 s** | 0.270 | **1.13x** | 3.88-4.41 / 3.30-3.76 |

Cold (round 0) columns: off 8.070 / 16.624 / 34.266 / 35.065; on 7.119 / 14.630 / 30.363 /
30.801. The flag-off column reproduces Phase 0's warm baselines (8.222 / 17.480 / 35.631) 4-7 %
faster, consistent with the lower load this session ran under. **No first-request PSO-compile
penalty appeared** — the Phase 3 warning the plan raised did not materialise, because the
server's warm-up decode and the first chunk retire the compile before any measured request.

**The plan projected 12.1-17.2 s at 2 400 tokens. The measurement is 31.2 s.**

Three receipts explain it, and together they are a mechanism, not a hedge.

**(1) The chunk width does not matter — at all.** `gemma3_real_row_batched_prefill_chunk_width_probe`,
n = 1 200, one process, warm weight cache, the token-by-token baseline with encode-ahead armed
(the lane as it actually runs, not a handicapped one):

| rows per command buffer | 16 | 32 | 64 | 128 | 256 | 512 | token-by-token |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms/token | 11.99 | 12.08 | 12.27 | 12.36 | 12.43 | 12.59 | 13.73 |

A 5 % spread across a 32x range of chunk widths, and *wider is slightly worse*. Whatever Tier A
is spending its time on is invariant in the batch width — which is the opposite of what
"amortize the weight stream over the chunk" predicts.

**(2) The GPU is 99 % busy; the CPU is not the problem.** From the batched path's own
per-command-buffer trace on the serve lane (`CAMELID_RESIDENT_TRACE=1`, 256-row chunks):
`encode` ~10 ms, `commit_wait` 2.91-3.21 s, `gpu_busy` 2.89-3.20 s. CPU-side encode is
**0.04 ms/token** of a 12 ms/token cost, and `commit_wait - gpu_busy` is ~0.01 ms/token. The
time is inside the kernels. Per-row `gpu_busy` runs 11.3 ms at chunk base 0 and 12.6 ms at base
768, so attention accounts for ~1.3 ms across that depth range and the depth-0 floor is
~11.3 ms/token.

**(3) The reason is in the kernel, and it is a constant.**
`q8_0_block_linear_ksplit_f32y_wire_nsg8_verify` carries `constexpr uint MAX_T = 8;` and an
outer `for (uint t0 = 0; t0 < n_rows_in; t0 += MAX_T)` loop. **Every weight block is re-read
once per 8-column tile.** The effective batching factor is 8, hard-coded, no matter how wide
the chunk is — which is exactly why (1) is flat. Per token, for any chunk >= 8:

- weight traffic = (weight bytes) / 8 = 0.74 GB / 8 = **93 MB/token** (was 741 MB/token);
- activation traffic = `sum over projections of (out_rows/2) * in_dim * 4` = **1.40 GB/token**,
  because this is a GEMV: every threadgroup walks the whole activation panel, and that is
  chunk-invariant too.

So Tier A trades 0.65 GB/token of weight traffic for nothing it can avoid, and inherits a
1.40 GB/token activation re-read that the single-token lane also pays but gets for free — its
panel is 4.6 KB and lives in cache, while a 256-row FFN panel is 7.1 MB and does not. The
combined 1.49 GB/token at this host's ~120 GB/s is 12.4 ms, which brackets the measured
11.3-12.6 ms. Measured GEMM throughput on this path is **1.396 GFLOP/token / 11.3 ms =
0.12 TFLOPS**, against the 3.4 TFLOPS Q8 GEMM wall this host is measured at (§15) and against
the 0.8-1.5 TFLOPS the plan credited the batched-column GEMV. That 10x gap is the whole
shortfall.

Honest limit on this attribution: separating "DRAM-bandwidth-bound on the activation re-read"
from "issue-bound in the inner loop" needs GPU performance counters, which are
entitlement-gated on this host with a confirmed-dead headless route (§15). The chunk-invariance
and the `MAX_T = 8` constant are read off the code and the measurements; the split between
those two causes is not.

### 17f. What this changes for Tier B — read this before Phase 4

1. **The plan's 77-86 % attribution to weight re-streaming is refuted by measurement.** Removing
   essentially all of it (741 -> 93 MB/token) bought 1.1x, not 2-3x. The residual is not
   dispatch overhead either (§17e(2): 0.04 ms/token of encode). Phase 0's own note — that the
   depth-0 `gpu_busy` floor was 9.8 ms against a 6.18 ms weight-traffic roofline — was already
   pointing at this and was read too optimistically.
2. **Tier B's headline is now smaller than it looked.** On the batched path, per-row `gpu_busy`
   grows 11.3 -> 12.6 ms across chunk bases 0 -> 768, i.e. attention adds ~1.3 ms over that
   depth range; the campaign's own attention law puts the TOTAL attention cost of a
   2 400-token prompt at 35.68 M layer-positions x 1.789e-4 ms = **6.4 s of the measured
   31.2 s**. Collapsing attention to zero would leave ~25 s. **Tier B as specified cannot reach
   the plan's 1.3-2.4 s, or anything near it.** The
   campaign's stopping rule 5 ("Phase 4's kernel must beat Tier A's per-row attention by >= 3x
   on wall clock") is not the binding question any more — the binding question is the GEMM.
3. **The remaining prize is the GEMM, and it is where bit-identity has to be spent.** The tiled
   simdgroup-MM path (`half_mm_batched_f16o`, staged panels, fixed 8 KiB threadgroup memory) is
   what reaches ~3 TFLOPS on this host for other rows, and the tree is explicit that it is
   "numerically equivalent to the scalar k-split GEMM but not byte-exact: tile MMA accumulation
   order". **So the campaign must now choose: bit-identity, or the speed.** It cannot have both
   with the current kernel set. That choice belongs to the user, not to this phase.
4. **There is a bit-identical middle option, and it is a kernel rewrite, not a bolt-on.** A
   batched GEMV that stages the activation panel in threadgroup memory and reuses it across
   output rows keeps every output element's reduction tree intact — same `sumq` then `* w_scale`
   ordering, same two-stage `simd_sum` — so it would stay bit-identical while removing the
   1.40 GB/token re-read. Raising `MAX_T` alone is NOT that fix: it is bit-identical and cheap,
   but it can only recover the 93 MB/token weight term (~0.8 ms/token, ~7 %), and
   `sumf[NR0][MAX_T] + yl[MAX_T][NQ]` is `10 * MAX_T` live floats per lane, so 16 lands at 160 —
   the same register-spill cliff the v3 attention kernel already records at 88.
5. **Default stays OFF.** Phase 3 was to flip it after receipts. On a 1.07-1.13x win with a
   bit-identity guarantee the flip is defensible but not obviously worth the surface; the
   recommendation is to hold the flip until the GEMM question in (3) is decided, and to run
   Phase 3 as "decide the GEMM", not "flip the flag".

### 17g. Deliberately not done

- **No kernel MSL was written or changed.** Tier A is host wiring plus existing kernels; that
  was the charter and it held.
- **`MAX_T` was not raised.** It is a one-line change with a real register-pressure risk and a
  ~7 % ceiling (§17f(4)); doing it inside a phase whose gate is bit-identity, without counters
  to see the spill, would be guessing. Recorded as an available lever.
- **The 68 MB of dead f16 KV mirrors at head_dim 256 was not reclaimed.** The plan lists it as
  a free memory win, and it is, but the batched prefill dual-writes them exactly as the decode
  encode does — which is what keeps the two lanes' cache state identical. Removing the write
  would be invisible to G1 (the snapshot reads the f32 primary) and is therefore exactly the
  kind of change that should not ride along inside a bit-identity phase. Still owed.
- **No pack item was softened and no failing leg was excused.** The two depth-50 failures are
  the Phase 1 ones, unchanged, and are reported as failures.
- **No promotion of any surface.** No README, COMPATIBILITY, STATUS or ledger edit. This phase
  adds a lane behind a flag, not a claim.

## 18. Long-prompt TTFT campaign, Phase 3 record — the prefill GEMM (2026-07-31)

Branch `feat/gemma3-batched-prefill`, continuing from §17. Phase 2 ended with a mechanism, not
a mystery: Tier A's batched path is pinned at ~0.12 TFLOPS because
`q8_0_block_linear_ksplit_f32y_wire_nsg8_verify` carries `constexpr uint MAX_T = 8`, so every
weight block is re-read once per 8-column tile and the activation panel is re-walked by every
threadgroup. §17f(3) put the choice to the user: bit-identity, or the speed.

**The user chose the speed.** Phase 3 replaces the batched-prefill GEMV with a tiled
simdgroup-matmul path. Bit-identity against today's prefill output is explicitly no longer the
bar for the prefill GEMMs; correctness is carried by the Phase 1 harness — KV-cache equivalence
with a published envelope, the 7-mutant harness, and external-oracle token parity — which is
the standard the shipped lane already meets. Decode is untouched and stays bit-exact.

### 18a. THE ENVELOPE, PINNED BEFORE ANY MEASUREMENT

This section was committed **before** the first comparison was run; `git log` is the receipt.
Pinning after seeing the number is how a bound becomes a rubber stamp.

```
kv_equivalence::meets_bound(bound = 5.0e-2, outlier_factor = 8.0)
```

applied to the whole `KvSnapshot` — every layer's K and V at every prompt position **and** the
final hidden — of the MM prefill against `n` sequential `forward_token` prefill decodes.

**Why 5.0e-2 and not the 2.122e-4 reduction-noise floor.** 2.122e-4 is the max |logit diff| a
change that only re-associates f32 reductions produces on this row (§9). Phase 3 is strictly
larger than that by construction: the tiled kernel stages the dequantized Q8_0 weight
(`half(float(q) * w_scale)`) and the activation panel in **half** before the MMA. Half
round-to-nearest is 2^-11 = 4.88e-4 *relative*, per element, before any reduction — so a bound
at the f32-reduction-noise level is known-unreachable by arithmetic, and pinning there would be
pinning a gate that must fail. The framing that matters: the weights are already Q8_0, whose
quantization step is ~1/127 = 7.9e-3 relative, so representing `q * w_scale` in half adds an
error ~16x *below* the quantization error both paths already carry.

**Why 5.0e-2 is still a gate and not a rubber stamp.** The weakest window-mutation signature
this campaign has ever recorded is max |ΔKV| **4.29e-1** (`w-len-513`, `window_plus_one`, §16g),
and the next weakest is 5.83e-1 (`w-len-513`, `window_minus_one`). 5.0e-2 sits **8.6x below the
weakest recorded mutant**, so the scalar half alone still separates numerics from every window
defect on record by an order of magnitude. A bound loose enough to admit 4.29e-1 would not be a
gate; this one is not that.

**Why the outlier factor is 8.0.** §16g's finding is that the scalar half is the weak half: the
sharpest mutant (`w-len-513`) has a per-position **median of exactly 0.0** with 287 743 elements
differing, where any finite factor fires. Across the mutants that do have a non-zero median, the
ratio (max |ΔKV| ÷ per-position median) runs 41x - 1525x; the **weakest is 41x**
(`w-multi-2400`, `window_plus_one`: 1.788 / 4.37e-2). 8.0 leaves a **5.1x margin below the
weakest recorded ratio** while still allowing a genuine 8x per-position spread for a uniform
numerics change — which one must allow, because a position early in the prompt legitimately
carries less accumulated round-off than a deep one.

**What the pin commits to.** If the measurement exceeds either half, that is a Phase 3 failure
reported as a failure, with the measured numbers, not a bound quietly moved to fit. The decode
gates are unchanged and unmoved: `gemma3_real_row_resident_forward_matches_runnable_oracle` must
stay 50/50 at exactly **2.122e-4** and `metal_attention_decode_split3_is_bit_identical_to_v1`
must pass; either moving means decode was touched.

### 18b. THE PIN FAILED — TWICE. What was wrong with it, and the amendment

§18a was written before the first comparison and it did not survive it. Both failures are
reproduced here verbatim rather than edited out, because a phase that pins a bound and then
quietly moves it has published nothing.

**Failure 1, at n = 5:**

```
max |final_hidden diff| 4.487305e0 exceeds the published bound 5.000000e-2
(caches agree to 1.486051e-2)
```

**Failure 2, at n = 256, after splitting the hidden out:**

```
max |KV diff| 5.847931e-2 exceeds the published bound 5.000000e-2
at K layer 10 kv_head 0 position 9 dim 201: 70.81397 (0x428da0c1) vs 70.87245 (0x428dbeb2)
```

**The error was the scale, not the reasoning.** §18a argued — correctly, and this still holds —
that half round-to-nearest is 2^-11 = 4.883e-4 *relative* per operand element, so the 2.122e-4
f32-reduction-noise floor is unreachable on this path by arithmetic. It then picked an
**absolute** number as if this row's tensors were O(1). Measured, they are not:

| tensor | max magnitude on this row |
|---|---:|
| K / V caches | **1.408e2** (n = 2 400; 8.26e1 at n = 5) |
| final hidden | **3.294e4** |

So the pinned 5.0e-2 was **3.6e-4 relative** at kv_scale 1.396e2 — *below* the half round-off
floor the same paragraph had just derived. It was unreachable by arithmetic, which is precisely
the failure mode §18a set out to avoid, arrived at from the other direction. And one scalar was
covering two tensors **three orders of magnitude apart**: the diff site printed above
(70.81397 vs 70.87245) is a 8.3e-4 relative disagreement on a value of 71, which is exactly
half precision doing what half precision does.

**The amendment.** The gate is still `kv_equivalence::meets_bound`, still with both halves, and
the outlier factor is **unchanged at 8.0** — the half §16g showed carries the power. What
changes is that each tensor is bounded on its own scale, and the caches and the final hidden are
compared separately:

| clause | amended pin | justification |
|---|---|---|
| caches, scalar | `2.0e-3 x kv_scale` fed to `meets_bound` | ~2x the single-GEMM half floor (two operands staged in half, ~9.8e-4 per output before cancellation), and still **below** the weakest recorded window mutant, which is 4.29e-1 absolute = **3.1e-3 relative** at this row's kv_scale |
| caches, outlier | `8.0x` the per-position median — **unchanged from §18a** | 5.1x below the weakest recorded mutant ratio (41x, `w-multi-2400`/`window_plus_one`); `w-len-513`'s median is exactly 0.0, where any finite factor fires |
| final hidden | `1.0e-2` relative to its own magnitude | arithmetic, not measured: 2^-11 per element x 2 operands = 9.8e-4 per GEMM output, and the residual stream sums 52 GEMM-fed blocks (26 layers x attention + FFN); in quadrature sqrt(52) x 9.8e-4 = 7.1e-3, rounded up |

**The thin part, said plainly.** The cache scalar bound (2.0e-3 relative) now sits only **1.5x**
below the weakest window-mutation signature on record (3.1e-3 relative). In absolute terms the
measured worst numerics is 9.905e-2 against a weakest mutant of 4.29e-1 — 4.3x. That is a real
narrowing versus Phase 2, which had infinite separation because it was bit-identical, and it is
the price of the speed the user chose. **The outlier half is what carries this gate**, at 4.5x
measured against a 41x weakest mutant — 9.1x of separation — exactly as §16g predicted it would
have to.

### 18c. What shipped — and what was REUSED rather than written

**The kernel is not new and it is not experimental.** `q8_0_block_wire_mm` (`src/metal.rs:1431`)
is the **shipped default prefill GEMM for every other Q8_0 row on this host**: it is armed by
`CAMELID_METAL_MM`, which `apply_default_fast_stack` (`src/main.rs:5825-5843`) sets to `1` in the
CLI unless the operator says otherwise. `docs/perf-deep-dive/METAL_PARITY_PLAN.md:30,37,128` is
explicit that this kernel — not the scalar k-split GEMM — is "the established prefill baseline"
and the parity reference other kernels are held to. So Phase 3 did not adopt an unproven kernel
to buy speed; it stopped **excluding** gemma3 from the one the rest of the tree already runs.

It fits without modification, which was checked rather than assumed:

- **No head_dim dependence anywhere.** Tile constants `NR0 = 64` (weight rows), `NR1 = 128`
  (prompt columns), `NK = 32` (`src/metal.rs:1443-1445`) are shape constants; head_dim enters the
  prefill GEMMs only through `q_dim`/`kv_dim`, as a plain output row count. There is no per-lane
  fixed-size array, so nothing here is in the `MAX_DPL` / `MAX_DOCT` family.
- **Fixed 12 288 B of threadgroup memory** (`src/metal.rs:1429-1430`), independent of head_dim,
  rows and chunk width.
- **256-value contraction alignment already satisfied**: NK = 32 is exactly one Q8_0 block.

Not reused, and why: `half_mm_batched_f16o` (`src/metal.rs:4106`) is the *attention*-as-matmul
GEMM — half in, half out, for the S = Q·Kᵀ and P·V panels. It is Tier B's kernel, and Tier B's
`use_attn_mm` gate additionally requires `!has_qk_norm`, which this arch fails. Phase 3 is the
weight GEMMs only.

| piece | where |
|---|---|
| `PrefillGemm` (EnvDefault / ForceScalar / ForceMm) | `src/metal.rs` |
| `gemma3_prefill_mm_enabled` — `CAMELID_GEMMA3_PREFILL_MM`, default OFF | `src/metal.rs` |
| `PREFILL_MM_THREADGROUP_BYTES` / `_ROW_TILE` / `_TOKEN_TILE`, `threadgroup_alloc_fits`, `assert_threadgroup_fits` | `src/metal.rs` |
| MM admission + half staging panels + the two encode closures, inside `prefill_tokens_windowed_inner` | `src/metal.rs` |

**Zero new MSL.** The whole change is host wiring plus one existing elementwise kernel
(`f32_to_f16`, `src/metal.rs:3459`) to stage the activation panel.

**What did NOT move, which is most of the correctness argument.** Every per-row stage still runs
the EXACT single-token kernel at a row byte offset: per-head QK-norm, RoPE, K/V scatter and
attention. The sliding-window mask is untouched — the per-row `(window_start, position_count)`
still come from `schedule_window_bounds` and still ride in through the attention scalar's
`kv_base_offset`. The sandwich norms, GeGLU and both residual adds are the same kernels on the
same f32 buffers. **Decode is not on this path at all.**

**The threadgroup-limit query the tree never had.** The campaign plan flagged that
`maxThreadgroupMemoryLength` is queried nowhere in the tree. It is now: the MM admission
predicate checks it (and declines, rather than asserting, because it has a fallback), and the two
genuinely size-dependent allocations that had no guard at all — the K-quant resident GEMV scratch
(`scratch_ints * 4`) and the flash prefill tiles (`128 * head_dim + 7296`, which is 40 064 B at
head_dim 256, past the 32 KiB limit) — now assert against it with both numbers in the message
instead of handing an over-large request to Metal.

**Admission relaxes nothing.** `MAX_DPL` (`src/metal.rs:4595`), `MAX_DOCT` (`:4339`) and every
`<= 128` host gate they guard are byte-for-byte unchanged. The MM predicate is separate and
conjunctive: the flag, all seven weights Q8_0, 128-multiple output row counts, 32-multiple
contraction widths, and the device threadgroup limit. Miss any one and the path falls back to the
Phase 2 scalar GEMV **losslessly** — asserted, not assumed: the synthetic fixture's `ffn_dim` is
288, and running it with `ForceMm` must still come out bit-identical to token-by-token.

### 18d. The gates

| gate | result |
|---|---|
| **Decode, bit-exact** — `gemma3_real_row_resident_forward_matches_runnable_oracle` | **PASS, 50/50 greedy tokens identical, overall max abs logit diff EXACTLY 2.122e-4** — the pinned constant, digit for digit (per-depth 6.247e-5 / 7.820e-5 / 9.584e-5 at depths 1/5/50). Decode did not move. |
| **Decode, bit-exact** — `metal_attention_decode_split3_is_bit_identical_to_v1` | **PASS**, unchanged — decode attention is raw-bit identical across all six geometries including the production 4x1x256 and the windowed non-zero `kv_base_offset` read |
| **G6 — KV envelope** (§18a/§18b), n = 5/256/257/513/1024/2400 | PASS, table below |
| **G1 unchanged where it still applies** — `metal_batched_windowed_prefill_is_bit_identical_to_token_by_token`, `gemma3_real_row_batched_prefill_kv_bit_identical` | **PASS, still bit-identical.** The synthetic fixture (10 lengths x 2 arch shapes) and the real row (n = 5/256/257/513/1024, 14 811 136 elements at n=1024) both come out `bit_identical=true`. Both now pin `PrefillGemm::ForceScalar` explicitly, so a shell with the Phase 3 flag armed cannot silently turn a bit-identity assertion into a different claim. Phase 2's guarantee is intact and independent of Phase 3's presence in the tree. |
| **G7 — mutation harness** | **PASS, all 7/7 mutants caught, `survivors` empty**, re-run with Phase 3 in the tree (1 604 s, one process). Per-mutant token/KV reproduces §16g and §17d digit for digit: `window_minus_one` 0/9 and 9/9, `window_plus_one` 0/9 and 9/9, `no_lower_bound` 9/9 and 9/9, `window_on_all_layers` 6/9 and 9/9, `window_on_wrong_layers` 9/9 and 9/9, `layer_pattern_shift_by_one` 8/9 and 9/9, `rope_tables_swapped` 9/9 and 9/9 — including the finding that a one-position window error changes NO generated token on any of the nine items. **See the separation table below: this re-run is what lets Phase 3's numerics be compared to the mutation signatures at the SAME prompt lengths, rather than against a remembered number.** |
| **Window-edge pack vs the pinned oracle**, MM armed | **70/72 legs token-AND-text identical, prompt tokenization 24/24** — the Phase 1/2 baseline EXACTLY. The two failures are bit-for-bit the pre-existing depth-50 pair: `w-len-256` (256 prompt tokens) at generated index **13** and `w-len-513` (513 prompt tokens) at index **5** — same items, same indices, both unanchored ladder items carrying the open-ended "name one item mentioned above" question, neither a window item. Their min top-2 margins moved (camelid 0.0225 / 0.0857 nat against oracle 0.2042 / 0.3140), which is expected since the prefill numerics changed; they are reported as failures, `all_pass: false`, and are neither excused nor "fixed". `text_reencode_artifact` fired on 0/72, and token identity is scored on `camelid.generated_token_ids`, not on a re-encode. |
| `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle` | **PASS, 5/5 identical, 7.820e-5** — unchanged. Its logit-diff constant was allowed to move this phase since prefill numerics change; it did not, because this test drives the TOKEN-BY-TOKEN prefill, which Phase 3 does not touch. |
| `cargo fmt --check`, `clippy --all-targets -D warnings`, `cargo test --all-targets`, `scripts/check-public-scrub.sh` | **all clean.** `cargo fmt --check` rc=0, `cargo clippy --all-targets -- -D warnings` rc=0, `cargo test --all-targets` rc=0 (40 green targets, zero failures), `scripts/check-public-scrub.sh` rc=0. |

**G6, measured** (`gemma3_real_row_prefill_mm_kv_meets_published_envelope`, chunk 256, MM prefill
vs `n` sequential `forward_token` prefill decodes — the SHIPPED lane, not Phase 2's intermediate):

| n | differing / compared | kv max abs | kv REL | per-position median | **outlier ratio** | hidden max abs | hidden REL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 72 253 / 72 320 | 1.486e-2 | 1.799e-4 | 1.159e-2 | **1.3x** | 4.487e0 | 1.362e-4 |
| 256 | 3 702 590 / 3 702 784 | 5.848e-2 | 4.188e-4 | 1.967e-2 | **3.0x** | 9.069e0 | 2.753e-4 |
| 257 | 3 717 053 / 3 717 248 | 5.848e-2 | 4.188e-4 | 1.967e-2 | **3.0x** | 9.069e0 | 2.753e-4 |
| 513 | 7 419 702 / 7 420 032 | 6.075e-2 | 4.351e-4 | 1.967e-2 | **3.1x** | 1.037e1 | 3.148e-4 |
| 1 024 | 14 810 580 / 14 811 136 | 9.905e-2 | 7.094e-4 | 2.221e-2 | **4.5x** | 3.975e1 | 1.207e-3 |
| 2 400 | 34 712 394 / 34 713 600 | 9.905e-2 | 7.035e-4 | 2.315e-2 | **4.3x** | 6.228e1 | 1.891e-3 |

Bounds: cache REL 2.0e-3 (worst 7.094e-4, **2.8x margin**), hidden REL 1.0e-2 (worst 1.891e-3,
**5.3x margin**), outlier factor 8.0 (worst 4.5x, **1.8x margin**). The outlier ratio rises with
depth and then **plateaus** — 4.5x at 1 024, 4.3x at 2 400 — rather than running away, which is
what a uniform numerics change should do and what a mask defect would not.

The gate also refuses to pass by vacuity: it asserts the comparison is **not** bit-identical,
because a bit-identical result would mean the fixture had silently fallen back to the scalar GEMM
and the gate had proved nothing.

**The separation, at matched lengths — the number the whole phase turns on.** The mutation
harness was re-run on the shipped token-by-token lane (its own gate, §17d), so its signatures and
Phase 3's numerics are measured on the same row, the same schedule and the same prompt lengths.
Weakest off-by-one mutant per length against Phase 3's measured numerics:

| n | Phase 3 numerics: max &#124;ΔKV&#124; / median / ratio | weakest off-by-one mutant: max &#124;ΔKV&#124; / median / ratio | separation on max | separation on ratio |
|---:|---|---|---:|---:|
| 513 | 6.075e-2 / 1.967e-2 / **3.1x** | 4.289e-1 / **0.0** / **inf** | **7.1x** | infinite |
| 1 024 | 9.905e-2 / 2.221e-2 / **4.5x** | 1.416e0 / 2.802e-3 / **505x** | **14.3x** | **112x** |
| 2 400 | 9.905e-2 / 2.315e-2 / **4.3x** | 4.414e0 / 3.007e-2 / **147x** | **44.6x** | **34x** |

Two things this settles. **(a)** Phase 3's numerics do not swallow the campaign's headline defect
at any length tested — the margin is smallest at 513, where the mutation is weakest because a
513-wide prompt barely clips a 512 window, and it grows with depth exactly as it must. **(b)**
The per-position **outlier ratio is the discriminator**, not the scalar: Phase 3's ratio sits at
3-4.5x and is flat in depth, while every mutant's is 147x-infinite. A uniform numerics change and
a localized mask defect look nothing alike in that statistic, which is the property §16g
predicted the Tier B gate would have to rest on.

### 18e. THE MEASUREMENT

Three instruments, in increasing distance from the kernel. Every one records its 1-minute load
average, because this host runs other sessions' work and §16/§17's 6-15 % spread must not be
mistaken for a lever.

**(1) The three paths in ONE process, same warm weight cache, same pipelines**
(`gemma3_real_row_prefill_gemm_probe`, n = 1 200, chunk 256, load 2.87-3.05). This is the
cleanest comparison in the phase: the env flag is a latched `OnceLock`, which is exactly why
`PrefillGemm` exists.

| prefill path | wall | ms/token | whole-path TFLOPS |
|---|---:|---:|---:|
| token-by-token (today's shipped lane) | 21.432 s | 17.860 | 0.078 |
| batched GEMV (Phase 2) | 14.682 s | 12.235 | 0.114 |
| **batched MM (Phase 3)** | **3.045 s** | **2.538** | **0.550** |

**4.82x over Phase 2's GEMV, 7.04x over the shipped token-by-token lane**, at the kernel.
"Whole-path TFLOPS" divides the MODEL's GEMM work — 2 x 26 836 992 MACs x 26 layers =
1.3955 GFLOP/token — by the WHOLE per-token time, attention and norms and RoPE and scatter
included. It is therefore a lower bound on the GEMM's own efficiency and an honest end-to-end
figure at the same time.

**(2) The chunk-width sweep — the mechanism, made visible.**
Phase 2's §17e(1) headline was that chunk width did not matter *at all*: 11.99 -> 12.59 ms/token
across a 32x range, 5 % spread, *wider slightly worse*, because `MAX_T = 8` fixed the batching
factor at 8 no matter how wide the chunk was. Same probe, same n = 1 200, MM armed (load 4.02):

| rows per command buffer | 16 | 32 | 64 | **128** | 256 | 512 | token-by-token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 2 GEMV, ms/token | 11.99 | 12.08 | 12.27 | 12.36 | 12.43 | 12.59 | 13.73 |
| **Phase 3 MM, ms/token** | **4.203** | **3.178** | **2.775** | **2.575** | **2.551** | **2.563** | 13.646 |

The Phase 3 row falls steeply and then **stops falling exactly at 128** — the kernel's
`NR1` prompt-column tile — and is flat beyond it. That is the batching factor becoming visible
in wall-clock: it is now the tile width, not a constant buried in a GEMV. It also settles the
one free parameter without guessing: the shipped default of 256 rows is already at the floor,
and no tuning attempt is owed on this axis.

**(3) The traffic arithmetic, and where it stops explaining things.**
Per token, over all 26 layers, on this row (wire Q8_0 = 34 B per 32 weights; layer weights
741.4 MB total):

| term | Phase 2 GEMV (`MAX_T = 8`) | Phase 3 MM (128-column tile) |
|---|---:|---:|
| weight stream | 741.4 / 8 = **92.7 MB** | 741.4 / 128 = **5.79 MB** |
| activation re-read | **1 400 MB** | **21.8 MB** |
| GEMM output writes | 1.84 MB | 1.84 MB |
| f32 -> half staging | — | 1.60 MB |
| **total** | **1 494 MB** | **31.0 MB** |
| at this host's ~120 GB/s | 12.45 ms/token | 0.26 ms/token |
| **measured** (n = 1 200, chunk 256) | **12.24 ms/token** | **2.55 ms/token** |

A **48x** reduction in GEMM-side traffic. Phase 2's row is bracketed by its own roofline
(12.45 predicted, 12.24 measured) — it was bandwidth-bound and the model said so. **Phase 3's
row is not, and that is the finding:** 2.55 measured against 0.26 predicted, because once the
traffic is gone the GEMM is no longer what prefill is spending its time on.

The remainder decomposes with no free parameters, using the attention slope Phase 0 refitted on
this host (1.789e-4 ms per layer-position) and Sigma_l(d) = 22*min(d,512) + 4*d:

| n | Sigma_l / n | attention, predicted | GEMM compute @3 TFLOPS | predicted total | measured |
|---:|---:|---:|---:|---:|---:|
| 1 200 | 11 267.7 | 2.016 ms | 0.47 ms | 2.49 ms | **2.54 ms** (-2.1 %) |
| 2 400 | 14 866.9 | 2.659 ms | 0.47 ms | 3.13 ms | **3.16 ms** (-0.8 %) |

(2 400 ms/token is from the served TTFT below; 1 200 from the in-process probe.)

**Prefill is now 79 % attention at 1 200 tokens and 84 % at 2 400.** That is the whole
finding of Phase 3 restated: the GEMM stopped being the cost.

**(4) TTFT, end to end, served.** Same binary, one server alive at a time, PID saved and killed
by that PID with death verified (`ps -p` + `pgrep` + port). Streamed SSE timed request-start to
the first chunk carrying non-empty content — never inferred from a non-streaming total. Prompts
are exact token-id arrays so the tokenizer is out of the loop, and every request takes a distinct
prompt window so no two share a prefix (`prompt_cache_hit` false on all 27). Round 0 is the cold
column and is excluded from the mean; the mean is rounds 1-2. All three legs ran inside 20
minutes of each other at 1-minute load 2.50-3.73.

| N | token-by-token (today) | batched GEMV (Phase 2) | **batched MM (Phase 3)** | MM vs GEMV | MM vs today |
|---:|---:|---:|---:|---:|---:|
| 600 | 8.734 s (sd 0.062) | 6.998 s (sd 0.024) | **1.304 s (sd 0.003)** | **5.36x** | **6.70x** |
| 1 200 | 18.010 s (sd 0.410) | 14.450 s (sd 0.004) | **3.050 s (sd 0.001)** | **4.74x** | **5.90x** |
| 2 400 | 38.174 s (sd 0.090) | 30.372 s (sd 0.016) | **7.573 s (sd 0.062)** | **4.01x** | **5.04x** |

Cold (round 0): 9.301 / 18.073 / 37.708 off; 7.015 / 14.407 / 30.302 GEMV; 1.351 / 3.077 /
7.562 MM. **No first-request PSO-compile penalty appeared on the new path either** — the MM cold
column is within 1-4 % of its warm mean, and at 600 tokens it is the warm rounds that are faster.

Per-token and against the envelope:

| N | MM ms/token | whole-path TFLOPS | GEMM traffic |
|---:|---:|---:|---:|
| 600 | 2.174 | 0.642 | 31.0 MB/token |
| 1 200 | 2.542 | 0.549 | 31.0 MB/token |
| 2 400 | 3.155 | 0.442 | 31.0 MB/token |

The TFLOPS column FALLS with prompt length, which is the right shape and the point: the GEMM work
per token is constant, so a falling whole-path rate is attention taking a larger share.

**Cross-check against the committed Phase 2 columns**, disclosed rather than smoothed. §17e
measured 7.772 / 16.314 / 35.162 s flag-off and 7.231 / 14.952 / 31.232 s flag-on, 2.5 hours
earlier at load 3.0-4.5. Today's GEMV column is 3-4 % FASTER than that flag-on column and today's
token-by-token column is 8-12 % SLOWER than that flag-off column. Both sit inside the 6-15 %
run-to-run spread this host is documented at (§16/§17, Phase 0 §9), and the host has been under
sustained multi-session load for seven hours. The Phase 3 conclusion does not rest on either
direction: measured against Phase 2's OWN committed flag-on numbers the MM path is still
5.5x / 4.9x / 4.1x.

### 18f. WHERE THE PREFILL ENVELOPE NOW SITS — and what it does to Tier B

**§17f(3) put a choice to the user: bit-identity, or the speed. The speed was chosen, and this
is what it bought and what it cost.**

Bought: **4.0-5.4x on served TTFT against Phase 2's batched GEMV and 5.0-6.7x against the shipped
token-by-token lane** — 2 400-token TTFT from 38.2 s to **7.6 s**, 1 200 from 18.0 s to **3.1 s**,
600 from 8.7 s to **1.3 s**. At the kernel, 12.235 -> 2.538 ms/token (4.82x). The mechanism is a
**48x** cut in GEMM-side memory traffic, 1 494 -> 31.0 MB/token, and it is confirmed three ways:
the chunk sweep now saturates exactly at the kernel's 128-column tile, the Phase 2 roofline
bracketed its own measurement, and the Phase 3 residual is fully accounted for by attention.

Cost, stated without softening:

1. **The prefill GEMMs are no longer bit-identical to the shipped lane**, and cannot be made so
   with this kernel. The measured distance is 7.09e-4 relative on the caches and 1.89e-3 relative
   on the final hidden.
2. **The KV gate's scalar half narrowed from infinite separation to 4.3x** (9.905e-2 measured
   against the weakest recorded window-mutation signature of 4.29e-1). The outlier half keeps
   9.1x. Both are real gates; neither is what bit-identity was.
3. **Decode is unaffected**, and that is asserted, not assumed — this path is not on the decode
   graph at all and the decode gates are unmoved.

**Three things this changes for Phase 4 / Tier B, and they invert §17f(2).**

1. **§17f(2) is withdrawn on its own arithmetic.** It said "collapsing attention to zero would
   leave ~25 s" at 2 400 tokens and concluded Tier B "cannot reach the plan's 1.3-2.4 s, or
   anything near it". That was true *of the Phase 2 path*, where 25 s of the 31 s was GEMM.
   Phase 3 removed that 25 s. At 2 400 tokens attention is now **~78 %** of prefill, and
   collapsing it would leave roughly 0.47-0.7 ms/token = **1.1-1.7 s** — inside the campaign
   plan's §6 Tier B projection, which §17f(2) had written off. **Tier B is the whole remaining
   prize, and it is now the ONLY remaining prize.**
2. **The campaign's stopping rule 5 is back in force and is the right rule again.** It requires
   Phase 4's kernel to beat the per-row attention by >= 3x on wall clock. With attention at ~78 %
   of prefill that is now the binding question, exactly as the plan originally framed it and
   contrary to §17f(2)'s reframing.
3. **The GEMM has ~5x of headroom left and it is not worth taking.** 0.550 TFLOPS whole-path
   against this host's ~3.4 TFLOPS Q8_0 wall looks like a 6x gap, but that ratio is against the
   WHOLE per-token time, 78 % of which is attention. The GEMM's own share is ~0.47 ms/token of
   2.55; taking it to zero would buy 18 % and taking attention to zero would buy 78 %. **There is
   no second tuning attempt owed on the GEMM** — the chunk sweep already settled the one free
   parameter, the traffic model closes to within 3-8 %, and there are no GPU counters on this host
   (§15) with which to chase the rest. The campaign's stopping rule is respected by stopping here,
   not by guessing.

### 18g. Deliberately not done

- **No new MSL.** Not one line. The kernel was found in the tree, already shipped and already the
  default for other rows, and reused unmodified.
- **The default was NOT flipped.** `CAMELID_GEMMA3_PREFILL_MM` stays OFF, on top of
  `CAMELID_GEMMA3_BATCH_PREFILL` which also stays OFF. A path that is not bit-identical to the
  shipped lane should be flipped in its own phase with its own decision, not as a side effect of
  the phase that built it.
- **No second GEMM tuning attempt** — see §18f(3). The stopping rule allowed two; the first
  (chunk width) was measured and showed the default is already at the floor, and the second would
  be chasing 18 % of a cost that is no longer the bottleneck.
- **`MAX_T` was still not raised.** §17g recorded it as an available lever worth ~7 %; it is now
  moot for this row, since the GEMV is not on the fast path any more. It stays available for any
  row that cannot take the tiled kernel.
- **The 68 MB of dead f16 KV mirrors at head_dim 256 was still not reclaimed.** Owed since the
  campaign plan's §7; still owed.
- **No promotion of any surface.** No README, COMPATIBILITY, STATUS or ledger edit. This phase
  adds a second flag behind a flag.
- **No pack item softened, no failing leg excused.**

### 18h. Handover

**The flag stack, as it stands.** `CAMELID_GEMMA3_BATCH_PREFILL=1` arms Tier A (Phase 2,
bit-identical, 1.25x measured today). Adding `CAMELID_GEMMA3_PREFILL_MM=1` swaps its seven
weight GEMMs onto the tiled kernel (Phase 3, not bit-identical, 4.0-5.4x on top). Both default
OFF. Arming the MM flag alone does nothing — the only production caller of
`prefill_tokens_windowed` is the gemma3 batched-prefill seam.

**The decision this phase does NOT make.** Whether either default flips. Phase 2 recommended
holding the Tier A flip until the GEMM question was decided; the GEMM question is now decided
and measured, so the flip decision is ripe — but it is a decision about shipping a
non-bit-identical prefill to users, and it belongs to the user, not to the phase that built it.
Everything needed to make it is in §18b (what the numerics cost), §18d (what the gates say) and
§18e (what the speed is).

**Evidence bundle:** `qa/evidence-bundles/gemma3-1b-q8-prefill-mm-20260731-head-2f0134c7/`.

**For Phase 4 / Tier B, in priority order:**

1. Attention is **79-84 %** of prefill now. Everything else is noise by comparison.
2. The campaign plan's Design C (windowed attention-as-matmul via `half_mm_batched_f16o` +
   `softmax_causal_rows`) is unchanged and still applies, including its two real blockers: the
   `!has_qk_norm` clause on `use_attn_mm` (`src/metal.rs:13908`) and the `<= 128` head_dim clause.
   Phase 3 did **not** touch either, and the f32->half Q/K conversion the plan proposes is now a
   smaller step than it was, because this phase already established that staging half operands on
   this row costs ~7e-4 relative and passes the envelope.
3. The threadgroup-limit helpers (`threadgroup_alloc_fits` / `assert_threadgroup_fits`) exist now
   and Tier B should use them at its head_dim-dependent allocation, which is the one the campaign
   plan flagged as the reason the unchanged flash layout hard-fails at head_dim 256.
4. Phase 0's chat-path tokenizer finding is now the second-largest remaining item after attention:
   `parse_special=true` costs 0.53 ms per prompt token, i.e. **1.27 s at 2 400 tokens** — which is
   17 % of today's 7.573 s MM TTFT, against 3.7 % of the 35 s it was measured on. It is outside
   the engine and untouched by any tier. It was the right call not to smuggle it into Phase 2 or 3;
   it should now be scoped on its own.

## 19. Long-prompt TTFT campaign, Phase 4 record — Tier B, the attention (2026-07-31)

Branch `feat/gemma3-batched-prefill`, continuing from §18. §18f withdrew §17f(2)'s pessimism on
its own arithmetic and left exactly one prize: with the weight GEMMs collapsed, prefill was
**79 % attention at 1 200 prompt tokens and 84 % at 2 400**, and every row still ran its own
`encode_attention` over its whole window. Phase 4 replaces that per-row loop with the
attention-as-matmul chain the tree already ships for non-windowed rows, and teaches that chain
the per-query-row LOWER mask bound it has never had.

### 19a. What was built — and what was REUSED

**Reused, unmodified except for one uniform.** `half_mm_batched_f16o` (`src/metal.rs:4106`),
`softmax_causal_rows` (`:4256`) and `transpose_v16` (`:3938`) are the S = K·Qᵀ / row-softmax /
O = Vᵀ·P chain `prefill_tokens` already runs for other Q8_0 rows on this host. The operands are
the **half K/V mirrors** (`cache_k16` / `cache_v16`) that the scatter already dual-writes on
every token and that the campaign plan §7 recorded as **68 MB of dead allocation at head_dim
256**, because the only reader (the split-K/v2 decode route) requires head_dim ≤ 128. They are
not dead any more; Phase 4 is their first reader on this row.

**New MSL: one `window` uniform and ~15 lines**, all in `half_mm_batched_f16o` and
`softmax_causal_rows`:

| site | what |
|---|---|
| `half_mm_batched_f16o`, S pass (`causal_mode == 1`) | lower TILE cull `r0 + NR0 + window <= t0 + q_offset + 1` — the tile is below the window when its LARGEST key plus the window fails to reach its SMALLEST query — plus the same test at 32×32 quadrant granularity on `sg_active` |
| `half_mm_batched_f16o`, PV pass (`causal_mode == 2`) | lower `k_start` from the tile's smallest query, aligned DOWN to `NK` = 32; the `kk0` loop starts there instead of 0 |
| `softmax_causal_rows` | `lo = (window == 0 \|\| q_abs < window) ? 0 : q_abs + 1 - window` on the max and sum loops, and the write loop **ZEROES** `[0, lo)` rather than skipping it |

**Why P is zeroed below `lo` and not skipped.** The PV pass's `k_start` is a per-TILE bound taken
from the tile's smallest query, while the mask is per-ROW. For any row above that smallest query,
PV therefore reads columns in `[k_start, lo_row)`. Zeroing is what makes a per-tile bound sound
for a per-row mask; skipping would read whatever was there.

**One parameterised kernel, not two.** `window = 0` collapses every added term: the culls never
fire, `k_start` is 0, `lo` is 0. So the 4 global gemma3 layers, and every non-gemma3 caller of
this chain, run the identical arithmetic — the `window = 0` case IS the pre-change kernel, which
is the regression that keeps `prefill_tokens` byte-unchanged for every other row.

**The convention, and where it lives.** The predicate is written `key + window > q_abs`, never
`q_abs - window`, so it is unsigned-safe below the window with no saturating branch. The `>`
rather than `>=` is where "the window INCLUDES the current position" lives, and it is the same
expression `schedule_window_bounds` (`src/metal.rs:12891`) pins host-side; the host passes the
schedule's own `window[l]` straight through as the uniform rather than restating it.

**The `!has_qk_norm` clause on `use_attn_mm` — the question §18 left open.** It is about the
**surrounding plumbing, not the GEMM**. In `prefill_tokens` that clause exists to force the
*weight* GEMM to emit f32 Q/K so the f32 per-head norm can run in cpu_reference order
(`src/metal.rs:13985-13990` states this). On the gemma3 seam the weight GEMM already emits f32 —
both the Phase 2 GEMV and Phase 3's `q8_0_block_wire_mm` do — the per-head QK-norm and RoPE
already run the exact single-token f32 kernels per row, and Q is converted to half only AFTER
both. The condition the clause enforces is already met, and `half_mm_batched_f16o` itself has no
QK-norm dependence anywhere. `prefill_tokens`' own gate is untouched.

**Admission relaxes nothing.** `MAX_DPL` (`src/metal.rs:4595`), `MAX_DOCT` (`:4339`) and every
`<= 128` host gate they guard are byte-for-byte unchanged. The Tier B predicate is separate and
conjunctive: the flag, Phase 3's MM GEMM (the context panel stays half into the O projection),
f32 K/V caches (the half mirrors are the operands), head_dim a multiple of 64, a 64-aligned chunk
start, no degenerate `Some(0)` window, the S/P scratch inside `attn_mm_scratch_cap_bytes`, and
`threadgroup_alloc_fits` for the FIXED 8 192 B this chain needs — fixed, so head_dim never enters
it. Miss any one and the path falls back to Phase 3's per-row attention rather than declining the
prefill.

**The scratch is linear in the CHUNK, not quadratic in the prompt.** S and P are
`[head][chunk row][n_pad]` half, so 2 × 4 × 256 × 2 432 × 2 B = **10.0 MB** at n = 2 400 and
33.5 MB at 8 192 — the campaign plan's ~1 GB-at-8k figure was for the untiled design, and query
blocking is already done by the chunk loop. The cap check and the per-row fallback are
implemented anyway.

### 19b. THE BUG THIS PHASE FOUND, in full

The first G6 run failed at n = 257, 513 and 2 400 and passed at 5, 256 and 1 024:

```
n=257:  kv_rel 8.675e-2  outlier_ratio 369.0x
n=513:  kv_rel 9.649e-2  outlier_ratio 425.6x
n=2400: max |KV diff| 3.312101e1 at K layer 17 kv_head 0 position 2372 dim 253:
        24.620667 (0x41c4f720) vs -8.500366 (0xc1080180)
```

The pattern is the diagnosis: those are exactly the lengths with a **ragged last chunk**
(k < chunk — 257 and 513 end with a 1-row chunk, 2 400 with a 96-row one), and 2 372 sits inside
2 400's last chunk. `softmax_causal_rows` derives its per-head base as
`head * rows_per_block * n_pad`; it had been handed `rows_per_block = k`, while the two GEMMs
were handed the panel's true row pitch, `chunk`, as their batch strides. For any chunk narrower
than the panel, **every head above head 0 read and wrote the wrong panel offset.** The fix is one
word — `rows_per_block` is the panel pitch, and the live-row count rides in the dispatch height
instead.

Three things worth recording. **(1)** It is a silent-corruption bug, not a crash: every index
stayed in bounds. **(2)** It only appears at four of the six gate lengths, and only because those
lengths were chosen to straddle chunk and window edges — a sweep of round numbers would have
passed. **(3)** It is precisely the class the campaign's §16g finding predicted token identity
would be weak against and KV-equivalence strong against, and that is how it surfaced: the G6
gate named the layer, head, position and dim.

### 19c. The gates

| gate | result |
|---|---|
| **Decode, bit-exact** — `gemma3_real_row_resident_forward_matches_runnable_oracle` | **PASS, 50/50 greedy tokens identical, overall max abs logit diff EXACTLY 2.122e-4** — the pinned constant, digit for digit (depth-50 leg 9.584e-5). Decode did not move. |
| **Decode, bit-exact** — `metal_attention_decode_split3_is_bit_identical_to_v1` | **PASS**, unchanged, all six geometries including the production 4x1x256 and the windowed non-zero `kv_base_offset` read |
| **G6 — KV envelope**, n = 5/256/257/513/1024/2400 | **PASS against §18b's bounds, UNMOVED.** Table below. |
| **G5 — the mask, at the kernel** (`windowed_attn_mm_mask_matches_the_pinned_window_convention`) | **PASS.** At windows 0/1/37/64/65/128 the number of non-zero P entries is EXACTLY `min(q+1, window)` — an integer identity, so `>=` for `>` moves it by one and fails; `P[q][q-window]` is exactly 0.0 and `P[q][q-window+1]` strictly positive; the surviving weights match a CPU softmax over `[max(0,q+1-w), q]` to **5.949e-5**; and query blocking at a non-zero `q_offset` reproduces the unblocked P **bit for bit** — the code path the campaign plan flagged as otherwise unshippable-untested. |
| **G1, still bit-identical where it still applies** — `metal_batched_windowed_prefill_is_bit_identical_to_token_by_token`, `gemma3_real_row_batched_prefill_kv_bit_identical` | **PASS.** Synthetic fixture (10 lengths x 2 arch shapes) and the real row (n = 5/256/257/513/1024, 14 811 136 elements at 1 024) both `bit_identical=true`. Both pin `PrefillGemm::ForceScalar` **and** `PrefillAttn::ForceRow`, so a shell with the Phase 4 flag armed cannot turn a bit-identity assertion into a different claim. Tier A's guarantee is intact and independent of Phases 3 and 4 being in the tree. |
| `gemma3_session_level_token_by_token_prefill_matches_runnable_oracle` | **PASS, 5/5 identical, 7.820e-5** — unchanged, because this drives the TOKEN-BY-TOKEN prefill, which Phase 4 does not touch. |

**G6, measured** (`gemma3_real_row_prefill_attn_mm_kv_meets_published_envelope`, chunk 256, the
full Tier A + Phase 3 + Phase 4 stack against `n` sequential `forward_token` prefill decodes —
the SHIPPED lane, so these numbers are directly comparable to §18d's):

| n | differing / compared | kv max abs | kv REL | per-position median | **outlier ratio** | hidden max abs | hidden REL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | 72 252 / 72 320 | 4.434e-2 | 5.368e-4 | 1.403e-2 | **3.2x** | 6.000e0 | 1.822e-4 |
| 256 | 3 702 622 / 3 702 784 | 6.901e-2 | 4.942e-4 | 3.276e-2 | **2.1x** | 3.020e1 | 9.168e-4 |
| 257 | 3 717 085 / 3 717 248 | 6.901e-2 | 4.942e-4 | 3.283e-2 | **2.1x** | 3.020e1 | 9.168e-4 |
| 513 | 7 419 752 / 7 420 032 | 7.527e-2 | 5.390e-4 | 3.164e-2 | **2.4x** | 3.020e1 | 9.168e-4 |
| 1 024 | 14 810 670 / 14 811 136 | 1.644e-1 | 1.177e-3 | 3.293e-2 | **5.0x** | 8.423e1 | 2.557e-3 |
| 2 400 | 34 712 607 / 34 713 600 | 1.927e-1 | 1.369e-3 | 3.329e-2 | **5.8x** | 8.423e1 | 2.557e-3 |

Bounds, all three carried over from §18b without amendment: cache REL 2.0e-3 (worst 1.369e-3,
**1.46x margin**), hidden REL 1.0e-2 (worst 2.557e-3, **3.9x margin**), outlier factor 8.0 (worst
5.8x, **1.38x margin**). **The envelope was not widened, and that was not a choice with much
room in it**: §18b already recorded the scalar bound sitting only 1.5x below the weakest recorded
window mutation, so a wider bound would have stopped being a gate. Phase 4 had to fit inside a
bound pinned before Phase 3 measured anything, and it does.

The gate refuses to pass by vacuity **twice**: the result must not be bit-identical to the
token-by-token lane (which would mean the MM GEMM had fallen back), and must not be bit-identical
to the per-row attention path either (which is exactly what a silent Tier B admission failure
would produce).

**The separation, at matched lengths.** Phase 4's numerics against §18d's weakest recorded
off-by-one window mutant at the same prompt length:

| n | Phase 4: max &#124;ΔKV&#124; / median / ratio | weakest off-by-one mutant: max &#124;ΔKV&#124; / median / ratio | separation on max | separation on ratio |
|---:|---|---|---:|---:|
| 513 | 7.527e-2 / 3.164e-2 / **2.4x** | 4.289e-1 / **0.0** / **inf** | **5.7x** | infinite |
| 1 024 | 1.644e-1 / 3.293e-2 / **5.0x** | 1.416e0 / 2.802e-3 / **505x** | **8.6x** | **101x** |
| 2 400 | 1.927e-1 / 3.329e-2 / **5.8x** | 4.414e0 / 3.007e-2 / **147x** | **22.9x** | **25x** |

Said without softening: **the scalar half narrowed again**, from Phase 3's 7.1x / 14.3x / 44.6x
to 5.7x / 8.6x / 22.9x — roughly halved at every length, which is what adding a second half-staged
matmul chain to a path that already had one should do. It is still an order of magnitude at the
lengths that matter and never below 5.7x. **The outlier half is what carries this gate**, at
25x-infinite, exactly as §16g predicted the Tier B gate would have to.

### 19d. THE MEASUREMENT

**(1) Both attention paths in ONE process** (`gemma3_real_row_prefill_attn_probe`), same warm
weight cache, same pipelines, Phase 3's tiled MM GEMM pinned on **both** sides so what is timed is
attention and only attention:

| n | per-row attention (Phase 3) | **batched windowed attention (Phase 4)** | speedup | 1-min load |
|---:|---:|---:|---:|---:|
| 1 200 | 3.019 s (2.516 ms/token) | **0.916 s (0.764 ms/token)** | **3.29x** | 6.48 |
| 2 400 | 7.556 s (3.148 ms/token) | **1.821 s (0.759 ms/token)** | **4.15x** | 5.95 |

**The campaign's stopping rule 5 is cleared** — it required >= 3x over Tier A's per-row attention
on measured wall clock, and this is 3.29x / 4.15x on WHOLE-prefill wall with only attention
changed, so it understates the attention term's own speedup.

**The shape is the finding, not the ratio.** Per-token cost is now **flat in prompt length** —
0.764 ms at 1 200 and 0.759 ms at 2 400 — where the per-row path grew 2.516 → 3.148. That is the
window's O(1)-per-query property arriving in wall clock: each query tile reads at most 9 kv tiles
no matter how long the prompt is. Backing out §18e's non-attention residual (0.47 ms/token,
unchanged by this phase), attention itself went from 2.016 → **~0.29** ms/token at 1 200 and
2.659 → **~0.29** at 2 400: **~7x and ~9x on the attention term**, and its share of prefill fell
from 79 % / 84 % to **~38 %** at both lengths.

**(2) The tile census**, computed with the kernel's own cull predicates over the actual dispatch
grid (all 26 layers, chunk 256):

| n | tiles in the grid | dropped by the causal cull | **dropped by the WINDOW cull** | computed |
|---:|---:|---:|---:|---:|
| 1 200 | 5 642 | 702 | **1 210 (21.4 % of the grid, 24.5 % of what causal alone kept)** | 3 730 |
| 2 400 | 20 696 | 1 430 | **9 570 (46.2 % of the grid, 49.7 % of what causal alone kept)** | 9 696 |

Read this against the campaign plan's §3.4 prediction of 31.4 % / 58.7 % skipped, and the
difference is arithmetic rather than disagreement: the plan counted one 2 400 × 2 400 grid, while
the chunked dispatch re-covers `[0, base)` for each chunk's query tiles, and 4 of the 26 layers
are global and can never be window-culled. Over the 22 sliding layers alone the 2 400 figure is
**54.6 %**, against the plan's 58.7 %. The plan's other claim — that the skip "is worth almost
nothing at 600 tokens and everything at 2400" — is reproduced: 21.4 % at 1 200 against 46.2 % at
2 400.

**(3) The remaining gates.**

| gate | result |
|---|---|
| **G7 — mutation harness**, re-run with Tier B in the tree | **PASS, all 7/7 mutants caught, `survivors` empty** (1 823 s, one process). Per-mutant token/KV reproduces §16g / §17d / §18d **digit for digit**: `window_minus_one` 0/9 and 9/9, `window_plus_one` 0/9 and 9/9, `no_lower_bound` 9/9 and 9/9, `window_on_all_layers` 6/9 and 9/9, `window_on_wrong_layers` 9/9 and 9/9, `layer_pattern_shift_by_one` 8/9 and 9/9, `rope_tables_swapped` 9/9 and 9/9 — including the finding the whole gate design rests on, that a one-position window error changes **no generated token on any of the nine items**. |
| **Window-edge pack vs the pinned oracle**, Tier B armed | **70/72 legs token-AND-text identical, prompt tokenization 24/24 — the Phase 1/2/3 baseline EXACTLY.** The two failures are bit-for-bit the pre-existing depth-50 pair: item 13 (256 prompt tokens) at generated index **13** and item 17 (513 prompt tokens) at index **5**, same items, same indices, same oracle margins (0.2042 / 0.3140). camelid's min top-2 margins moved 0.0225 → **0.0165** and 0.0857 → **0.0733**, which is expected because the prefill numerics changed; they are reported as failures, `all_pass: false`, and are neither excused nor "fixed". |
| **The capability row's own claim, re-established under the NEW DEFAULT** | **PASS.** The shipped row asserts 15/15 sub-512 legs and 9/9 windowed legs at 606/1205/2403 prompt tokens, measured on the token-by-token prefill. Re-run against the same committed oracle captures with **no campaign env vars set at all**: windowed **9/9, `all_pass: true`, prompt tokenization 3/3**; sub-512 **15/15, `all_pass: true`, prompt tokenization 5/5**. Nothing in the row became false, so the row is not edited. |
| **The default posture actually reaches Tier B** | **PASS, and measured rather than argued.** A server started with `env -u` on all three campaign variables and `CAMELID_RESIDENT_TRACE=1` reports **`70 gemm=mm attn=mm`** and not one `attn=row` — every batched-prefill command buffer took Phase 3's GEMM and Phase 4's attention by default. |
| `cargo fmt --check`, `clippy --all-targets -D warnings`, `cargo test --all-targets`, `scripts/check-public-scrub.sh` | **all clean.** fmt rc=0; clippy rc=0 (two findings fixed in-phase: an `int_plus_one` in the mask test's CPU reference and an `unused_mut`, both test-only); `cargo test --all-targets` rc=0 with **60 green targets, 1 817 tests passed, 0 failed**; scrub rc=0. |

**(4) TTFT, end to end, served.** Same binary, one server alive at a time, PID saved and killed by
that PID with death verified (`ps -p` + a port check). Streamed SSE timed request-start to the
first chunk carrying non-empty content — never inferred from a non-streaming total. Prompts are
exact token-id arrays so the tokenizer is out of the loop, and every request takes a distinct
prompt window (`prompt_cache_hit` false on all 27). Round 0 is the cold column and is excluded
from the mean; the mean is rounds 1-2. All three legs ran inside 6 minutes of each other.

| N | token-by-token (`BATCH_PREFILL=0`) | Phase 3, per-row attention | **Phase 4, batched attention** | vs Phase 3 | vs token-by-token |
|---:|---:|---:|---:|---:|---:|
| 600 | 8.078 s (sd 0.046) | 1.316 s (sd 0.001) | **0.520 s (sd 0.016)** | **2.53x** | **15.54x** |
| 1 200 | 17.453 s (sd 0.394) | 3.101 s (sd 0.017) | **0.971 s (sd 0.002)** | **3.19x** | **17.97x** |
| 2 400 | 37.475 s (sd 0.188) | 7.703 s (sd 0.002) | **1.962 s (sd 0.025)** | **3.93x** | **19.10x** |

1-minute load: 3.06-4.71 (token-by-token), 2.96-3.17 (Phase 3), 3.00 (Phase 4). Cold (round 0):
9.288 / 17.286 / 37.457 off; 1.359 / 3.089 / 7.701 Phase 3; 0.534 / 0.970 / **1.884** Phase 4 —
the Phase 4 cold column is *faster* than its warm mean at 2 400, so no first-request PSO-compile
penalty appeared on the new path either.

Per-token, and the rate against the model's own work:

| N | Phase 4 ms/token | whole-path TFLOPS (weight-GEMM FLOPs / whole time) | attention GFLOP |
|---:|---:|---:|---:|
| 600 | 0.867 | 1.610 | 21.5 |
| 1 200 | 0.809 | **1.725** | 55.4 |
| 2 400 | 0.817 | **1.708** | 146.2 |

**ms/token stopped growing with prompt length** — 0.867 / 0.809 / 0.817 against Phase 3's
2.193 / 2.585 / 3.209 — which is the sliding window's O(1)-per-query property showing up
end-to-end and not merely at the kernel. The whole-path TFLOPS column, defined exactly as §18e
defined it (the MODEL's 1.3955 GFLOP/token of weight-GEMM work divided by the WHOLE per-token
time, attention and norms and RoPE and scatter included, so it is a lower bound on GEMM
efficiency), went from **0.442** at 2 400 in Phase 3 to **1.708** — half of this host's ~3.4
TFLOPS Q8_0 GEMM wall, now that the whole prefill costs little more than its GEMMs.

**All three lengths land inside the campaign plan's §6 Tier B projection** (0.4-0.6 s / 0.7-1.2 s
/ 1.3-2.4 s), which §17f(2) had written off as unreachable and §18f withdrew.

### 19e. THE DEFAULT FLIP — the decision, and what it rests on

Phase 3 declined to flip: "a path that is not bit-identical to the shipped lane should be flipped
in its own phase with its own decision". This is that phase. All three flags now default **ON**
with `=0` as the operator opt-out, for the gemma3 row only.

**What it buys:** 19.1x on 2 400-token TTFT, 18.0x at 1 200, 15.5x at 600 — 37.5 s to **1.96 s**.
The five-turn chat of the campaign plan §1.5, ~105 s of cumulative prefill before the campaign,
is now ~5 s.

**What it costs, stated without softening:** the prefill GEMMs and prefill attention are no longer
bit-identical to the token-by-token lane and cannot be made so with these kernels. Measured
distance at 2 400: 1.369e-3 relative on the caches, 2.557e-3 relative on the final hidden. The KV
gate's scalar separation from the weakest recorded window mutation is 5.7x-22.9x, down from Tier
A's infinite and Phase 3's 7.1x-44.6x.

**Why the receipts support it anyway:**

1. The envelope was **pinned before Phase 3 measured anything and was not widened**. It could not
   have been: §18b already put the scalar bound 1.5x below the weakest recorded mutation.
2. The mask is proven by an **integer identity**, not a tolerance, at six windows including 1 and
   the two tile-aligned cases, plus the boundary pair, plus q_offset blocking bit-identical.
3. The mutation harness catches **7/7** with the batched attention in the tree.
4. **Decode is untouched and asserted so** — 50/50 at exactly 2.122e-4, split3 raw-bit identical.
5. **The capability row's own shipped claim was re-established under the new default**, 9/9 and
   15/15, rather than being assumed to survive.
6. **The default was proven to actually engage**, `70 gemm=mm attn=mm` with no env set.

**Why nothing else changes.** All three flags are read only from `prefill_tokens_windowed`, whose
single production caller ANDs the first with `config.gemma3.is_some()`
(`src/inference/metal_resident.rs`). So for every non-gemma3 row the whole stack is inert in every
env state — asserted by `batched_windowed_prefill_never_arms_for_a_non_gemma3_arch`, now extended
to check the posture itself, including that `=0` is honoured (a default-on flag that ignores its
own opt-out is a flag with no off switch). The module is macOS+Metal-only, so no non-Metal host
sees any of it, and none of the three is in `MANAGED_ENV_KEYS`, so the execution planner never
writes them and cannot read its own output back as an opt-out (the §-latch failure mode).

**No shipped surface gained a throughput claim.** `performance_measured` stays
`not_claimed_resident_lane_throughput_is_a_separate_unshipped_measurement_phase`; no README,
COMPATIBILITY, STATUS or ledger row was edited.
### 19f. Deliberately not done

- **No new kernel, and almost no new MSL.** One `window` uniform on two existing kernels and
  about fifteen lines. `half_mm_batched` (the f32-output twin) was deliberately left alone: it is
  used by one ignored test, and giving the window to only the production variant keeps the
  surface smaller.
- **`prefill_tokens`' own `use_attn_mm` gate was NOT touched**, including its `!has_qk_norm` and
  `head_dim <= 128` clauses. §19a explains why the clause does not apply on the gemma3 seam;
  relaxing it *there* would be a change to every other row's admission and is not this phase's
  business.
- **`MAX_DPL` / `MAX_DOCT` and the `<= 128` host gates are byte-for-byte unchanged.** Tier B is a
  separate admission predicate over a chain with no per-lane fixed-size array.
- **The banded S panel was not built.** The campaign plan §6(3) proposed `[head][qb][576]` instead
  of `[head][qb][n_pad]`, exploiting the 9-tile window. It is unnecessary here: the chunk loop
  already blocks queries, so S and P are 10.0 MB at n = 2 400 and 33.5 MB at 8 192 — the plan's
  ~1 GB figure was for the untiled design. It stays available if the context ceiling is ever
  raised far enough to matter.
- **No second tuning attempt.** The stopping rule allowed two. The first measurement cleared the
  3x bar at 3.29x / 4.15x and showed per-token cost flat in prompt length, which is the shape the
  design predicted; there are no GPU counters on this host (§15) with which to chase the rest, so
  a second attempt would be guessing.
- **The 68 MB of f16 KV mirrors is no longer dead**, which closes an item owed since the campaign
  plan §7 — not by reclaiming the memory but by giving it a reader. If Tier B is ever disabled on
  this row the mirrors go back to being dead weight, and the reclaim is owed again.
- **No pack item softened, no failing leg excused, no bound moved.**

### 19g. WHERE PREFILL NOW STANDS, and what remains

Prefill at 2 400 prompt tokens is **0.817 ms/token served** (1.962 s), from 15.615 ms/token
(37.475 s) before the campaign. Decomposed with the same no-free-parameter model §18e used —
§18e's non-attention residual is 0.47 ms/token and this phase did not touch it:

| term | at 1 200 | at 2 400 | share at 2 400 |
|---|---:|---:|---:|
| weight GEMMs + norms + RoPE + scatter (unchanged since Phase 3) | 0.47 ms | 0.47 ms | **~58 %** |
| attention (was 2.016 / 2.659) | ~0.29 ms | ~0.29 ms | **~36 %** |
| serve-side residual (measured minus in-process probe) | 0.045 ms | 0.058 ms | ~6 % |

**The bottleneck has changed hands again, and it has gone back to the GEMM.** §18f(3) declined a
second GEMM tuning attempt on the grounds that the GEMM was 18 % of prefill and attention 78 %;
taking the GEMM to zero would have bought 18 %. It is now **~58 %**, and the same ~5x of nominal
headroom (1.708 TFLOPS whole-path against a ~3.4 TFLOPS Q8_0 wall) is worth roughly three times
what it was. That reopens the question §18f(3) closed — but it does not change the answer yet,
because the reason it was closed still holds: the chunk sweep already settled the one free
parameter, the traffic model closes to within 3-8 %, and there are **no GPU counters on this host**
(§15, entitlement-gated with a confirmed-dead headless route). The next GEMM attempt needs Xcode
on the T7 first, not another guess.

**The largest remaining item is now outside the engine.** Phase 0 measured the chat-path
tokenizer at `parse_special=true` costing **0.53 ms per prompt token**, i.e. **~1.27 s at 2 400
tokens**. When that was measured it was 3.7 % of a 35 s TTFT and 17 % of Phase 3's 7.573 s. It is
now **~65 % of the entire engine prefill** — a chat-path request at 2 400 tokens would spend more
time tokenizing than computing. Every TTFT number in this campaign takes prompts as exact token-id
arrays, so the tokenizer is deliberately out of the loop and none of these numbers include it.
**This is the single highest-value remaining item for real chat TTFT**, it is outside the engine,
and it is untouched by any tier.

Also still owed, unchanged: context above 2 403 prompt tokens is unmeasured on this row.

### 19h. Handover

**The flag stack, as it now ships.** All three default ON for the gemma3 Q8_0 row, `=0` to opt
out, and they nest: `CAMELID_GEMMA3_BATCH_PREFILL=0` restores the pre-campaign token-by-token
prefill exactly; `CAMELID_GEMMA3_PREFILL_MM=0` keeps Tier A's bit-identical batched GEMV and
turns off Phase 4 with it; `CAMELID_GEMMA3_PREFILL_ATTN_MM=0` keeps Phase 3 and restores per-row
attention. Every intermediate posture is a tested configuration, not a hypothetical: the gates pin
each side explicitly through `PrefillGemm` / `PrefillAttn`.

**Evidence bundle:** `qa/evidence-bundles/gemma3-1b-q8-prefill-attn-mm-20260731-head-95429a42/`,
including the FAILING first G6 run.

**For whoever picks this up:**

1. **The tokenizer** (§19g). Biggest remaining TTFT item for the chat path by a wide margin, and
   it is not a kernel problem.
2. **The GEMM, but only with a profiler.** ~58 % of prefill and ~5x of nominal headroom; blocked
   on §15's entitlement problem, not on ideas.
3. **Context above 2 403 tokens** is unmeasured. The S/P panels are linear in the chunk, not
   quadratic in the prompt, so 8 k costs 33.5 MB of scratch and the cap gate will not fire — but
   "will not fire" is arithmetic, not a measurement.
4. **The f16 KV mirrors are live now** (§19f). If Tier B is ever turned off on this row they go
   back to being 68 MB of dead allocation and the reclaim is owed again.
5. **The two depth-50 window-edge failures are still there**, unchanged since Phase 1, still
   unadjudicated, still reported as failures.
