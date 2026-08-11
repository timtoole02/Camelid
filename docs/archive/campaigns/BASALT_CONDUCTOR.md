# BASALT — NVFP4 weight format support (pilot: Gemma 4)

Conductor document. Status: **DRAFT — awaiting Tim sign-off on D-B decisions and G0 review.**
Date: 2026-07-16. Owner: Tim. Executor: Claude Code agent(s), one phase per session unless told otherwise.
Companion deliverable: `BASALT_RECON.md` (produced by Phase 0, checked into repo root alongside this file).
Prerequisite reading for the agent, in order: `REFERENCE_PIN_QWEN35.md`, `RECEIPTS.md`, `DECISIONS.md`, `src/runnable/dequant.rs` module doc, `src/gemma4_runtime.rs` module doc, `RUNNABLE_LANE_SPEC.md` (locate it; referenced from dequant.rs).

Campaign name: BASALT — black volcanic rock, for Blackwell; sits next to CAIRN in the rock pile, which is fitting because every claim this campaign produces terminates in a CAIRN ledger entry.

---

## 1. Mission

Land **NVFP4** (E2M1 4-bit elements, FP8-E4M3 per-16-element block scales, plus a per-tensor F32 scale) as a first-class weight format in Camelid:

1. Load and run NVFP4 GGUF files produced by the pinned llama.cpp toolchain — CPU lane first, token-parity receipted.
2. CUDA dequant-in-kernel decode path that works on **any** supported CUDA GPU (pre-Blackwell included) — the bandwidth win is the near-term payoff.
3. A hardware-gated Blackwell (sm_120/sm_100) tensor-core MMA path — attempted only if real Blackwell silicon is available; otherwise recorded as BLOCKED-HW with zero claims.
4. Surface alignment: capability matrices, README, CAIRN ledger, and Evidence Chip states all updated in the same PR that makes the claim, drift gate green.

Pilot model: **the Gemma 4 model already in the validated lane** (the one whose forward is bit-for-bit anchored in `tests/gemma4_forward.rs` and served by `src/gemma4_runtime.rs`). Phase 0 records the exact model, size, and GGUF provenance in the recon doc.

### Why this is worth a campaign

- NVFP4 costs 9 bytes per 16 weights = **4.5 bits/weight** on quantized tensors, vs 8.5 for Q8_0. The Gemma 4 runtime currently holds ~8 GB of Q8_0-resident weights; the projection is ~4.2 GB at NVFP4 plus keep-list overhead. That is the difference between "partial offload" and "fully CUDA-resident" on a 6 GB card. Phase 0 replaces this projection with an inventory-derived number; do not quote 4.2 GB anywhere user-facing.
- Decode is bandwidth-bound (SIROCCO's whole premise). Bytes moved per token drop ~1.9× vs Q8_0; measured decode speedup is whatever it is — Phase 4 measures it, nobody promises it.
- Unlike MXFP4 (E8M0 power-of-two scales, 32-element blocks), NVFP4's fractional E4M3 scales on 16-element blocks plus the second-level tensor scale give materially lower quantization error. Whether it beats Q4_K_M in quality is an **open question upstream** — Phase 3 answers it for our pilot with a pre-registered protocol, and the answer is allowed to be "it doesn't."
- Ecosystem: NVFP4 merged into llama.cpp as `GGML_TYPE_NVFP4` (reported type id 40) across late March–April 2026, with the Blackwell-native MMQ path enabled around build b8967 (2026-04-29) for sm_120. Our pin `acd79d6` is **build 9632 (2026-07-02)** — it should contain all of this. Phase 0 verifies; nothing downstream proceeds on "should."

## 2. Non-goals (v1)

- Activation quantization, FP4 KV cache, attention in FP4. KV stays on the existing `kv_f16` path.
- MXFP4.
- The Gemma 4 MoE variant and any multimodal (vision/audio) tensor paths. Text decode path of the pilot model only.
- Metal/NEON NVFP4 paths. CUDA + CPU reference only. (Metal gets a follow-on campaign if v1 lands.)
- imatrix-style calibrated quantization. NVFP4 weight-only is data-free; calibration is out of scope.
- Supported-lane promotion. BASALT lands NVFP4×Gemma-4 in the **Runnable lane**. Promotion to Supported follows the existing exact-row evidence policy as a separate, later decision.

## 3. Format specification (normative, with one arbiter)

**The arbiter for every byte-level question is the pinned llama.cpp implementation at `acd79d6`** (`ggml/src/ggml-quants.c`, `ggml/src/ggml-common.h`, and the CUDA/CPU consumers of the type). The NVIDIA NVFP4 recipe (NVIDIA developer blog, "Introducing NVFP4," 2025) is the cross-check. Where the two disagree, match the pin bit-for-bit and record the deviation in the design note. We interoperate with files, not with blog posts.

### 3.1 Element format — FP4 E2M1

1 sign bit, 2 exponent bits, 1 mantissa bit. Full value table (low 3 bits; sign bit high):

| bits | value | bits | value |
|---|---|---|---|
| 000 | 0.0 | 100 | 2.0 |
| 001 | 0.5 | 101 | 3.0 |
| 010 | 1.0 | 110 | 4.0 |
| 011 | 1.5 | 111 | 6.0 |

Negative zero (1000) exists. Max magnitude 6.0. Two elements per byte; nibble order per the pin's packing, not per assumption.

### 3.2 Block format

16 consecutive elements share one FP8 **E4M3** scale: 8 bytes of packed nibbles + 1 scale byte = 9 bytes/block. E4M3 facts that will bite if ignored: bias 7, max finite 448, **no infinities**, NaN = S.1111.111. A NaN or zero scale byte in an incoming file is a load-time error, never a silent zero — fail closed, same posture as `runnable/dequant.rs`.

### 3.3 Two-level scaling and the per-tensor scale

Reference recipe (verify against pin):

- `s_tensor` (F32) = `amax(tensor) / (6 × 448)` — chosen so block scales fit E4M3 range.
- Stored block scale `S_b` = `cast_e4m3(amax(block) / (6 × s_tensor))`.
- Element quantization: `q_i = nearest_e2m1(w_i / (decode_e4m3(S_b) × s_tensor))` — note the division is by the **decoded** (post-cast) scale, so E4M3 rounding of the scale is accounted for. If the pin divides by the pre-cast scale instead, match the pin and flag it.
- Dequant: `ŵ_i = e2m1(q_i) × decode_e4m3(S_b) × s_tensor`.

**CRITICAL — per-tensor scale plumbing.** Where `s_tensor` physically lives and who applies it was the subject of a documented upstream functional-correctness dispute (llama.cpp discussion #22042: in-block vs. a post-GEMM `ggml_mul` node vs. a tensor-field slot, with layout-breaking proposals still live as of spring 2026). Whatever the pin actually does is our wire truth. Consequences for us:

- Camelid's `q*_wire_row_dot` signatures (`src/inference`) do not carry a per-tensor scale today. NVFP4 either threads a scale parameter through its own `nvfp4_wire_row_dot`, or applies it once per output row/tensor at a defined seam. **Silently pre-folding `s_tensor` into block scales is forbidden** — it can saturate E4M3 and is exactly the kind of quiet corruption this project exists to prevent. This is decision D-B2.
- Every consumer (CPU dequant, CPU wire-dot, CUDA kernel, runnable-lane decode) gets a test that fails if the tensor scale is dropped or applied twice. A 2× or 1/448× logit error is embarrassingly easy to produce here.

### 3.4 Enum and dispatch touchpoints

- `src/gguf/reader.rs` — `GgufTensorType`: add `NVFP4`, wire `from_id` with the **id extracted from the pin's headers** (reportedly 40; extract, don't trust). Today the id maps to `Unknown(i32)` and admission fails closed — Phase 0 captures that refusal as the baseline receipt.
- `crate::tensor` — `Nvfp4Block` + `decode_nvfp4_tensor`, following the existing `decode_q4_k_tensor` conventions.
- `crate::inference` — `nvfp4_wire_block_dequant` + `nvfp4_wire_row_dot`, following the `q4_k_wire_row_dot` pattern (dot on mmap'd wire bytes; rayon over rows).
- `src/runnable/{admit.rs, dequant.rs, smoke.rs}` — extend the v1 covered set per the lane's single-dispatch principle. Anything not NVFP4-ready continues to refuse.
- `src/gemma4_runtime.rs` — a resident-weights path that mirrors the Q8_0 residency but holds NVFP4 wire blocks and calls the NVFP4 matvec.
- `src/cuda_resident` — NVRTC-compiled (cudarc 0.19) dequant-in-kernel matvec for Phase 4; block-scaled MMA for Phase 5.

## 4. Oracle and parity policy

Same oracle, new lane. `REFERENCE_PIN_QWEN35.md` pins `acd79d6` build 9632 with the full tool set (`llama-quantize.exe`, `llama-cli.exe`, `llama-perplexity.exe`, CUDA 12.9 build, Windows target). BASALT reuses that exact pin and build:

- **Golden quantizer (v1):** NVFP4 GGUFs are *produced by the pin's* `llama-quantize`, not by Camelid. Camelid v1 is consume-side. (A native Camelid quantizer with byte-parity against the pin tool is Phase 2b, optional.)
- **Golden vectors:** decode tables and random-block dequant outputs generated from the pin's reference C functions, checked into `fixtures/`, asserted byte-identical by Camelid unit tests.
- **Token parity (CPU lane):** greedy decode on the same NVFP4 file, Camelid vs pin `llama-cli` CPU — target token-identical, same as the TinyLlama-era discipline. If cross-engine token-identity proves unreachable for a reason we can name (accumulation order), the fallback is the existing parity-harness tolerance methodology, with the reason documented in the recon, not hand-waved.
- **Quality oracle:** pin's `llama-perplexity` on the same file provides the cross-engine perplexity row.
- **Self-parity (GPU):** CUDA path vs Camelid CPU path under existing `qa/determinism` conventions.

If Phase 0 discovers the pin does **not** contain NVFP4 (date math says it should; receipts say what's true), STOP. The fallback is a second, BASALT-scoped pin document (`REFERENCE_PIN_NVFP4.md`, same table format as the Qwen35 pin) at the earliest stable post-merge build — Tim approves the SHA before anything else runs.

Upstream flux warning: #22042-adjacent proposals could change the NVFP4 wire layout in future llama.cpp. We freeze on the pin. Add a CAIRN watch item so a future re-pin forces an explicit compatibility check rather than a silent assumption.

## 5. Phase plan

Rules that apply to every phase: evidence bundles land under `qa/evidence-bundles/basalt/phase<N>/` per `RECEIPTS.md` conventions; every gate ends with STOP for human review; no README/matrix edits before Phase 6; a NO-GO at any gate is a valid, respectable outcome that terminates in a postmortem note, not in threshold shopping.

### Phase 0 — Recon and pre-registration → Gate G0

Work:
1. **Pin verification.** In the pin's llama.cpp checkout at `acd79d6`: confirm `GGML_TYPE_NVFP4` exists; extract type id, block struct layout (field order, sizes, nibble packing), per-tensor scale mechanism (§3.3), and the rounding used by the reference quantizer. Capture file/line receipts (grep output, header excerpts) into the recon doc.
2. **Hardware probe.** Record GPU(s), SM version, driver, CUDA toolkit on the Windows target into `hw_probe.json`. Determine whether any available device is sm_100/sm_120+. This single fact schedules or blocks Phase 5.
3. **Baseline refusal receipt.** Feed a pin-quantized NVFP4 GGUF (tiny model is fine) to current Camelid `main`; capture the fail-closed refusal. This is the "before" photo.
4. **Gemma 4 tensor inventory.** Enumerate the pilot's tensors: names, shapes, current types, K%16 divisibility, and a proposed keep-list (embeddings, output head — check whether the pilot ties lm_head to the embedding, norms, any tensor the pin's quantizer itself keeps high-precision; mirror the pin's per-tensor type choices exactly and record them). Emit `tensor_inventory.json` + derived size table: Q8_0-resident bytes today vs projected NVFP4-resident bytes.
5. **Eval pre-registration.** Commit `basalt_eval_protocol.md` BEFORE any NVFP4 quality number exists: fixed prompt set (content + sha256), greedy decode, N tokens per prompt, perplexity corpus slice (content + sha256), metrics (top-1 agreement vs Camelid Q8_0 baseline; KL on logits over the eval slice; ppl via both Camelid and pin `llama-perplexity`), comparison rows (Q8_0 baseline, Q4_K_M, NVFP4), and the GO rule for G3 (default in §6, Tim may adjust only before P3 runs).
6. Write `BASALT_RECON.md` consolidating 1–5, plus D-B decision drafts.

Gate G0: all five artifacts exist; D-B1–D-B5 drafted with a recommendation each; pin verified NVFP4-capable (or STOP → fallback pin proposal). Human review.

### Phase 1 — Format core, pure Rust, no CUDA → Gate G1

Work: `Nvfp4Block`, encode/decode LUTs, `decode_nvfp4_tensor`, `nvfp4_wire_block_dequant`. Tests: exhaustive decode table (all 256 scale bytes × all 16 element codes) asserted byte-identical to pin-generated golden vectors; ≥10k random pin-quantized blocks dequantized identically; property tests (representable-value round-trip, scale saturation at 448×6, zero-block, NaN-scale rejection); per-tensor scale application unit-tested against the "dropped/doubled" failure modes. No `unsafe`. CPU only.

Gate G1: `cargo test` green including new suites; golden-vector fixtures checked in with provenance (pin SHA + generator script in `tools/` or `scripts/`); bundle written.

### Phase 2 — GGUF load and cross-engine interop → Gate G2

Work: enum + reader wiring; runnable-lane admission/dequant/smoke coverage for NVFP4; quantize the pilot Gemma 4 with the pin's `llama-quantize` to NVFP4; load it in Camelid; full-tensor dequant spot-checks vs pin dequant on sampled tensors.

Gate G2: pilot NVFP4 GGUF loads clean; sha256 + tensor-type inventory receipts; measured file/resident size table replacing the §1 projection; sampled-tensor dequant parity receipts. Optional Phase 2b (separate session, only after G2): native `camelid quantize --type nvfp4` with byte-parity vs the pin tool.

### Phase 3 — CPU inference and pre-registered quality eval → Gate G3

Work: `nvfp4_wire_row_dot`; `gemma4_runtime` NVFP4-resident path (mirror Q8_0 residency; same KV, RoPE, soft-cap, KV-sharing logic — the forward math does not change, only the weight decode); run the pre-registered protocol exactly.

Gate G3: (a) token-parity receipt vs pin `llama-cli` CPU on the same file per §4; (b) quality table with all pre-registered rows and both ppl sources; (c) GO/NO-GO applied per the pre-registered rule, decision recorded either way. NO-GO here means NVFP4 quality on this pilot is not competitive — the campaign pivots to a postmortem + upstream comparison note, and that is a publishable result too.

### Phase 4 — CUDA dequant decode path (any GPU) → Gate G4

Work: NVRTC dequant-in-kernel matvec mirroring the existing CUDA matvec structure; residency plumbing in `src/cuda_resident`; per-tensor scale applied in exactly one place (kernel epilogue or host-side row scale — per D-B2). Self-parity vs the Phase 3 CPU path under `qa/determinism` conventions. Perf: decode tok/s and achieved GB/s vs Q8_0 and Q4_K_M on the same box, same prompts, SIROCCO measurement hygiene (WDDM noise discipline, warm-up policy, N runs, medians).

Gate G4: parity receipts + perf table. Claim discipline: any speedup number carries the hardware row and bundle id, and is phrased as measured-on-this-box, never as a general claim.

### Phase 5 — Blackwell block-scaled MMA path → Gate G5 (HARDWARE-GATED)

Precondition: `hw_probe.json` shows sm_100/sm_120+ silicon actually attached. If not: write a one-paragraph BLOCKED-HW record into the ledger and STATUS.md, skip the phase entirely, make no forward-looking performance statements. Do not simulate, do not extrapolate, do not "should be roughly."

If hardware exists: prefill-side GEMM via PTX block-scaled MMA (e2m1 elements, E4M3 scale operands; NVRTC target `sm_120a` — verify the pin-era CUDA 12.9 NVRTC accepts it, else precompiled PTX per D-B4), with the pin's post-b8967 NVFP4 MMQ kernels as the reference implementation to study. Correctness vs the Phase 4 dequant path; prefill TTFT/throughput receipts.

Gate G5: correctness + perf receipts on named Blackwell hardware, or the BLOCKED-HW record. Both are green outcomes.

### Phase 6 — Surface alignment → Gate G6

Work: CAIRN ledger entries (`ledger/camelid-ledger.json`, schema-valid) for every user-visible NVFP4 statement, each citing bundle ids; rows/updates in `CAPABILITY_MATRIX.md`, `SUPPORT_MATRIX_v0.1.md` (Runnable lane), `STATUS.md`, README; Evidence Chip state in the frontend (copper only for receipt-backed states — NVFP4 quality vs Q4_K_M gets stated with its measured numbers, not adjectives); docs page containing §3 plus the pin-layout findings; a QUANT_TRUTH-style self-consistency pass across all matrices (no table may contradict another — that failure mode has happened before and has a conductor named after it).

CI: extend coverage minimally; `[ci-os: windows]` commit tag for iteration; new jobs must aggregate under the `ci-gate` required check.

Gate G6: CAIRN drift check green; matrix self-consistency receipt; PR merged.

## 6. Pre-registered defaults (Tim-adjustable only before the relevant phase runs)

- G3 GO rule: NVFP4 top-1 agreement vs the Q8_0 baseline must be within **2.0 points** of Q4_K_M's top-1 agreement vs the same baseline, on the pre-registered set; KL and ppl reported alongside, no gate on them in v1.
- Token parity target: token-identical for 8 prompts × 128 greedy tokens, CPU vs pin CPU.
- G4 self-parity: existing CUDA-vs-CPU tolerance conventions from `qa/determinism` apply unchanged; NVFP4 gets no looser budget than Q4_K.
- Perf receipts: median of ≥5 runs, warm, fixed context length, reported with achieved-bandwidth calculation per the SIROCCO method.

## 7. Decision records (draft in P0, number into DECISIONS.md sequence on acceptance)

- **D-B1 — Wire-compat target.** Adopt the pin's `GGML_TYPE_NVFP4` layout byte-for-byte (recommended) vs a Camelid-private layout. Recommendation rationale: interop receipts both directions, pin tools as golden quantizer/oracle, zero format invention.
- **D-B2 — Per-tensor scale seam.** Thread `s_tensor` explicitly through the NVFP4 matvec/kernel signatures (recommended) vs fold at load. Folding into block scales is rejected outright per §3.3.
- **D-B3 — Runnable-lane scope.** NVFP4 joins the runnable-lane v1 covered set now, vs pilot-model-only until G4. Recommendation: pilot-only until G3 passes, then lane-wide.
- **D-B4 — Phase 5 kernel route.** NVRTC `sm_120a` runtime compile vs precompiled PTX/cubin shipped per-arch. Decide only if Phase 5 unblocks.
- **D-B5 — Quantizer ownership.** Pin-tool-only for v1 (recommended) vs native quantizer in-scope (Phase 2b).

## 8. Sharp edges (agent: read twice, test each)

- E4M3 scale saturation at 448 — a block whose amax exceeds `448 × 6 × s_tensor` clips; the pin's quantizer chooses `s_tensor` to prevent it, and our tests must cover the boundary.
- NaN/zero scale bytes in-file → load error, fail closed (§3.2).
- Per-tensor scale dropped or double-applied — the single most likely functional bug in this campaign (§3.3). Dedicated tests in every consumer.
- Tied embedding/output in the Gemma family — if the pilot ties them, the keep-list decision affects the logit path twice. Mirror the pin's choice; receipt it.
- K%16 ≠ 0 tensors: keep source precision, never pad silently.
- Prompt-prefix cache and any cached-state must key on tensor type/quant identity — the PR #419 warm-hit bypass lesson applies verbatim; an NVFP4 load must never reuse Q8-era cached state.
- WDDM timing noise on every Windows perf number — STAMPEDE hygiene, no exceptions.
- Upstream layout flux (#22042) — frozen at pin; CAIRN watch item on re-pin.
- `GgufTensorType::Unknown(i32)` ordering/serde: adding a real variant must not disturb existing serialized receipts or ledger entries that mention type names — check `src/receipt/verify.rs` consumers.

## 9. Evidence classes and bundle layout

Per DRAYAGE conventions: **HARNESS** = machine-checked in CI (unit/property/golden-vector suites, self-parity tests); **CERT** = end-to-end receipts on named hardware (token parity, quality table, perf tables), bundled and hash-manifested under `qa/evidence-bundles/basalt/phase<N>/`, referenced by id from CAIRN entries and any prose claim. If the runs are agent-driven, LOOP_RUNTIME hash-chained iteration receipts apply.

## 10. Agent conduct

One phase per session. Read the gate before writing code. STOP at every gate even if confident. Never edit README, STATUS, or matrices before Phase 6. Never invent a number a receipt doesn't contain. If the pin contradicts this document, the pin wins and the discrepancy goes in the recon. If anything is ambiguous, ask Tim; do not pick silently. A NO-GO with a clean postmortem is a success state.

## 11. Open items for Tim (answer before G0 review)

1. Blackwell access: is there any sm_120 device available (local or borrowable) in the campaign window? Determines whether Phase 5 is scheduled or pre-declared BLOCKED-HW.
2. Confirm the pilot = the current validated-lane Gemma 4, or name a different size.
3. Sign off / adjust the §6 pre-registered defaults.
4. D-B1–D-B5 recommendations: accept or amend.
