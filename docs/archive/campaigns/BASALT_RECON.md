# BASALT_RECON — Phase 0 recon and pre-registration (Gate G0)

Campaign: BASALT (NVFP4 weight format, pilot Gemma 4 E4B-it). Date: 2026-07-16.
Status: **G0 SIGNED 2026-07-16 (Tim: T1 no Blackwell; T2–T7 accepted as recommended → DECISIONS.md D17).** Evidence bundle: `qa/evidence-bundles/basalt/phase0/`.
Conductor: `BASALT_CONDUCTOR.md`. Eval pre-registration: `basalt_eval_protocol.md` (committed
with this document, before any NVFP4 quality number exists).

Method note: recon ran as six parallel lanes (pin source extraction, Camelid touchpoints,
hardware probe, pilot tensor inventory, baseline refusal, size projection) plus a follow-up
ecosystem scan. Lane receipts: `pin_extraction_receipts.md`, `camelid_touchpoints.md`,
`hw_probe.json`, `tensor_inventory_raw.json` / `tensor_inventory.json`, `refusal_receipt.md`
in the bundle. One incident occurred and is receipted: `incident-20260716-hard-hang.md`.

## 0. Executive summary

1. **Pin verified NVFP4-capable, and the golden-quantizer premise holds — via the
   per-tensor override path.** Type id 40 (`nvfp4`, `LLAMA_FTYPE_MOSTLY_NVFP4`=39) exists in
   pin `acd79d603` (build 9632) with CPU + CUDA consumers. There is no NVFP4 positional ftype
   (upstream never merged one), but `llama-quantize --tensor-type '<regex>=nvfp4'`
   **empirically and deterministically converts Q8_0 → NVFP4** (proven on Qwen3-0.6B: 8.50 →
   4.50 BPW, byte-identical across repeat runs; §5). The override regex controls the
   keep-list exactly, which upgraded the eval design to a format-isolated comparison
   (`basalt_eval_protocol.md` §1).
2. **The wire format differs from the conductor's §3 in load-bearing ways** (§2): a 64-element
   36-byte superblock (not a bare 16-element block), an *unsigned* E4M3 scale whose `0x7F`
   NaN sentinel the pin CPU silently flushes to 0.0 (**`0xFF` decodes to 240.0 — the pin's
   CPU and CUDA backends disagree on it; §1 [G1 errata]**), a doubled element LUT paired
   with the scale convention (the pair rule), and **no in-block per-tensor scale** — the
   per-tensor scale is an *optional sidecar tensor* mechanism used only by Python-converted
   ModelOpt checkpoints.
3. **The 6 GB full-residency motivation is refuted for this pilot** (§7): projected NVFP4
   resident weights are **6.204 GB** (not ~4.2 GB), because 3.71 GB of embeddings
   (`per_layer_token_embd` + tied `token_embd`) are kept at Q8_0 by every known production
   path. The decode-bandwidth motivation survives intact: the 294 matmul weights shrink
   1.889× (4.19 → 2.22 GB read per token), and a matmul-only GPU-resident set (~2.44 GB, with
   file-backed embeddings — the Metal lane's existing pattern) fits the 6 GB card comfortably.
4. **No Blackwell silicon on this box** (RTX 3060 Laptop, sm_86): Phase 5 is pre-declared
   **BLOCKED-HW** unless Tim names borrowable sm_120/sm_100 hardware (§6).
5. Baseline refusal receipt: current `main` fails closed on NVFP4 at GGUF **parse** (not
   admission) — captured before any enum change (§5, `refusal_receipt.md`).
6. A hard machine hang occurred mid-recon, caused by an orphaned `llama-cli` REPL spin —
   root-caused, receipted, and turned into binding harness rules (`llama-completion` only for
   scripted generation; orphan sweep after any agent death). See the incident file.

## 1. Item 1 — Pin verification (normative format facts)

Arbiter: pin `acd79d603` build 9632 source. Full receipts with file:line and header excerpts:
`pin_extraction_receipts.md`. The facts a Rust implementation must not get wrong:

- **Type**: `GGML_TYPE_NVFP4` = **40**, name string `nvfp4`. Two distinct ftype enums exist —
  `GGML_FTYPE_MOSTLY_NVFP4` = 26 (ggml) vs `LLAMA_FTYPE_MOSTLY_NVFP4` = **39** (llama.h; this
  is what `general.file_type` carries in a GGUF). Camelid's receipt map keys on the latter (§5).
- **Block struct** (`ggml-common.h:211-217`): `block_nvfp4` = **36 bytes per 64 elements**
  (`QK_NVFP4=64`, `QK_NVFP4_SUB=16`): `uint8_t d[4]` (four UE4M3 sub-block scales) **first**,
  then `uint8_t qs[32]` (packed E2M1 nibbles). `static_assert sizeof==36`.
- **Nibble packing**: sub-block `s` (0..3) occupies `qs[s*8 .. s*8+7]`; **low** nibble of byte
  `s*8+j` = sub-block element `j` (0..7), **high** nibble = element `8+j` (8..15) — the
  MXFP4-style half/half split, *not* adjacent-element pairing.
- **Element LUT** (`kvalues_mxfp4`, `ggml-common.h:1116-1118`):
  `{0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}` — E2M1 magnitudes **doubled**; nibble bit 3 =
  sign. Value = `kvalues[nibble] * ue4m3_to_fp32(d[s])`.
- **Scale format is UE4M3 — unsigned E4M3** (`ggml-impl.h:500-553`): bit 7 stripped (`&0x7F`),
  bias 7, subnormals. **[G1 errata, fixture-arbitrated]** Two Phase 0 prose claims were wrong
  and are corrected by the pin-generated golden vectors (2.17 M assertions, Phase 1 bundle):
  (i) the CPU decode's sentinel check is on the **raw byte `0x7F` only** — `0x00` → 0.0,
  `0x7F` → 0.0, but **`0xFF` decodes through exp/man to 240.0**; the pin's CUDA mirror
  flushes both, i.e. **the pin's own backends disagree on `0xFF`**, which strengthens the
  D17/T5 admission refusal of both bytes (Camelid decode is pin-CPU-bitwise; admission
  refuses files containing either). (ii) `decode(0x7E)` = **224.0**, not 112.0; the encoder
  saturates every input ≥ 248 to `0x7E`, so bytes `0x78..0x7D` are decodable but
  encoder-unreachable. The doubled-LUT/scale-convention **pair rule** stands as the headline
  hazard (12 × 224 = 2688 = the true 6 × 448 max — the factor-of-two lives across the pair);
  the committed fixtures, not prose, are the normative statement of both conventions.
  Encoder clamps input to 448.0, rounds half-up; quantizer feeds `amax(sub-block)/6.0`.
- **Element rounding** = exhaustive nearest-LUT search (`best_index_mxfp4`,
  `ggml-quants.c:299-310`), first-wins ties — **not** IEEE round-nearest-even.
- **Per-tensor scale**: no in-block or GGUF-KV tensor-level factor exists. The mechanism is
  **optional sidecar F32 tensors** `<name>.scale` (`weight_scale_2`, shape {1} or {n_expert})
  and `<name>.input_scale`, created only by the Python `convert_hf_to_gguf` repack of
  ModelOpt/compressed-tensors checkpoints and applied **post-matmul** via a `ggml_mul` node
  (`llama-graph.cpp:1085-1114`); all `TENSOR_NOT_REQUIRED`. `llama-quantize`-produced NVFP4
  carries **no sidecars** and no tensor factor of any kind. Tied-embedding guard:
  `llama-model.cpp:1472-1477`.
- **CPU consumers**: `vec_dot_type = Q8_0`; x86 gets only the **generic scalar** kernel
  (`arch-fallback.h:83-85` aliases it — no AVX2/AVX-512 NVFP4 kernel in this pin); real SIMD
  exists only for ARM. Consequence: pin-side CPU reference legs are slow but well-defined,
  and any future Camelid x86 SIMD NVFP4 kernel is greenfield.
- **CUDA consumers**: dequant (`convert.cu:620-658`); **MMVQ dp4a gemv on all NVIDIA archs**
  (`vecdotq.cuh:331-359`) — a working decode-path reference for our sm_86 box; MMQ with a
  Blackwell-native FP4 tensor-core path (`mma.cuh:1125-1154`,
  `kind::mxf4nvf4.block_scale.scale_vec::4X ... ue4m3`) gated to sm_120..<Rubin, with a
  dequant-to-q8 MMA fallback elsewhere.
- **K%64**: conversion hard-requires `K%64==0` (`base.py:672`; gguf-py raises).
  `llama-quantize` with an NVFP4 override **throws per incompatible tensor** — `QK_NVFP4=64`
  and `tensor_type_fallback` (`llama-quant.cpp:362-408`) has no NVFP4 case, so `ncols%64≠0`
  hits the `default:` throw "no tensor type fallback is defined for type nvfp4" (`:390-392`).
  Not a blanket inability — it never fired on the qwen3 test model, and the E4B pilot's
  quantized tensors are all `K%64==0` (§4).
- Upstream flux: the pin source contains **no reference to discussion #22042**; its sole
  design citation is the OCP MX v1.0 spec URL. The CAIRN watch item (§10, W1) still applies.

## 2. Conductor errata (pin wins; §3 arbiter rule applied)

| # | Conductor says | Pin truth |
|---|---|---|
| E1 | §3.2: 16 elems + 1 scale byte = 9-byte block | 64-elem, 36-byte superblock `{d[4], qs[32]}`; 9 B/16 elems is the *density*, not the struct |
| E2 | §3.2: scale is (signed) E4M3, NaN = S.1111.111 | **Unsigned** E4M3, bit 7 stripped; NaN sentinel `0x7F` |
| E3 | §3.2: NaN/zero scale byte = load-time error | Pin CPU **flushes raw `0x7F` to 0.0** silently; **[G1 errata]** `0xFF` decodes to 240.0 on CPU while the CUDA mirror flushes it — pin backends disagree; zero scale is a *legitimate* all-zero block, not an error → open item T5 |
| E4 | §3.3: two-level scaling with `s_tensor` in the format | No in-block tensor scale; optional **sidecar tensors** applied post-matmul; `llama-quantize` files have none |
| E5 | §3.3: element quantization rounding `nearest_e2m1` | Exhaustive nearest-LUT search, first-wins ties (not RNE); scale rounding = half-up |
| E6 | §1: ~4.2 GB NVFP4 projection | 6.204 GB (embeddings kept Q8_0); see §7 |
| E7 | §2: KV stays on the existing `kv_f16` path | The gemma4 runtime KV cache is **f32** (`gemma4_runtime.rs:489-495`); oracle flags `-ctk f32 -ctv f32` |
| E8 | §9: "DRAYAGE conventions", HARNESS/CERT classes | No such taxonomy exists in the repo. Actual conventions: harness-emitted vs server-claim parity receipts; schema families (`camelid.parity-receipt/v1`, `capability-receipt/v1`, `quality-tier/v1`, `raw_decode_parity.v1`); bundles = `manifest.json` + `SHA256SUMS` |
| E9 | Prereq reading `RUNNABLE_LANE_SPEC.md` | Not in the repo (external doc). In-repo authorities: `src/runnable/admit.rs` covered set ("taken verbatim from the spec") + `RUNNABLE_LANE_RECON.md` |
| E10 | §4/§6: token parity "8 prompts × 128", speed-column style | Amended to gemma4 lane-native packs (9 prompts / 320 tokens): gemma is SPM and MUSTER sealed that SPM rows can't ride the speed-column raw pack. See `basalt_eval_protocol.md` §3 |
| E11 | §4: "id reportedly 40 … build 9632 should contain all of this" | Confirmed: id 40, all consumers present |

## 3. Item 2 — Hardware probe

`hw_probe.json`: RTX 3060 Laptop GPU, CC 8.6 (sm_86, Ampere), driver 576.83, 6144 MiB VRAM;
CUDA toolkit 12.9 (V12.9.86), NVRTC 12.9; Windows 11 build 26220.8764. **No sm_100/sm_120
device.** Consequences: Phase 5 pre-declared BLOCKED-HW (open item T1); Phase 4 targets the
dequant-in-kernel path, for which the pin's own dp4a MMVQ is a same-box reference; D-B4 is
moot unless hardware appears.

## 4. Item 4 — Pilot identity and tensor inventory

Pilot (confirmed identity for open item T2): `gemma-4-E4B-it-Q8_0.gguf` from
`unsloth/gemma-4-E4B-it-GGUF`, sha256 `a2232a64…` (full value in
`basalt_eval_protocol.md` §1), 8,192,951,456 B — byte-matched between the HF LFS oid and the
two committed evidence-bundle manifests. Currently **not on disk**; Phase 2 re-downloads it
plus the BF16 sibling (`21eb0c95…`, 15.05 GB — archival source only per protocol
Amendment 2; produced rows quantize from the Q8_0 baseline) and the upstream imatrix.

Inventory (from the 64 MiB header fetch; parser + raw dump in the bundle): GGUF v3,
`arch=gemma4`, 720 tensors — 296 Q8_0, 423 F32, 1 BF16. Key KVs: 42 blocks, d_model 2560,
ffn 10240, heads 8/kv 2, head_dim 512 (SWA 256), ctx 131072, `shared_kv_layers=18`,
final softcap 30, `general.file_type=7`.

- **Tied lm_head**: no `output.weight`; logits are always `token_embd.matvec` in the runtime
  (`gemma4_runtime.rs:1671`). NVFP4-quantizing the head means quantizing the embedding table
  (conductor §8 hazard confirmed live).
- **K%64 all-clear**: every quantized (Q8_0) matrix has K%64==0; the only violations are 42
  F32 `layer_output_scale` scalars that no path would ever repack.
- Byte closure: header 15,824,717 B + tensor bytes 8,177,125,544 B + 1,195 B alignment
  padding = file size, exact; all 720 per-tensor recomputations matched.

## 5. Item 3 — Baseline refusal receipt + the quantize discrepancy

Full receipt: `refusal_receipt.md` (bundle). Summary:

- **Refusal captured, fail-closed at parse.** Against `main` `4f9603f0`, both
  `camelid inspect` and `camelid runnable-smoke` on a real pin-produced NVFP4 GGUF exit 1
  with `unsupported GGUF feature: tensor token_embd.weight has unknown or removed GGML type
  Unknown(40)` (`src/gguf/reader.rs:530-535` — `token_embd` is simply the first NVFP4 tensor
  the parser reaches). Refusal point is **parse-time**, *not* the admission gate — adding the
  enum variant will move it to admission and change the refusal-text class, which is why this
  baseline was captured first.
- **Test artifact**: `models/qwen3-0.6b-NVFP4-basalt-refusal.gguf`, sha256 `7337b616…9146`,
  341,454,496 B, produced by the pin's quantizer from the local Qwen3-0.6B Q8_0
  (`9465e63a…b031`): 197 tensors q8_0→nvfp4 (incl. `token_embd`, tied head), 113 kept f32;
  604.15 → 319.96 MiB (4.50 BPW). **Determinism: byte-identical sha on an independent re-run**
  despite multithreaded quantization. Kept on disk for Phase 1 golden-vector work.
- **Discrepancy resolved — both earlier source claims were true, the conclusion was wrong.**
  There is no NVFP4 positional ftype (`llama_ftype_get_default_type`, `llama-quant.cpp:792-833`,
  has no case — stands) and `tools/` greps clean for "nvfp4" (stands). The working path:
  `--tensor-type '.*=nvfp4'` → `parse_tensor_type` (`tools/quantize/quantize.cpp:313-343`) →
  `parse_ggml_type` (`:301-311`), which matches **all** ggml type names case-insensitively
  against the trait table where `GGML_TYPE_NVFP4=40` is registered as `"nvfp4"`
  (`ggml/src/ggml.c:744-751`). The positional ftype stays Q8_0 (bypassing the "invalid output
  file type" throw), per-tensor overrides apply at `llama_tensor_get_type` (`:678-691`), and
  `general.file_type=39` (`LLAMA_FTYPE_MOSTLY_NVFP4`, `llama.h:156`) is set via
  `--override-kv` post-write. Cosmetic trap: the log header says "quantizing … as Q8_0".
- **Pin sanity generation** (validity proof): `llama-completion.exe` (sha `9547c455…`,
  `-no-cnv --no-warmup`, timeout-wrapped, RAM-checked) on the NVFP4 file: *"The capital of
  France is Paris. The capital of France is also"*, exit 0; loader reports
  `type nvfp4: 197 tensors / file type = NVFP4` (loader dump is verbosity-gated in this
  build — needed `-v`). Tool history disclosed in the receipt: the first attempt used
  `llama-cli` and caused the incident (§0.6).
- **ftype numbering correction**: the receipt-label map (`src/receipt/mod.rs:235-256`) needs
  an entry for **39** (`LLAMA_FTYPE_MOSTLY_NVFP4` — what `general.file_type` actually
  carries), not "26" as the touchpoints lane stated (26 is `GGML_FTYPE_MOSTLY_NVFP4`, a
  different enum). Add it in the same change as the enum variant, and pick the variant's
  Debug name deliberately — it becomes the receipt-visible quantization label.
- `tensor_inventory.json` carries a correction note referencing the resolved discrepancy.

## 6. Item 5 — Eval pre-registration

Committed as `basalt_eval_protocol.md` (repo root) with every input pinned by sha256:
rows Q8_0-baseline / NVFP4 / Q4_K_M-df (gated comparator) / Q4_K_M-im (report-only); gemma4
lane-native packs (9 prompts, 320 greedy positions) replacing the §6 speed-column default
(E10); teacher-forced top-1 agreement as the gated metric with the §6 2.0-point GO rule
unchanged; exact full-logit KL and dual-instrument perplexity report-only; single comparator
pin (build 9632, CPU); `llama-completion` mandated, `llama-cli` banned; bench-memory hygiene
binding. Two deliberate deviations from §6 defaults are flagged inside the protocol for
Tim's G0 sign-off (open item T3).

## 7. Size projection and the residency verdict

Derived table: `tensor_inventory.json` + markdown in the bundle. Projection of
resident-weight bytes (not a file-size promise; Phase 2 measures the real artifact):

| | bytes | note |
|---|---|---|
| Q8_0 today (all tensors) | 8,177,125,544 | |
| NVFP4 projected | **6,204,494,296** | ratio 0.759, shrink 1.318× |
| — of which kept embeddings | 3,707,764,736 | `per_layer_token_embd` (2.99 GB!) + tied `token_embd`, kept Q8_0 |
| — of which NVFP4 matmuls (294 tensors) | 2,219,212,800 | 1.889× smaller than their Q8_0 form (4.19 GB) |
| — of which kept F32/BF16 (gates, projs, norms) | 277,514,408 | incl. `inp_gate`/`proj` 220.2 MB, `per_layer_model_proj` BF16 55.1 MB |

**6 GB verdict: fully-resident REFUTED** — 5917.1 MiB of weights vs 6144 MiB VRAM leaves
~227 MiB, under any workspace/KV allowance (receipted with 256/512 MiB floors). The single
tensor that breaks the conductor's ~4.2 GB guess is `per_layer_token_embd.weight` (48% of the
projected total), which no production path repacks.

**What survives**: decode bandwidth (the campaign's actual lever — per-token weight reads are
dominated by the 294 matmuls, which shrink 1.889×), and a **partial-residency design**: a GPU
set of matmuls + gates + norms ≈ **2.44 GB**, with the two embedding tables file-backed/CPU
(row-gather per token; exactly the existing Metal-lane pattern where "embeddings stay
file-backed"). Fits 6144 MiB with multi-GB headroom. This is a Phase 4 design option, not a
promise; open item T6 asks Tim to confirm the campaign proceeds on these grounds.

Keep-list provenance caveat: the projection applies the Python-conversion keep rules
(embeddings/norms never repacked, only 2D-scaled weights convert); the pilot's **observed**
`llama-quantize` keep-list is captured at Phase 2 and G2 replaces this table with
measurement. Known repair to the lane output: the prose figure "423 F32 tensors
167,413,928 B" in the projection summary is a typo — the exact kept-F32 total is
222,464,168 B; the per-family table and grand totals were verified exact by integer
recomputation (they reconcile to the byte).

## 8. Upstream ecosystem scan

Web recon 2026-07-16, metadata/API-only (no weight downloads); full linked report preserved
in the scout lane transcript, distilled here:

- **Upstream timeline (llama.cpp)**: type added in PR #19769 (merged 2026-03-11,
  `GGML_TYPE_NVFP4=40` + `LLAMA_FTYPE_MOSTLY_NVFP4=39`, ModelOpt convert path; the author
  explicitly dropped quantization support); CUDA dp4a #20644 (03-26); generic MMQ #21074
  (04-01); **gemma4 NVFP4 tensor mapping #21971 (04-16)** and NVFP4 LM head #23046 (05-16) —
  all pre-pin; **Blackwell native MMQ #22196 = tag b8967, merged 2026-04-28** (conductor's
  date claim confirmed). **A named `llama-quantize` NVFP4 ftype target was never merged**
  (attempts #22858/#25446 closed unmerged; #22897/#25153 still open) — the pin's capability
  is type + the pre-existing `--tensor-type` override mechanism, exactly as §5 resolved.
- **#22042 (per-tensor scale correctness): open but dormant since 2026-05-12.** All four
  floated options avoid a GGUF disk-format change; nothing NVFP4-layout-touching merged after
  the pin. Watch item W1 stays, with low urgency. **However #24331 (2026-06-16), an NVFP4
  `llama-graph` edge-case fix, post-dates the pin's commit — the pin contains that bug**; new
  watch item W3.
- **Date footnote**: the upstream `b9632` tag points at a commit dated 2026-06-14, not
  2026-07-02 as the local pin doc states. The pin's identity is the SHA (`acd79d603`), which
  the empirical runs validate regardless; the date label should not be quoted.
- **E4B NVFP4 checkpoints**: no official NVIDIA/Google E4B NVFP4 exists (NVIDIA covers only
  31B and 26B-A4B). Community: `cosmicproc/gemma-4-E4B-it-NVFP4` is genuine **ModelOpt
  convention** (`weight_scale_2` + `input_scale` present — would convert to a
  **sidecar-bearing** GGUF: the exact file class D-B2 fails closed on);
  `unsloth/gemma-4-E4B-it-NVFP4` (2026-07-14) is **mixed compressed-tensors** (NVFP4 on MLPs
  only, FP8 attention, 8-bit packed embeddings, `weight_global_scale` naming) and likely
  doesn't convert with upstream tooling at all. Real-world NVFP4 arrives in ≥3 conventions —
  reinforcing pin-GGUF-only as the v1 wire target.
- **NVFP4 GGUFs in the wild**: ~100+ repos, all long-tail uploaders (zero from
  unsloth/bartowski/ggml-org; one from llama.cpp maintainer CISCai). Notably
  `FreedomAISVR/Gemma-4-E4B-it-NVFP4-GGUF` (5.19 GB, 2026-05-18) — an **optional Phase 2
  interop test case** (unknown provenance, no claims; its size vs our 6.2 GB projection
  suggests quantized embeddings). Smallest found: 0.58 GB (Qwen3.5-0.8B).
- **Published quality (reported as stated)**: PR #19769 measured Qwen3-4B NVFP4 at PPL +8.0%
  vs F16 (KLD 0.110) against **Q4_1**, not Q4_K_M; a calibrated custom NVFP4 of Qwen3.6-27B
  (discussion #23853) came out *behind* unsloth Q4_K_M (KLD 0.045 vs 0.022); third-party
  consensus: "quality vs Q4_K_M is unsettled", possibly worse for small models. The
  pre-registered protocol's "the answer is allowed to be it doesn't" posture is
  well-supported.

## 9. Decision drafts D-B1–D-B5 (recommendations for G0)

- **D-B1 — Wire-compat target: ACCEPT pin layout byte-for-byte.** Concretely: enum variant id
  40; 64-elem/36 B superblock; MXFP4-style nibble split; doubled-LUT × half-scale pair kept
  together as one convention. Rationale unchanged (interop receipts both directions, pin tools
  as golden quantizer — now empirically confirmed viable, zero format invention).
- **D-B2 — Per-tensor scale seam: REFRAMED by pin truth.** There is no in-block scalar to
  thread. Recommendation for v1: implement the in-block 4× UE4M3 sub-scales only, and **fail
  closed at admission on any NVFP4 GGUF carrying `.scale`/`.input_scale` sidecar tensors**
  (machine-readable refusal; silently ignoring them would be exactly the quiet math corruption
  §3.3 warns about — a sidecar-bearing ModelOpt file computes wrong logits without them).
  Sidecar *application* becomes a follow-on once a sidecar-bearing fixture exists. Phase 2
  asserts the pilot artifact is sidecar-free. The conductor's dropped/doubled-scale test
  requirement maps onto (a) the doubled-LUT/half-scale pair tests and (b) the
  sidecar-refusal admission test. Folding remains rejected — now doubly so, since the pin
  never folds either.
- **D-B3 — Runnable-lane scope: pilot-only until G3 passes, then lane-wide.** Unchanged.
  (Empirically `llama-quantize` can mint NVFP4 for other covered archs, so lane-wide
  admission after G3 is cheap; smoke stays gated on oracle-qualified combos per the existing
  guardrail.)
- **D-B4 — Phase 5 kernel route: moot; BLOCKED-HW.** No sm_120/sm_100 device (§3). Decide
  NVRTC-`sm_120a` vs precompiled PTX only if hardware materializes. No forward-looking perf
  statements.
- **D-B5 — Quantizer ownership: pin-tool-only for v1, CONFIRMED VIABLE.** The golden
  quantizer exists and ran deterministically; the canonical command shape is
  `llama-quantize --allow-requantize --tensor-type '<regex>=nvfp4'
  --override-kv general.file_type=int:39 <src> <dst> <base-ftype>` (§5). Native
  `camelid quantize --type nvfp4` (Phase 2b) stays optional and is deprioritized.

## 10. Phase-plan impact and watch items

- **Phase 1**: add to the test plan — the doubled-LUT/half-scale pair; UE4M3 NaN-sentinel and
  zero-byte semantics per the T5 decision; nibble-split order; exhaustive 256×16 decode table
  vs pin-generated golden vectors. Note: gguf-py's NVFP4 class is **decode-only** (no
  `quantize_blocks`), so golden vectors come from the pin's C reference via a small generator
  harness (checked into `tools/`/`scripts/` with provenance, per conductor).
- **Phase 2**: downloads (Q8_0 8.2 GB + BF16 15.05 GB + imatrix 4.4 MB; ~40 GB with produced
  rows, disk is fine); produce the **four** producible protocol rows (`NVFP4-mm`, `Q4K-mm`,
  `Q4_K_M-df`, `Q4_K_M-im`) from the Q8_0 baseline per protocol Amendment 2 (`NVFP4-all` is
  BLOCKED-HOST — whole-tensor f32 staging of `per_layer_token_embd` needs ~22.5 GB commit)
  with ×2 determinism sha checks; capture observed per-tensor logs (observed: the 84
  `inp_gate`/`proj` 2-D F32 tensors convert to Q8_0 in every row, identically across the
  `-mm` pair) vs §7; assert sidecar-absence; pin-load sanity via `llama-completion`;
  measured size table replaces §7. Verify pin loads `arch=gemma4` before anything else.
  Optional interop leg: the wild `FreedomAISVR` E4B NVFP4 GGUF (§8) as a
  load-or-refuse-cleanly test case, unknown-provenance, no claims.
- **Phase 3**: per `basalt_eval_protocol.md`; one small harness addition (forced-decode mode).
- **Phase 4**: reference implementation = the pin's dp4a MMVQ on this very box; residency
  design decides the file-backed-embeddings split (§7).
- **Phase 5**: BLOCKED-HW record into ledger + STATUS at Phase 6 unless T1 changes.
- **Phase 6**: CAIRN watch item W1 (below); ftype-26 receipt-label mapping; extend the
  contract tripwire test (`capabilities_support_statuses_stay_exact_row_allowlisted`) for any
  new row — the MUSTER lesson; drift-gate D only bites if the ledger row carries
  `identity.sha256` — include it.
- **W1 (CAIRN watch item)**: NVFP4 wire layout frozen at pin build 9632. Any future re-pin
  must re-verify `block_nvfp4` layout, UE4M3 semantics, and sidecar mechanism against §1
  before NVFP4 receipts transfer (#22042-adjacent proposals could change the format upstream).
- **W2**: pin build 9632 renamed CLI semantics — `llama-cli` is interactive-only; every
  scripted BASALT leg uses `llama-completion`/`llama-perplexity`/`llama-quantize` (binaries
  pinned by sha256 in the protocol §2).
- **W3 (new)**: upstream PR #24331 fixed an NVFP4 `llama-graph` edge case **after** the
  pin's commit — the pin contains that bug. If any pin-side NVFP4 leg misbehaves in
  Phase 2/3, check against #24331's symptom before suspecting Camelid or the artifact; the
  receipt documents whichever way it lands. (Oracle-side bug, not grounds for re-pinning
  mid-campaign.)

## 11. Open items for Tim (G0 review)

- **T1 (conductor §11.1)** Blackwell: none locally (§3). Confirm Phase 5 = BLOCKED-HW, or
  name borrowable sm_120/sm_100 silicon.
- **T2 (§11.2)** Pilot: confirmed as gemma-4-E4B-it (identities in §4). Confirm or rename.
- **T3 (§11.3)** §6 defaults: two amendments need sign-off — lane-native packs (9 prompts /
  320 tokens; protocol §3, errata E10) and the 80% sanity guard on the GO rule (protocol
  §5.2). The 2.0-point GO rule itself is unchanged.
- **T4 (§11.4)** D-B1–D-B5 as revised in §9 — accept or amend (D-B2's reframing is the
  substantive one).
- **T5 (new)** NaN/zero scale posture (errata E3): conductor mandates fail-closed; pin
  silently flushes NaN→0.0 and treats zero as legitimate. Recommendation: decode semantics
  match the pin bit-for-bit (parity math), **and** Camelid's admission layer scans NVFP4
  tensors and refuses files containing NaN scale sentinels (`0x7F`/`0xFF`); zero scale bytes
  admit (they are real all-zero blocks). This honors both the arbiter rule and the
  fail-closed posture without inventing decode semantics. **[G1 errata: the premise was
  half-wrong — pin CPU flushes only raw `0x7F`, while `0xFF` decodes to 240.0 and the pin's
  CUDA mirror flushes it (backend disagreement). The accepted posture is unchanged and
  strengthened; see §1 [G1 errata] and DECISIONS.md D17 addendum.]**
- **T6 (new)** Campaign grounds after E6/§7: fully-CUDA-resident on 6 GB is refuted for this
  pilot. Recommendation: proceed — the decode-bandwidth lever (1.889× on matmul reads) and
  the ~2.44 GB partial-residency option are the real payoff; Phase 4 measures, nothing is
  promised.
- **T7 (new)** Quality-comparison row design (protocol §1): the gated pair is
  **format-isolated** — `NVFP4-mm` vs `Q4K-mm`, identical files except the 294 matmul
  weights' format (uniform 4.5 bpw each, same keep-list by construction via the override
  regex). The practical rows (standard `Q4_K_M` data-free + imatrix) and the exploratory
  `NVFP4-all` (embeddings + tied head quantized; projected ~4.46 GB → puts full 6 GB
  residency back in play as a *measured* question) are report-only, never gated. This
  supersedes the conductor §6 comparator naming. Sign off or amend before Phase 2 produces
  the rows.

## 12. Artifact index (bundle `qa/evidence-bundles/basalt/phase0/`)

| artifact | content |
|---|---|
| `pin_extraction_receipts.md` | §1 facts with file:line + header excerpts |
| `camelid_touchpoints.md` | full Camelid-side seam map (enum, admission, runtime, receipts, ledger, CLI) |
| `hw_probe.json` | §3 |
| `tensor_inventory_raw.json` / `tensor_inventory.json` | 720-tensor inventory / keep-list + projection (with correction note) |
| `tools/gguf_header_inventory.mjs` | header parser used for the inventory (provenance comment inside) |
| `quantize_nvfp4.txt` | pin llama-quantize Q8_0→NVFP4 per-tensor log (Qwen3-0.6B) |
| `refusal_receipt.md` | baseline fail-closed captures + quantize provenance + discrepancy resolution |
| `incident-20260716-hard-hang.md` + `pin_sanity_excerpt.txt` | the hang incident, root cause, corrective rules |
| `manifest.json` + `SHA256SUMS` | bundle inventory per evidence-bundle conventions |

Gate G0 checklist (conductor §5): all five Phase 0 artifacts exist ✔; D-B1–D-B5 drafted with
recommendations ✔ (§9); pin verified NVFP4-capable ✔ (§1, §5). **STOP — human review.**

## 13. Invariant-lane matrix (Amendment 3 §2)

Added at stage S2 (the SHA_E engine-freeze commit). **Canonical form:**
`qa/invariant_lanes.json`, schema `qa/invariant_lanes.schema.json`
(`camelid.invariant-lanes/v1`, ledger-convention versioned tag + provenance
block). This section is the human-readable twin — if the two ever disagree, the
JSON wins and the disagreement is a bug. Enforcement (§2.4 mechanism, recorded
in DECISIONS.md D17 micro-decisions): `tests/invariant_matrix_binding.rs`
`include_str!`-binds every file an enforced cell names (rename/move breaks the
build), asserts every named test fn appears in its bound file's text (a fn
rename fails the meta-test, not the build — the §2.4-permitted substitution),
validates the JSON against the schema with a hand validator (serde_json, no new
deps), and trips the committed fixtures.

Cell legend: **E** = enforced (test fn named in the JSON cell), **na** =
structural reason (file-anchored, meta-test-guarded), **open:P4** = nothing
lands until Phase 4 (CUDA-resident NVFP4).

| lane \ invariant | I-unknown-type | I-nan-scale | I-sidecar | I-scale-once | I-k-div | I-carveout | I-cache-quant | I-plat |
|---|---|---|---|---|---|---|---|---|
| **L1 runnable** (admit/dequant/smoke) | E `rejects_unknown_quant_naming_tensor` + parse-trip fixture | E `nvfp4_nan_sentinel_refused_at_decode` (+ S1 fixture via scan-seam trip) | E `rejects_nvfp4_sidecar_scale_tensor` + NEW end-to-end admission fixture | E via sidecar refusal (v1 posture, see below) | E `decode_nvfp4_tensor_refuses_non_multiple_of_64_elements` + NEW parse-trip fixture | E `rejects_nvfp4_outside_pilot_arch` (+2 boundary companions + smoke gate) | na (no warm state on this lane — below) | E §9 cfg twins + NEW pilot-admit fixture twin |
| **L2 gemma4 wire** (the NVFP4-executing lane) | E `wire_quant_new_admits_nvfp4_and_still_refuses_uncovered` | E `wire_quant_new_refuses_nan_sentinel_scale_bytes` (fixture seam-tripped, S1 precedent) | E `sidecar_fixture_trips_d_b2_end_to_end` (END-TO-END on committed fixture) | E via sidecar refusal + bitwise matvec/decode anchors | E parse-boundary trip (in-lane re-check is defense-in-depth, see below) | na (no architecture axis to carve — below) | na (structurally cannot survive a swap — below) | E `windows_only_check_refuses_nvfp4_off_windows` (+ NaN-fixture load leg off-Windows) |
| **L3 CUDA resident** (NVFP4 residency = P4) | E `cuda_lane_check_refuses_nvfp4_before_the_from_wire_panic` (typed, pre-panic) | open:P4 | open:P4 | open:P4 (kernel-epilogue vs host row scale is the P4 D-B2 seam) | open:P4 | E `cuda_lane_check_admits_the_supported_projection_formats` | open:P4 | open:P4 |
| **L4 Metal** (§9.3 na, upgraded where S1 shipped refusals) | E `metal_lane_check_refuses_any_nvfp4_tensor` (UPGRADE over prescribed na) | na (NVFP4 never binds; refused at load) | E `sidecar_check_refuses_nvfp4_with_scale_tensors` (UPGRADE — D-B2 now runs in the Metal load path) | na | na | E `metal_lane_check_admits_files_without_nvfp4` | na | E (the lane refusal IS the platform posture; shared with I-unknown-type deliberately) |
| **L5 native quantizer** | na | na | na | na | na | na | na | na — all eight: Phase 2b never scheduled; D-B5 pin-tool-only |

**Counts: 19 enforced / 15 na / 6 open (all open:P4). 40/40 cells filled — no
empty cells, meta-test-asserted.**

### 13.1 Fixtures (§2.6)

S1 pair unchanged (byte-identical through the generator refactor; shas
re-verified): `nvfp4_sidecar_trip.gguf`, `nvfp4_nan_sentinel_trip.gguf`. S2
quartet added by `scripts/basalt-nvfp4-golden/gen_sidecar_fixture.mjs`
(deterministic, <4 KB, shas pinned in `tests/invariant_matrix_binding.rs` and
`tests/fixtures/gguf/SHA256SUMS`):

- `nvfp4_unknown_type_trip.gguf` — GGML type id 99 → parse refusal ("unknown or
  removed GGML type"), the file boundary shared by every lane. (Was id 41 until 2026-08-01.
  Picking 41 was a mis-specification on the day this fixture was authored: the pin
  receipt in this same bundle (`pin_extraction_receipts.md` §1) already listed 41 as
  allocated. The fixture kept tripping the refusal only because Camelid had not
  implemented 41; implementing it (the 1-bit Q1_0 format) exposed that. 99 is far
  outside the range any real file uses, so the next type to be implemented cannot
  quietly defuse it again.)
- `nvfp4_k_div_trip.gguf` — NVFP4 first dim 48 → parse refusal ("not divisible
  by block size 64"), never a silent pad.
- `nvfp4_sidecar_admit_trip.gguf` — sidecar trio + `tokenizer.ggml.model`, so
  runnable admission reaches the quant axis: D-B2 trips end-to-end from file
  bytes on every platform (sidecar check precedes the §9 gate).
- `nvfp4_pilot_admit.gguf` — BF16-free pilot shape: ADMITS on Windows (first
  file-boundary positive control for the D-B3 carve-out) and trips the §9 TK2
  refusal on the ubuntu/macos CI legs.

Cells whose full-file trip is unreachable carry `fixture:"seam:<reason>"` (the
S1 NaN precedent): L2 I-nan-scale (config parsing precedes the WireQuant scan —
asserted, not assumed), L3/L4 refusal helpers (real-model + hardware/cfg-bound
load paths; helpers are cfg-independent and unit-driven on every host), and the
two I-scale-once cells (v1 posture: no out-of-block scale exists on the wire to
apply twice — the sidecar refusal is the enforcement; in-block sub-scale
single-application is bitwise-pinned by the golden decode/matvec suites).

### 13.2 I-cache-quant verdict (investigated honestly, PR #419 lesson)

**na everywhere NVFP4 can execute — structurally, with meta-test-guarded
anchors; no cell was faked and no P3-FINDING was needed.** Facts (file-anchored
in the JSON): the only cross-request decode cache in the serve stack is the
Llama-path single-slot prompt-prefix cache (`src/api/mod.rs
lookup/store/clear_prompt_prefix_cache`). NVFP4 cannot reach it: NVFP4 files
are gemma4-only (D-B3), gemma4 chat resolves to a per-model-id
`Gemma4ServeRuntime` (replaced wholesale on reload, cleared on unload) or fails
typed (`model_not_ready`) — never a silent Llama-path fallback — and the cache
payload is a `LlamaInferenceSession` with zero call sites in
`gemma4_runtime.rs` (asserted literally by the meta-test). The cache key
additionally carries `model_id` AND `model_path` (distinct quant rows are
distinct files, G2 §1), and the slot is cleared on every active-model switch;
key-miss on model change is pinned by
`prompt_prefix_cache_reuses_exact_prompt_and_invalidates_key_changes`. The
runnable lane serves only qwen35/gemma3 (`is_runnable_serve_arch`) with a
`&self`-immutable runtime — no warm state exists. L3's cell stays open:P4: the
future NVFP4-resident path must prove its own keying/reset story rather than
inherit this note.

Out-of-matrix observation (Llama lane, not an NVFP4 surface, reported for
completeness): `AppState::cached_weights` is keyed by model id alone (path not
in the key, `src/api/mod.rs load_weights_lru`); entries are removed on unload,
so staleness would require re-registering the same id against different bytes
without an unload. Flagged to Tim as an observation, not a matrix cell.

### 13.3 Source reconciliations (extracted, not invented)

- **Zero scales:** conductor §3.2/§8 say "NaN **or zero** scale byte → load
  error"; the signed T5 posture (D17) is narrower — sentinel bytes 0x7F/0xFF
  refuse, **zero scales admit** (and the G1 addendum strengthened the sentinel
  story: the pin's own backends disagree on 0xFF). The matrix records the
  signed T5/D17 semantics; the conductor text predates the sign-off and was
  deliberately left unedited (campaign docs are append-only history).
- **K%16 vs K%64:** conductor §8 states K%16 (the sub-block) as quantizer
  keep-list guidance; the consume-side wire unit is the 64-element superblock,
  and every consume-side check enforces %64 (parse `tensor_nbytes`, decoder,
  WireQuant). The column is defined at %64 per the amendment; noted here so
  nobody reads §8 as a consume-side 16.
- **Defense-in-depth honesty (I-k-div L2):** `WireQuant::new`'s block-alignment
  and byte-size re-checks are unreachable for NVFP4 through GGUF parse (parse
  guarantees first-dim divisibility) — the cell therefore cites the parse-trip
  as its enforcement and names the in-lane re-check as backstop, instead of
  pretending a lane-native trip exists.
- **L4 upgrades:** §9.3 prescribed na for Metal; S1 shipped typed refusals
  (`nvfp4_metal_lane_check`, plus routing `nvfp4_sidecar_check` through the
  Metal load path, which previously never ran it). Stronger-than-prescribed is
  recorded as **enforced with the upgrade noted** — honest in both directions.
- **Shared test across two cells (L4):** `metal_lane_check_refuses_any_nvfp4_tensor`
  backs both I-unknown-type and I-plat on L4 — the single refusal genuinely
  enforces both postures there; recorded in both cells deliberately rather than
  manufacturing a second test.

### 13.4 Ratchet (§2.5 — rules live as data in the JSON `ratchet` block)

- **R1** — lane-adding PRs add a ROW and fill all 8 cells in the same PR (the
  meta-test's full-population assertion makes a missing cell a failure).
- **R2** — invariant-signing PRs (new D17 addendum / new refusal-bearing sharp
  edge) add a COLUMN with its source cited, filled for all lanes.
- **R3** — Gate G4 closes every open:P4 cell (enforced with lane-native tests,
  or na with structural reasons). Teeth: the meta-test fails the moment the
  Phase-4 CUDA-lane refusal text disappears while any open:P4 cell remains.
- **R4** — scheduling Phase 2b re-opens the L5 row, same mechanics (proxy: the
  encoder's TEST-ANCHORING-ONLY marker).
