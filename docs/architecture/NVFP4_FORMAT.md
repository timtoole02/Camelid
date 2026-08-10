# Camelid NVFP4 weight format — spec + pilot findings (BASALT)

> [!NOTE]
> This is a **format spec, NOT a support claim**; adding a format here never implies support,
> parity promotion, or a catalog claim. For current support truth and release status, use
> [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and [`STATUS.md`](../reference/STATUS.md). NVFP4 is
> shipped as engine facts for the `gemma-4-E4B-it` pilot only and is **not** a supported or
> certified row; see the [Support status](#support-status) section for the honest boundary.

## Why NVFP4, and the pilot scope

NVFP4 is llama.cpp's E2M1 4-bit weight format with per-16-element UE4M3 sub-block scales
(`GGML_TYPE_NVFP4` = 40, added upstream 2026-03). BASALT (the campaign that landed this)
targeted **decode bandwidth** on a 6 GB card: the pilot's 294 matmul weights shrink 1.889×
vs Q8_0, moving fewer bytes per token. The vehicle is `gemma-4-E4B-it` and nothing else —
the runnable-lane admission is a pilot-model-only carve-out (D-B3), and the format admits on
Windows and macOS in this release (GABBRO M2). On macOS both a CPU wire lane and, as of GABBRO
M3 + M3-followup, a Metal GPU resident lane run NVFP4; the Metal lane is opt-in via the
macOS-only `gemma4-generate-gpu` subcommand (kernel `nvfp4_block_linear_row_ksplit_f32y_wire`),
self-parity-proven vs the CPU oracle — run end-to-end on the byte-exact artifact,
isolated 128-tok decode 12.12 tok/s. Runnable-lane admission covers the NVFP4 format and — as of
D-B6 (2026-07-17) — BF16 as a covered exact-decode type, so its only produced file (gemma4-E4B)
admits fully; its one BF16 tensor (`per_layer_model_proj`) was the prior admission blocker
(G2 §6b). It executes via `gemma4_runtime` — the CPU wire + CUDA-resident lanes plus an opt-in macOS
Metal resident lane (via `gemma4-generate-gpu`), not the generic
runnable serve bridge. "Runnable" here is format admission, not a runnable serve path. The 6 GB *full-residency* motivation was refuted at recon (the
Q8_0 embeddings are never repacked, ~3.7 GB of them), so the campaign proceeds on
decode-bandwidth + partial-residency grounds, not a fully-resident promise.

## Format specification (ggml `block_nvfp4`, exact)

This is the single source of truth for every phase (CPU decode, CPU wire dot, CUDA GEMV). It
is the pin's (`llama.cpp acd79d603`, build 9632) layout adopted byte-for-byte (D-B1); Camelid
invents no private layout.

```
QK_NVFP4     = 64        // elements per super-block
QK_NVFP4_SUB = 16        // elements per sub-block; 4 sub-blocks per super-block
struct block_nvfp4 {     // 36 bytes total, 4.5 bpw
    uint8_t d[4];        // [0..4)   four UE4M3 sub-block scales (d first)
    uint8_t qs[32];      // [4..36)  packed 4-bit E2M1 nibbles
}
static_assert(sizeof(block_nvfp4) == 36)
```

**E2M1 element LUT (`kvalues_mxfp4`, doubled magnitudes):**

```
kvalues_mxfp4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
```

These are **2× the true E2M1 magnitudes** (true = 0, .5, 1, 1.5, 2, 3, 4, 6); nibble bit 3 is
the sign, bits 2-0 index the magnitude. The compensating 0.5 lives inside the UE4M3 scale
decode, so the product `kvalues_mxfp4[nibble] * ue4m3_to_fp32(d[s])` is correct. This is the
**doubled-LUT × half-scale pair rule**: the doubled LUT and the half-scale convention are one
convention and must be kept together (12 × 224 = 2688 = the true 6 × 448 max — the factor of
two lives across the pair).

**Nibble packing (MXFP4-style half/half split, not adjacent pairing).** Sub-block `s` (0..3)
occupies bytes `qs[s*8 .. s*8+7]`. For byte `qs[s*8 + j]` (`j` in 0..7): the **low** nibble is
sub-block-local element `j` (0..7), the **high** nibble is element `8+j` (8..15). Decode:
`y[j] = kvalues_mxfp4[qs & 0x0F] * d`, `y[j+8] = kvalues_mxfp4[qs >> 4] * d`.

**Scale format — UE4M3 (unsigned E4M3).** The scale byte's top bit is stripped (`& 0x7F`);
bias 7, 3-bit mantissa, subnormals via `man * 2^-9`. The `0.5` correction factor is folded
into the decode (`raw * 0.5f`) to match the doubled LUT. The quantizer (pin C reference) feeds
`amax(sub-block) / 6.0` as the scale input, clamps input to 448.0, rounds half-up, and encodes
elements by exhaustive nearest-LUT search (`best_index_mxfp4`, first-wins ties — not IEEE
round-nearest-even).

## Pin-layout findings (fixture-arbitrated decode facts)

The normative decode facts, arbitrated by pin-generated golden vectors (Phase 1) after two
Phase 0 prose claims were found wrong — cite `BASALT_RECON.md` §1 [G1 errata] and
`qa/evidence-bundles/basalt/phase0/pin_extraction_receipts.md`:

- **UE4M3 sentinel decode.** Raw byte `0x00` → 0.0; raw byte `0x7F` → 0.0 (the pin CPU treats
  it as a NaN sentinel and flushes it). **`0xFF` decodes through exp/man to 240.0 on the pin
  CPU**, while the pin's CUDA mirror flushes `0xFF` to 0.0 — i.e. **the pin's own CPU and CUDA
  backends disagree on `0xFF`-scaled blocks.** Camelid decode is **pin-CPU-bitwise**
  (`0xFF` → 240.0), and admission refuses files containing either sentinel byte (see below).
- **`decode(0x7E)` = 224.0** (a Phase 0 aside said 112.0; corrected by the fixtures). The
  encoder saturates every input ≥ 248 to `0x7E`, so bytes `0x78..0x7D` are decodable but
  encoder-unreachable.
- **No in-block per-tensor scale.** The only in-block factors are the four UE4M3 sub-scales.
  The per-tensor `weight_scale_2` mechanism is an **optional sidecar F32 tensor** (`.scale` /
  `.input_scale`), created only by the Python ModelOpt/compressed-tensors convert path and
  applied post-matmul via a `ggml_mul` node; `llama-quantize`-produced NVFP4 carries none.
- **K % 64.** The wire unit is the 64-element super-block; every consume-side check enforces
  `ncols % 64 == 0` (parse, decoder, wire quant); a non-multiple-of-64 first dim is a typed
  parse refusal, never a silent pad.
- **Quantizer provenance.** NVFP4 is not a `llama-quantize` positional ftype target upstream;
  the pilot rows are minted via the per-tensor override
  (`llama-quantize --allow-requantize --tensor-type '<regex>=nvfp4'
  --override-kv general.file_type=int:39 <src> <dst> <base-ftype>`), empirically deterministic
  (byte-identical across repeat runs). `general.file_type` carries `LLAMA_FTYPE_MOSTLY_NVFP4`
  = **39** (not the ggml ftype 26); the enum's Debug name is the receipt-visible quant label.

## Signed refusal postures (fail-closed)

- **D-B2 — sidecar fail-closed.** v1 implements the four in-block UE4M3 sub-scales only and
  **fails closed at admission on any NVFP4 GGUF carrying `.scale` / `.input_scale` sidecar
  tensors** (silently ignoring a ModelOpt file's per-tensor scale2 would compute wrong logits).
  Enforced in both the runnable admission lane and the gemma4 wire lane. Sidecar *application*
  is a follow-on once a sidecar-bearing fixture exists.
- **T5 — NaN-sentinel refusal.** Decode semantics match the pin bit-for-bit (sentinels flush
  to 0.0), **and** Camelid admission scans NVFP4 tensors and refuses files containing `0x7F` or
  `0xFF` scale sentinel bytes (such files cannot even produce a well-defined cross-backend
  oracle). Zero scale bytes admit (they are legitimate all-zero blocks).
- **§9 — Windows/macOS-only (GABBRO M2).** NVFP4 admission is allowed on Windows and macOS and
  refuses on every other target via a runtime platform gate (`!cfg!(windows) && !cfg!(macos)`
  inside ordinary code, both lanes) with the typed error "NVFP4 is Windows/macOS-only in this
  release; see SUPPORT_MATRIX". cfg-twinned tests pin every side on the CI legs that run there
  (Windows and macOS assert admission; other targets assert the refusal). macOS runs the CPU
  wire lane (bit-exact, Gate G-M1; used by `serve`); its Metal GPU resident kernel
  (`nvfp4_block_linear_row_ksplit_f32y_wire`) is now wired via GABBRO M3 + M3-followup and
  reachable opt-in through the macOS-only `gemma4-generate-gpu` subcommand, self-parity-proven
  vs the CPU oracle (run end-to-end on the byte-exact artifact; isolated 128-tok decode 12.12 tok/s).

## Quality / performance receipts

Quality (Gate G3) and CUDA decode perf (Gate G4 + Phase 4b dp4a) are receipted in
`qa/evidence-bundles/basalt/phase3/BASALT_G3_SUMMARY.md` (the G3 NO-GO quality table) and
`qa/evidence-bundles/basalt/phase4/cert/BASALT_G4_SUMMARY.md` (CERT + dp4a perf). Phase 5
(Blackwell sm_120 tensor-core MMA) is **BLOCKED-HW** — no Blackwell silicon on the target.

## Support status

NVFP4 4-bit weights, gemma-4-E4B pilot, Windows + macOS. On macOS: a CPU wire lane (bit-exact,
used by `serve`) and an opt-in Metal GPU resident lane via the macOS-only `gemma4-generate-gpu`
subcommand (wired via GABBRO M3 + M3-followup, self-parity-proven vs the CPU oracle; not yet
exercised end-to-end on the byte-exact artifact; isolated 128-tok decode 12.12 tok/s). Engine facts: bit-exact CPU decode, validated on
x86 and on Apple Silicon/ARM (GABBRO Gate G-M1), + a Windows CUDA dp4a GEMV kernel (46/46
bit-identical). Measured vs the Q8_0 parent at matched 4.5 bpw:
behind Q4_K on quality (G3 NO-GO, 88.5% vs 92.6% top-1 agreement; 0.111 vs 0.065 mean-KL
nats), but 1.03x faster than Q8_0 CUDA decode (26.51 vs 25.80 tok/s) and 2.08 GB lighter VRAM
on an RTX 3060 Laptop (decode-only, this box). The macOS row is now a supported_exact_row_smoke row (a current-engine near-tie vs Q4_K; the frozen G3 NO-GO stands as history); NOT quality-competitive beyond the 2pp GO tolerance — a space/speed quant.
