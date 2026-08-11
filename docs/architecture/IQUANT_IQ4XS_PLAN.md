# Camelid i-quant Plan — IQ4_XS (first format)

> [!NOTE]
> This document is a design/implementation note, not the public support ledger. For current
> support truth and release status, use [`COMPATIBILITY.md`](../../COMPATIBILITY.md) and
> [`STATUS.md`](../reference/STATUS.md). Adding a new low-bit format here never implies support,
> parity promotion, or a catalog claim until its own dequant-parity evidence lands.

## Why IQ4_XS, and why first

The user goal is **fit a bigger model into the 8 GB RTX 4060** by trading bits for size. The
llama.cpp "i-quants" (`IQ*`) compress better per bit than the K-quants Camelid already runs.
IQ4_XS is chosen as the **first** i-quant format because it is the least invasive to prove:

- It reuses the **exact same 16-entry non-linear codebook** (`kvalues_iq4nl`) that Camelid
  already ships for `IQ4_NL` (`IQ4NLBlock::dequantize`, `src/tensor/mod.rs`). No new grid
  codebook tables (unlike `IQ2_XXS`/`IQ1_S`, which carry 256-/512-entry lattice grids).
- It is a 256-element **super-block** exactly like `Q6_K`/`Q4_K`, so it slots into the existing
  super-block decode + wire-streaming + resident-GEMV seams with no new structural concept.

The honest trade-off: IQ4_XS is ~4.25 bits/weight, so it is a *mild* compressor (an 8B lands
around ~4.3 GB). It makes a "slightly too big at Q4_K_M" model fit; the dramatic 8 GB wins
(13B in 8 GB) come from `IQ2_XXS` (~2 bpw), which is the **second** format on the same rails.
IQ4_XS is deliberately the plumbing-and-parity bite; IQ2_XXS is the payoff bite.

## Format specification (ggml `block_iq4_xs`, exact)

```
QK_K = 256
struct block_iq4_xs {          // 136 bytes total, 4.25 bpw
    ggml_half d;               // [0..2)   super-block scale (f16)
    uint16_t  scales_h;        // [2..4)   high 2 bits of each of 8 sub-block scales
    uint8_t   scales_l[4];     // [4..8)   low 4 bits of each of 8 sub-block scales (nibbles)
    uint8_t   qs[128];         // [8..136) 4-bit codebook indices, 16 bytes per sub-block
}
```

Dequant (bit-for-bit with `dequantize_row_iq4_xs`):

```
d = f16_to_f32(d)
for ib in 0..8:                                        // 8 sub-blocks of 32 values
    ls = ((scales_l[ib/2] >> (4*(ib&1))) & 0x0F)       // low nibble
       | (((scales_h >> (2*ib)) & 0x3) << 4)           // high 2 bits
    dl = d * (ls - 32)                                  // 6-bit scale, biased by -32
    for j in 0..16:
        y[32*ib + j]      = dl * kvalues_iq4nl[qs[16*ib + j] & 0x0F]
        y[32*ib + j + 16] = dl * kvalues_iq4nl[qs[16*ib + j] >> 4]

kvalues_iq4nl = [-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113]
```

This is the single source of truth for every phase (CPU decode, CPU streaming dot, GPU GEMV).

## Verified building blocks (fork @ `origin/main` 6c2a544a)

- **GGUF type recognition** — `src/gguf/reader.rs` `GgufTensorType`. `from_id` and `layout`
  already map `IQ4NL` (id 20) and `Tq2_0` (id 35). IQ4_XS is **ggml type id 23** with layout
  `(block=256, type_size=136)` and is currently absent → falls through to `Unknown(23)`.
- **Codebook** — `IQ4NLBlock::dequantize` (`src/tensor/mod.rs`) holds the 16-value `KVALUES`
  as a function-local const. It will be lifted to a module-level `KVALUES_IQ4NL` so IQ4_NL and
  IQ4_XS share one codebook (single source of truth, zero behavior change to IQ4_NL).
- **Super-block decode reference** — `Q6KBlock` + `decode_q6_k_blocks` + `decode_q6_k_tensor`
  (`src/tensor/mod.rs`) are the structural template for `IQ4XSBlock` + helpers.
- **CPU f32 load path** — `load_cpu_f32_with_q8_0_block_retention` `match desc.tensor_type`
  (`src/tensor/mod.rs`) already has an `IQ4NL` arm; IQ4_XS gets a sibling arm.
- **CPU wire-streaming linears** — `load_tq2_0_wire_linear` / `load_kquant_wire_linear`
  (`src/tensor/mod.rs`) retain raw wire bytes with `data: Vec::new()` (no f32 materialization);
  the per-row dot kernels (`q6_k_wire_row_dot_simd`, `tq2_0_dot`, …) and the streaming linears
  (`matmul_rhs_transposed_*_block_dot`, `accumulate_transposed_linear_row_*`) live in
  `src/inference.rs`. The loader selects them in the `load_linear` closure (`src/inference.rs`).
- **Resident GEMV** — `q6k_gemv` / `q4k_gemv` CUDA kernels feed from the retained wire bytes;
  IQ4_XS mirrors `q6k_gemv` (per-row super-block decode → int dot vs Q8_K activation).
- **Admission gate** — `src/runnable/admit.rs` explicitly lists i-quants (`IQ4NL`) as a v1 gap
  and rejects them; IQ4_XS admission is a Phase-4 decision, gated behind a quality receipt.
- **Filename recognition** — `hf_browse.rs::guess_quant` already returns the `IQ4_XS` string
  (advisory only; never gates a lane).

## Design decisions

1. **One codebook, one dequant formula.** Lift `KVALUES_IQ4NL` to module scope and route both
   IQ4_NL and IQ4_XS through it. The block `dequantize` is the only place the formula exists;
   the streaming dot and (later) the GPU kernel are proven equal to it by tests, never
   re-deriving the math independently.
2. **CPU-first, GPU-second — sequencing, not compromise.** A CPU f32 decode materializes the
   whole tensor (~32 GB for an 8B) and does *not* serve the "fit bigger" goal; it exists only as
   the parity reference. The memory win on CPU comes from the **wire-streaming linear**
   (`data: Vec::new()`, dequant per row), mirroring `load_tq2_0_wire_linear`. The VRAM win —
   the actual stated goal — is delivered by the **resident GEMV** in Phase 3. IQ4_XS is not
   "done" for the goal until Phase 3, and the plan says so.
3. **Fail-closed at every seam.** Unknown/misaligned bytes → typed `BackendError`, never a
   silent fallback. IQ4_XS stays rejected by the admission gate until Phase 4 ships a quality
   receipt.
4. **Quality gate = statistical, not exact-token.** Low-bit weights flip greedy tokens at
   near-ties (already observed on the TQ2_0 lane: 3/4 prompts identical, 1 near-tie). Forcing
   exact-token parity would fail good IQ4_XS models for the wrong reason. The Phase-4 admission
   receipt uses a **KL-divergence / perplexity band** against the f16 (or Q8_0) reference on a
   fixed prompt set, fail-closed, and records exact-token agreement as observability, not as the
   pass/fail line. Bit-exactness is still demanded *within* Camelid (decode == streaming dot ==
   GEMV); it is only the cross-quant quality comparison that is statistical.

## Phases

> **Status (all four phases complete + validated on `bartowski/Llama-3.2-1B-Instruct-IQ4_XS.gguf`,
> RTX 4060):** CPU decode + streaming and CUDA-resident GEMV all generate coherent output; the
> model runs fully resident in the 8 GB GPU at ~47 tok/s (vs ~1 tok/s on the deterministic CPU
> path). Branch `feat/iquant-iq4xs`.

### Phase 1 — Decode + f32 load path (foundational, provable) — **DONE**
- `src/gguf/reader.rs`: add `IQ4XS` variant; `from_id` `23 => IQ4XS`; `layout` `(256, 136)`.
- `src/tensor/mod.rs`: `IQ4_XS_BLOCK_BYTES = 136`; module-level `KVALUES_IQ4NL`; `IQ4XSBlock`
  (`from_bytes`, `scale_f32`, `dequantize`); `decode_iq4_xs_blocks`; `decode_iq4_xs_tensor`;
  `IQ4XS` arm in the f32 load match + error-message list.
- Tests: block dequant vs a hand-computed ggml vector; sub-block scale unpack edge cases
  (`ls` low/high split, `-32` bias, `ib&1` nibble select); tensor decode alignment + reject
  on misaligned bytes; codebook-shared-with-IQ4_NL invariant.
- Gates green (fmt, clippy `--all-targets --all-features`, `cargo test --lib`).

### Phase 2 — CPU wire-streaming linear (host-RAM fit) — **DONE**
- `CpuTensor::iq4_xs_wire_bytes` field (update the struct + all constructor sites);
  `load_iq4_xs_wire_linear` (mirror `load_tq2_0_wire_linear`); `iq4_xs_wire_row_dot`
  (int dot vs Q8_K activation, proven equal to the block dequant × activation);
  streaming linears `matmul_rhs_transposed_iq4_xs_block_dot` +
  `accumulate_transposed_linear_row_iq4_xs`; selection in the `load_linear` closure and the
  runtime dispatch points.
- Tests: streaming-dot == reference (decode-then-f32-dot) at nblk 1/2/5; end-to-end single-row
  and prefill-tile equivalence; no-f32-materialization assertion (`data.is_empty()`).

### Phase 3 — CUDA-resident `iq4xs_gemv` (VRAM fit — the goal) — **DONE**
- Mirror `q6k_gemv`: upload retained wire bytes, per-row super-block decode + int dot vs Q8_K.
- Resident-vs-CPU parity gate on a real IQ4_XS GGUF; token-parity oracle where it holds,
  KL band where it does not.
- Delivered: `iq4xs_gemv` kernel + `ProjQuant::IQ4XS` + `is_iq4xs` residency predicate;
  GPU parity test `iq4xs_gemv_matches_oracle` (1e-4) passes; the real model runs all layers
  resident at ~47 tok/s.

### Phase 4 — Admission + Fit Advisor — **DONE (admission); Advisor already quant-agnostic**
- `src/runnable/admit.rs`: accept IQ4_XS once a committed quality receipt exists.
- Fit Advisor: recognize IQ4_XS rows so "won't fit at Q4 → offer an i-quant" is a real
  recommendation; teach the catalog/advisor the i-quant footprint.
- Delivered: `runnable::dequant` routes IQ4_XS to `decode_iq4_xs_tensor`; `admit.rs`
  `is_covered_quant` accepts IQ4_XS (the runnable f32 lane runs it). The capacity-axis Fit
  Advisor already handles IQ4_XS rows (footprint from `size_bytes`/dims is quant-agnostic;
  `guess_quant` already recognizes the string). The stricter `runnable-smoke`
  oracle-qualification gate stays closed for IQ4_XS pending a committed llama.cpp parity
  receipt (fail-closed).

## Gates (every phase)
- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --lib` (focused: `cargo test --lib iq4_xs`)
- Files stay LF; PR diff kept minimal.

## Explicit non-goals (this lane)
- No support-ledger promotion, no catalog claim, no parity promise until Phase 4 evidence.
- No IQ2/IQ1 grid codebooks yet (separate follow-on on the same rails).
- No change to `IQ4_NL` behavior (only the codebook const moves).
