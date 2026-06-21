# LLAMA_CPP_ARCHAEOLOGY.md

What machinery makes llama.cpp fast in the lanes where Camelid trails, mapped to file paths, with a Rust-native verdict for each. llama.cpp @ `acd79d6`. Build flavor on this host (from `build/CMakeCache.txt`): `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, `GGML_CUDA_FA=ON`, `GGML_CUDA_GRAPHS=ON`, `GGML_NATIVE=ON`. CPU `system_info`: `AVX2=1 | AVX512=1 | F16C=1 | FMA=1 | LLAMAFILE=1 | OPENMP=1 | REPACK=1`.

This is concept study only. Hard rule honored: no llama.cpp/ggml code is linked, copied, or made a runtime dependency. Camelid stays Rust-native.

---

## 1. Centralized quantized matmul + chunk scheduler  ⭐ explains the CPU gap

- **File (verified):** `ggml/src/ggml-cpu/ggml-cpu.c` — chunk scheduler `atomic_fetch_add_explicit(&threadpool->current_chunk, 1)` at **1350-1352** + geometry adapt **1415-1441**; single activation quantize into `wdata` at **1313-1348**; `ggml_quantize_row_q8_0` at `ggml-quants.c:238-261`.
- **What it does:** ONE backend-owned GEMM for every projection (Q/K/V/O, gate/up/down, logits). It (a) converts the activation to the kernel's `vec_dot_type` **once** into shared `wdata` (block-parallel over K, QK8_0=32-blocks), (b) splits the output into 16×16 chunks handed to threads via an atomic `current_chunk` work-stealing counter (one atomic per chunk — no false sharing), (c) adapts chunk geometry when `nchunk0*nchunk1 < nth*4` (avoids idle threads on small/odd shapes), (d) optionally calls `llamafile_sgemm()` (tinyBLAS) before the generic loop.
- **Why it matters for speed:** scheduling and activation-quantization live *inside* the kernel owner, so there is one quantize + one tiled traversal per matmul regardless of role. Camelid's measured CPU gap is a near-uniform ~1.6× (prefill 0.62×, decode 0.66×) — the signature of a per-kernel-throughput/scheduling deficit, not one bad op.
- **Camelid equivalent:** `src/inference.rs::matmul_*_with_precision_with_plan()` + per-role bespoke paths (`try_x86_q8_ffn_down_decode_consumer_path`, `try_x86_q8_attention_qkv_decode_consumer_path`, …). Generic f32 uses Rayon in `CpuTensor::matmul`; Q8 uses bespoke loops. **No single Q8 GEMM owner** — role paths re-quantize the input and re-derive shapes independently (see `qa/evidence-bundles/x86-q8-llamacpp-mapping-*`).
- **Rust-native version:** one `matmul_q8_0_runtime_packed_x86(input_rows, packed_weight, out_shape)` owner that all FFN/attention/output projections call, doing a single activation quantize + one Rayon chunk loop with a tinyBLAS-style register tile (e.g. 4×N rows × K-block). The repo's own mapping note proposes exactly this (`CAMELID_X86_Q8_MATMUL_OWNER=ffn_down` as the first slice).
- **Worth implementing now?** This is the real CPU lever, but it is a multi-week architectural refactor (unify ~6 role paths under one tiled GEMM), not a small patch. **High value, high effort. Not a P0 small fix.**

## 2. tinyBLAS (llamafile sgemm) + runtime REPACK  ⭐

- **File (verified):** `ggml/src/ggml-cpu/llamafile/sgemm.cpp` — `tinyBLAS_Q0_AVX` register tiles via recursive `mnpack()`, `gemm<RM,RN>` keeps `C[RN][RM]` in registers and loops K **once per tile** (**1353-1520**); `gemm4xN` packs 4 FP16 scales into 64-bit → `_mm_cvtph_ps` (**1635-1682**); entry `llamafile_sgemm` **3928-3938**. Shape-specialized dispatch: **`gemm1xN` for 1×N prefill, `gemmMx4` for decode**. Q8_0 vec-dot `arch/x86/quants.c::ggml_vec_dot_q8_0_q8_0()` **1170-1204** (8 parallel `__m256` accumulators). `block_q8_0` = 2B fp16 + 32×i8 at `ggml-common.h:241-246`.
- **What it does:** tile-aware micro-kernels (register-blocked M×N×K) for quantized GEMM, plus a one-time weight **repack** into an interleaved layout that streams contiguously and feeds wide FMA/VNNI lanes with minimal shuffles. The K-loop runs **once per output tile** (not once per output column), so each repacked weight block is loaded to cache once and reused across the whole RM×RN tile. On this host it runs **AVX-512 + FMA**.
- **Why it matters:** prefill is compute-bound; a tiled GEMM that keeps the output tile in registers and reuses each repacked weight across the token chunk is what makes llama.cpp's per-kernel throughput higher. On TRUE CPU (CUDA hidden) llama 3B prefill=30.9 vs decode=8.8 t/s (~3.5× amortization) and Camelid prefill=18.9 vs decode=6.0 (~3.2×) — **both amortize similarly, so the gap is raw kernel throughput (0.61×), not batching.** It shows up identically on prefill and decode because the same tiled-vs-bespoke GEMM quality difference applies to both. (NB: llama-bench `-ngl 0` pp512=740 is GPU-offloaded, not CPU — see PERF_GAP_REPORT methodology.)
- **Camelid equivalent:** `src/tensor/mod.rs` `q8_0_runtime_packed_rows4_*` (SoA i8 repack — present and shipped) consumed by bespoke per-role loops. The repack idea is already Rust-native; the **tiled GEMM consumer is what's missing**.
- **Rust-native version:** a register-blocked Q8×Q8→f32 kernel over the existing rows4 repack, with `core::arch` AVX2 FMA (and an AVX-512 variant gated for compute-bound prefill only — see §9). Accumulate int8→i32 per block (associative ⇒ bit-exact), apply the f16 block-scale product in fixed order.
- **Worth implementing now?** The kernel itself is a focused, parity-safe slice (int accumulation is bit-exact). **P1**: medium effort, the highest-confidence CPU throughput lever. Must beat the current packed path on THIS host (measured: the current gated packed kernels *regress* here — see CAMELID_HOTSPOTS).

## 3. CUDA backend: MMQ + Flash-Attention + CUDA graphs

- **File:** `ggml/src/ggml-cuda/` — `mmq.cu` (quantized int8 tensor-core matmul), `fattn*.cu` (fused flash attention), graph capture gated by `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA=ON`.
- **What it does:** MMQ does Q8×Q8 matmul with `dp4a`/IMMA int8 paths directly on quantized weights (no dequant-to-f16 first). Flash-attention fuses score→softmax→weighted-V in one kernel that stays ~flat with context depth. CUDA graphs collapse per-token launch overhead.
- **Why it matters:** the flat-with-depth flash-attn is exactly why llama.cpp decode barely drops at ctx~1881 (51 t/s) while Camelid's collapses (campaign: 0.51× at depth).
- **Camelid equivalent:** `src/cuda_resident.rs` — `q8_gemv` (decode, measured ~76% DRAM BW — already efficient), `q8_gemm_batched` (prefill/verify), and a **split-K decode attention** (`attn_sk_partial`, committed `57977f6e`) that fills the 30 SMs at depth. Decode CUDA graph already in place.
- **Rust-native version:** already Rust+`cudarc`. The remaining gap is **deliberately parity-locked**: matching llama.cpp's flat curve needs coalesced K/V reads that re-associate the attention dot and flip near-tie greedy tokens (campaign Stage 3, reverted). llama.cpp's flat curve is *enabled by* its non-bit-exact flash attention.
- **Worth implementing now?** No. The campaign already extracted the max occupancy win strict token-parity allows. Further needs a fused QKV/gate-up kernel or non-bit-exact attention (off-limits). **Not worth chasing under the parity contract.**

## 4. CPU SIMD paths & dispatch

- **File:** `ggml/src/ggml-cpu/arch/x86/quants.c` (AVX2/AVX-512/VNNI vec-dots), runtime feature dispatch in `ggml-cpu.c`.
- **What it does:** picks the widest available ISA at runtime (here AVX-512), with VNNI `dpbusd` int8 dot where present.
- **Why it matters / Camelid status:** Camelid deliberately targets **AVX2 (x86-64-v3)**, not AVX-512 — its own 2026-06-20 characterization found AVX-512 *downclocks* this Tiger Lake chip and VNNI gave only +0.28–4.5% on memory-bound decode (reverted). So for DECODE this is a non-lever. For compute-bound PREFILL, an AVX-512 tiled GEMM is **untested** in Camelid (the characterization only tested VNNI-decode) and is the one place AVX-512 might net-win despite downclock — see §9.
- **Worth implementing now?** Only as the prefill-GEMM AVX-512 variant (§2/§9). Not as a blanket AVX-512 flip.

## 5. Tensor layout / mmap / loading

- **File:** GGUF mmap loader in `src/llama-model-loader.cpp`; weights stay quantized, mapped, paged on demand.
- **Camelid equivalent:** `src/wire_mmap.rs`, `src/gguf/reader.rs`, lazy Q8_0 file backing with a q8 file cache (`prefill_layer_major_*` paths). Camelid's "fast load" claim is real; first-token latency on CPU is dominated by prefill compute, not load.
- **Worth implementing now?** No gap identified. Parity-neutral.

## 6. Threading

- **File:** OpenMP threadpool + the in-kernel `current_chunk` scheduler (§1).
- **Why it matters:** llama.cpp schedules work *inside* the GEMM, so threads rarely idle on odd shapes. Camelid splits generic f32 with Rayon but runs many Q8 decode projections serially (one row), and its 2026-06-20 data shows memory-contention-limited scaling (decode peaks at 4 threads on this box).
- **Rust-native version:** fold the chunk scheduler into the §2 GEMM owner (Rayon `par_chunks` over output tiles). **Couples to §2.**

## 7. Batching / prompt processing

- llama.cpp: same `mul_mat` serves a [n_tokens × hidden] activation — one weight read per chunk (the §1/§2 amortization). Camelid: `forward_prefill_chunk_timed_fast` / `forward_greedy_verify_chunk` already pass the whole chunk in (structurally batched) — the deficit is the inner kernel (§2), confirmed by measurement (prefill routing/chunk-size sweep moved <3%).

## 8. Sampling & server overhead

- llama.cpp greedy argmax is trivial; server is C++ with prompt cache (`cache_prompt`). Camelid: Axum SSE server + a prompt-prefix cache (discovered empirically — a repeat-turn win, also a benchmarking trap). Sampling/streaming are **not** the bottleneck in either (decode stage dominated by the matvec). **Not worth chasing.**

## 9. Build flags / platform acceleration

- llama.cpp here: Release + AVX-512 + FMA + tinyBLAS + REPACK + OpenMP (CPU); CUDA FA + graphs (GPU).
- Camelid here: `.cargo/config.toml target-cpu=x86-64-v3` (AVX2/FMA/BMI2 autovec) + `[profile.release] lto="fat", codegen-units=1` — the historical "+39%" is already captured. AVX-512 deliberately excluded.
- **One untested idea:** an AVX-512 tiled GEMM *for prefill only* (compute-bound, amortizes the downclock). Research-lane (§2/§4); parity-safe if int-accumulation order is fixed.

---

## Net verdict

- The **CPU** gap is explained by §1+§2: llama.cpp's single tiled tinyBLAS GEMM with in-kernel scheduling and AVX-512, vs Camelid's AVX2 fragmented per-role kernels. Closing it is the "unified Q8 GEMM owner" refactor (P1, medium-high effort, parity-safe).
- The **CUDA** gap is §3: near-optimal already; the residual is a *deliberate* parity choice (no non-bit-exact flash attention). Not worth chasing.
- Sampling, server, mmap, dequant strategy, KV layout: **no actionable gap** under the contract.
