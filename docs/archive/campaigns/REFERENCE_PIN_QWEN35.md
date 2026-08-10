# REFERENCE_PIN_QWEN35 — oracle pin for the qwen35 (Ornith-1.0-9B) lane

Conductor: `ORNITH_9B_CONSTRAINED_VRAM_CONDUCTOR.md`, Item 0.
Status: **PINNED — `acd79d6` (build 9632), CUDA build, on the Windows target.**
Date: 2026-07-02.

## The pin

| field | value |
|---|---|
| `REF_QWEN35` commit | `acd79d6` ("jinja : add count/d/e filter aliases (#24606)"), llama.cpp build 9632 |
| Build location | `<home>\llama.cpp\build\bin\` (`llama-cli.exe`, `llama-server.exe`, `llama-quantize.exe`, `llama-imatrix.exe`, `llama-perplexity.exe`, `llama-tokenize.exe`, `llama-speculative.exe`) |
| Compiler | MSVC 19.44.35228.0, Windows AMD64 |
| Build flags | `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_COMPRESSION_MODE=size`, generator Ninja |
| CUDA backend | `ggml-cuda.dll` present; CUDA toolkit 12.9 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, driver 576.83 |
| OS | Windows 11 Home, build 26220 |
| Oracle invocation (exactness lanes) | `llama-cli`/`llama-server` with CUDA, `-ngl 99` (partial offload acceptable for oracle-correctness runs — the oracle must be correct, not fast) |

`llama-imatrix.exe`, `llama-perplexity.exe`, `llama-tokenize.exe` were built 2026-07-02 from
the same pinned checkout into the same build tree (no source change; targets were simply
not built before). All quant production (Item 4) and tokenizer/PPL work uses these.

## Why `acd79d6`, not "the most recent release tag"

The conductor's Item 0 instructs selecting the most recent llama.cpp release whose CUDA
backend passes its own qwen35 tests, on the premise that "the legacy parity pin `acd79d6`
predates `qwen35` and CANNOT load this model." **That premise is factually wrong, verified
three independent ways:**

1. The on-disk `acd79d6` binaries contain `LLM_ARCH_QWEN35` and the SSM tensor
   definitions (verified at the binary level during bringup — see
   `qa/ornith/G-LOAD-qwen35-coherence.md`).
2. The committed **G-PARITY receipt** (`qa/ornith/G-PARITY-qwen35-vs-llamacpp.md`, merged
   via PR #350) shows Camelid greedy-token-identical to `acd79d6` on 4/4 prompts with
   this exact model — the oracle loads, runs, and produces coherent, cross-validated
   output.
3. The `acd79d6` CUDA build loaded `ornith-1.0-9b-Q4_K_M.gguf` with `-ngl 99` on this
   GPU (5221 MiB used / 776 MiB free) and generated coherent text (bringup session,
   re-smoked for `RECEIPT_ITEM0_reference.json`).

Given that, the decision criteria become: (a) is there an upstream qwen35/SSM
**correctness** fix after `acd79d6` that we would be pinning ourselves out of, and
(b) what does a newer pin cost?

**(a) — checked 2026-07-02** (`git fetch` upstream, 228 commits `acd79d6..origin/master`):

- `git log acd79d6..origin/master -- src/models/qwen35.cpp src/models/delta-net-base.cpp`
  shows only MTP/EAGLE3 speculative-decoding feature work and an `hparams.n_layer`
  refactor — no correctness fix to the core qwen35 graph.
- `git log acd79d6..origin/master -- ggml/src/ggml-cuda` contains **zero** ssm/delta/qwen35
  hits — no CUDA SSM correctness fixes.
- The known Vulkan `ggml_ssm_conv`/`ggml_ssm_scan` corruption (upstream #19957) is a
  Vulkan-backend issue; we use CUDA only (per this conductor's ground rules), and the
  one Vulkan delta-net commit upstream (`d5fb10429`) is irrelevant to a CUDA oracle.

**(b)** — every existing qwen35 receipt (G-LOAD/G-PARITY/G-TOOLCALL/G-AGENT, merged on
main) is pinned to `acd79d6`. Re-pinning to a newer tag would fork the receipt chain and
force re-validation of the entire bringup for zero identified correctness benefit.

**Decision: `REF_QWEN35` := `acd79d6` CUDA build.** This also collapses the conductor's
dual-pin bookkeeping in practice: the legacy pin and `REF_QWEN35` are the same commit in
two build configurations (legacy CPU-only reference at `<home>\tools\llama-cpp\`
build b9632, used by the TinyLlama/llama-family lanes; CUDA build above for qwen35).
Receipts still name which binary they used; "never mix pins in one receipt" is preserved
trivially.

## Reference artifacts (hash manifest)

Source repo: `huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF`. Expected SHA256 =
the repo's LFS OIDs, independently recomputed locally after download.

| file | bytes | SHA256 | provenance |
|---|---|---|---|
| `ornith-1.0-9b-Q8_0.gguf` | 9,527,500,992 | `d0e4bebaa8b3450c62090df1408f2ee5ccb2094f9c610ffde564a654483d4f37` | HF pristine — **local recompute MATCHES the HF LFS oid** |
| `ornith-1.0-9b-Q4_K_M.gguf` (local) | 5,629,108,416 | `2711bf1ef034fa39eb899f793fe63bbb0aac21ebdacbcbe09406b5600ad5188f` | **home requant** from Q8_0 via `llama-quantize --allow-requantize` at `acd79d6` (GPU-lane bringup artifact; NOT the HF file) |
| `ornith-1.0-9b-Q4_K_M.gguf` (HF, not downloaded) | 5,629,108,704 | `5720d1f671b4996481274fffe01868c3c36e87c135cc8538471cc7bd6087b106` | HF LFS oid, recorded for disambiguation — quantized from bf16 upstream |
| `ornith-1.0-9b-Q6_K.gguf` | 7,359,259,072 | `33b6f6a3e3f05078438e12df8a4b55c8acf78ceadcc639d2af1cf35a026e8387` (HF LFS oid; local recompute pending download) | HF pristine (Item 6 verifier weights) |
| `ornith-1.0-9b-bf16.gguf` | 17,920,696,512 | `27bc753487eed85539c3aef63dd602b79cd060401b928c9ff7d30d5556eca260` (HF LFS oid; local recompute pending download) | HF pristine (Item 4 quant-production source) |

All files under `<home>\Camelid\models\`. This table is finalized in
`qa/ornith/constrained-vram/RECEIPT_ITEM0_reference.json` once local recomputes of the
two downloads complete.

**Quality note for Item 4:** the residency/parity receipts of the existing GPU lane were
produced with the *home-requant* Q4_K_M (Q8→Q4 without imatrix). Item 4's quality table
must therefore include the HF-style provenance chain (bf16 → imatrix quant) and should
treat the home requant as a distinct row, not as "Q4_K_M".
