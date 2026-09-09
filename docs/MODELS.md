# Model catalog and validation

Camelid deliberately supports exact model-and-quantization combinations rather than entire model families. Each supported file is validated token-for-token against a pinned llama.cpp reference. Files outside the supported set fail closed instead of silently using an unverified path.

## Model suggestions

Support is specific to the exact file and execution path. Downloadable models may still be experimental or awaiting validation. [Back to the quick start](../README.md#quick-start).

| Goal | Model | Pull ID |
|---|---|---|
| Smallest end-to-end test (~1.2 GB) | TinyLlama 1.1B Chat Q8_0 | `tinyllama` |
| **Recommended first model** | Llama 3.2 3B Instruct Q8_0 | `3b_instruct_q8` |
| Compact Windows CPU or tested M4 Metal chat (~2.9 GB) | LFM2.5 2.6B Q8_0 | `lfm2_5_2_6b` |
| Local embeddings and semantic retrieval | Nomic Embed Text v1.5 Q8_0 | `nomic` |
| Fits a 16 GB Apple Silicon Mac | Mistral 7B Instruct v0.3 Q8_0 | `mistral` |
| 12B reasoning on a tested 16 GB M4 Mac (**validation pending**) | Gemma 4 12B-It QAT Q4_0 | `gemma4_12b_it_qat_q4_0` |
| Reasoning and coding on a small budget | Qwen3 4B Q4_K_M | `qwen3_4b_q4` |
| Compact PrismML GPU model | Bonsai 4B Q1_0 | `bonsai_4b_q1` |
| PrismML browser/API vision | Bonsai 27B Q1_0 | `bonsai_27b_q1` |

### Full `camelid pull` catalog

Run `camelid pull <id>` to download a model into `./models`. Pull IDs resolve by unique substring; if a fragment matches several rows, Camelid lists the matches instead of guessing.

| Model | Quant | Arch | Size | Pull ID | GGUF file |
|---|---|---|---:|---|---|
| **Microsoft BitNet b1.58 2B 4T** *(experimental)* | `I2_S` | `bitnet-b1.58` | 1.2 GB | `bitnet_b1_58_2b_4t_i2_s` | `ggml-model-i2_s.gguf` |
| **Microsoft BitNet Embedding 0.6B** *(experimental)* | `I2_S` | `qwen3` | 0.4 GB | `bitnet_embedding_0_6b_i2_s` | `bitnet-embeddings-0.6b-bf16-i2_s.gguf` |
| **Microsoft BitNet Embedding 270M** *(experimental)* | `I2_S` | `gemma3` | 0.4 GB | `bitnet_embedding_270m_i2_s` | `bitnet-embeddings-270m-bf16-i2_s.gguf` |
| **Nomic Embed Text v1.5** | `Q8_0` | `nomic-bert` | 0.15 GB | `nomic` | `nomic-embed-text-v1.5.Q8_0.gguf` |
| **TinyLlama 1.1B Chat** | `Q8_0` | `llama` | 1.2 GB | `tinyllama` | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` |
| **Llama 3.2 1B Instruct** | `Q8_0` | `llama` | 1.3 GB | `1b_instruct_q8` | `Llama-3.2-1B-Instruct-Q8_0.gguf` |
| **Llama 3.2 1B Instruct** | `IQ4_XS` | `llama` | 0.7 GB | `iq4_xs` | `Llama-3.2-1B-Instruct-IQ4_XS.gguf` |
| **Llama 3.2 3B Instruct** | `Q8_0` | `llama` | 3.4 GB | `3b_instruct_q8` | `Llama-3.2-3B-Instruct-Q8_0.gguf` |
| **Llama 3.2 3B Instruct** | `Q4_K_M` | `llama` | 2.0 GB | `3b_instruct_q4` | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` |
| **Llama 3.2 3B Instruct** | `Q5_K_M` | `llama` | 2.3 GB | `3b_instruct_q5` | `Llama-3.2-3B-Instruct-Q5_K_M.gguf` |
| **Llama 3 8B Instruct** | `Q8_0` | `llama` | 8.5 GB | `llama3_8b` | `Meta-Llama-3-8B-Instruct.Q8_0.gguf` |
| **Llama 3.1 8B Instruct** | `Q8_0` | `llama` | 8.5 GB | `llama31_8b` | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` |
| **Gemma 3 1B-It** | `Q8_0` | `gemma3` | 1.1 GB | `gemma_3_1b` | `gemma-3-1b-it-Q8_0.gguf` |
| **Gemma 4 E2B-It** | `Q8_0` | `gemma4` | 5.0 GB | `gemma4_e2b` | `gemma-4-E2B-it-Q8_0.gguf` |
| **Gemma 4 E4B-It** | `Q8_0` | `gemma4` | 8.2 GB | `gemma4_e4b` | `gemma-4-E4B-it-Q8_0.gguf` |
| **Gemma 4 12B-It QAT** — fast single-Mac Metal lane *(active validation)* | `Q4_0` | `gemma4` | 7.0 GB | `gemma4_12b_it_qat_q4_0` | `gemma-4-12b-it-qat-q4_0.gguf` |
| **Gemma 4 12B-It** — legacy two-Mac distributed lane | `Q8_0` | `gemma4` | 12.7 GB | `gemma4_12b` | `gemma-4-12b-it-Q8_0.gguf` |
| **Gemma 4 26B-A4B-It QAT** — single-Mac Ghost-MoE/MTP | `Q4_0` | `gemma4` | 14.4 GB | `gemma4_26b` | `gemma-4-26B_q4_0-it.gguf` |
| **Qwen3 0.6B** | `Q8_0` | `qwen3` | 0.6 GB | `qwen3_0_6b` | `Qwen3-0.6B-Q8_0.gguf` |
| **Qwen3 1.7B** | `Q8_0` | `qwen3` | 1.8 GB | `qwen3_1_7b` | `Qwen3-1.7B-Q8_0.gguf` |
| **Qwen3 4B** | `Q8_0` | `qwen3` | 4.3 GB | `qwen3_4b_q8` | `Qwen3-4B-Q8_0.gguf` |
| **Qwen3 4B** | `Q4_K_M` | `qwen3` | 2.5 GB | `qwen3_4b_q4` | `Qwen3-4B-Q4_K_M.gguf` |
| **Qwen3 8B** | `Q8_0` | `qwen3` | 8.7 GB | `qwen3_8b` | `Qwen3-8B-Q8_0.gguf` |
| **Qwen3 14B** *(active validation; not supported)* | `Q4_K_M` | `qwen3` | 9.0 GB | `qwen3_14b` | `Qwen3-14B-Q4_K_M.gguf` |
| **Mistral 7B Instruct v0.3** | `Q8_0` | `llama` | 7.7 GB | `mistral` | `Mistral-7B-Instruct-v0.3-Q8_0.gguf` |
| **Mistral Nemo Instruct 2407** *(validation hold; not supported)* | `Q4_K_M` | `llama` | 7.5 GB | `mistral_nemo` | `Mistral-Nemo-Instruct-2407.Q4_K_M.gguf` |
| **LFM2.5 2.6B** *(supported exact-row smoke)* | `Q8_0` | `lfm2` | 2.9 GB | `lfm2_5_2_6b` | `LFM2.5-2.6B-Q8_0.gguf` |
| **Phi-3-mini-4k-instruct** *(supported exact-row smoke on Windows x86_64)* | `Q8_0` | `phi3` | 4.1 GB | `phi3` | `Phi-3-mini-4k-instruct-Q8_0.gguf` |
| **DeepSeek R1 Distill Qwen 7B** | `Q8_0` | `qwen25` | 8.1 GB | `distill_qwen` | `DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf` |
| **DeepSeek R1 Distill Llama 8B** | `Q8_0` | `llama` | 8.5 GB | `distill_llama` | `DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf` |
| **DeepSeek R1 0528 Qwen3 8B** *(validation hold; not supported)* | `Q4_K_M` | `qwen3` | 5.0 GB | `deepseek_r1_0528` | `DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf` |
| **Qwen2.5 Coder 7B** | `Q8_0` | `qwen25` | 8.1 GB | `qwen25_coder` | `qwen2.5-coder-7b-instruct-q8_0.gguf` |
| **Ornith 1.0 9B** — hybrid DeltaNet, `tool_capable` | `Q8_0` | `qwen35` | 9.5 GB | `ornith` | `ornith-1.0-9b-Q8_0.gguf` |
| **Bonsai 4B** | `Q1_0` | `qwen3` | 0.6 GB | `bonsai_4b_q1` | `Bonsai-4B-Q1_0.gguf` |
| **Ternary Bonsai 4B** | `Q2_0` | `qwen3` | 1.1 GB | `bonsai_4b_q2` | `Ternary-Bonsai-4B-Q2_0.gguf` |
| **Ternary Bonsai 4B** | `PQ2_0` | `qwen3` | 1.1 GB | `bonsai_4b_pq2` | `Ternary-Bonsai-4B-PQ2_0.gguf` |
| **Bonsai 8B** | `Q1_0` | `qwen3` | 1.2 GB | `bonsai_8b_q1` | `Bonsai-8B-Q1_0.gguf` |
| **Ternary Bonsai 8B** | `Q2_0` | `qwen3` | 2.2 GB | `bonsai_8b_q2` | `Ternary-Bonsai-8B-Q2_0.gguf` |
| **Bonsai 27B** | `Q1_0` | `qwen35` | 3.8 GB | `bonsai_27b_q1` | `Bonsai-27B-Q1_0.gguf` |
| **Ternary Bonsai 27B** | `Q2_0` | `qwen35` | 7.2 GB | `bonsai_27b_q2` | `Ternary-Bonsai-27B-Q2_0.gguf` |

The 7.0 GB Gemma 4 12B QAT Q4_0 row is the recommended 12B download. It runs on a single tested 16 GB M4 Mac through Camelid's accelerated Metal path; its exact-row support promotion is still in progress, so the Models page labels it as validation pending. The older 12.7 GB Q8_0 artifact remains available for the validated two-host distributed lane. Nine hash-pinned Phase 2 rows are **Runnable with disclosed reference-output variance**: LFM2.5 1.2B Thinking Q8_0, Gemma 3 4B-It Q8_0, Llama 3.1 8B Instruct Q8_0, Qwen 2.5 0.5B/1.5B Instruct Q8_0, Qwen 3.5 4B/9B Q8_0, DeepSeek R1 Distill Qwen 1.5B Q8_0, and Aya Expanse 8B Q4_K_M. They use the normal download-and-start path and may be used for local chat; one or more strict greedy token-ID probes differ from pinned llama.cpp, so the UI keeps an amber warning and withholds Verified/Supported, tools, and broader-context claims.

Mistral Nemo Instruct 2407 Q4_K_M, Qwen3 14B Q4_K_M, and DeepSeek R1 0528 Qwen3 8B Q4_K_M are downloadable catalog rows, not supported rows. The pinned bring-up evidence records Mistral Nemo cross-backend divergence and a blocked external comparator, Qwen3 14B without an external oracle or chat/API proof, and DeepSeek cross-backend divergence plus a missing native R1 marker/tool renderer. None inherits support from its architecture or a smaller sibling.

The three BitNet rows are bring-up targets, not promoted support rows yet. Camelid
can parse and execute their official canonical `I2_S` GGUF graphs through cleanroom
CPU, Metal, and CUDA projection kernels. Runtime-selectable `i2_s`, `tl1`, and `tl2`
strategies operate on the same published bytes; they do not claim compatibility with
BitNet.cpp's separately permuted TL files. Reference parity and bounded-context /
embedding-vector receipts remain outstanding. See
[the BitNet runtime notes](architecture/BITNET.md).

Gemma 4 26B-A4B QAT does **not** require two Macs. It runs on one 16 GB M4 Mac
through the opt-in Ghost-MoE Metal lane, which repacks the routed experts into a
paged `.cghost` artifact and keeps a bounded, persistent expert working set in unified memory
(measured 17–20 tok/s steady-state decode on the tested M4). The distributed lane remains
available as an alternative, but it is not a requirement. Replies on the single-Mac lane are
currently marked experimental with no parity guarantee. Setup, the recommended serve profile,
and measured receipts live in [docs/runtime/ghost-mode.md](runtime/ghost-mode.md#ghost-moe-v2-gemma-4-26b-a4b).

The full catalog, exact hashes, supported execution paths, and claim boundaries live in:

- [COMPATIBILITY.md](../COMPATIBILITY.md)—authoritative supported-row ledger
- [SUPPORT_MATRIX_v0.1.md](reference/SUPPORT_MATRIX_v0.1.md)—per-row support boundaries
- [RECEIPTS.md](reference/RECEIPTS.md)—reproducible validation receipts
- [benchmarks](benchmarks/BENCHMARKS.md)—recorded performance measurements

## Validation highlights

| Model row | Quant | Evidence |
|---|---|---|
| Mistral 7B Instruct v0.3 | Q8_0 | Exact-row smoke + bounded context 512→8192 + GPU/CPU parity |
| LFM2.5 2.6B | Q8_0 | Hash-pinned exact-row smoke on Windows CPU/runnable and Apple M4 macOS 26.5 arm64 resident Metal: 96/96 short greedy tokens, exact 512-token chat prompt + 8/8 reference-oracle tokens/text, and API/Models-page/WebUI/SSE smoke |

The LFM2.5 promotion is limited to `LiquidAI/LFM2.5-2.6B-GGUF@b421ad1d549afeda6a0fb2ad3a697cb5a7879adc`, file `LFM2.5-2.6B-Q8_0.gguf` (2,874,779,456 bytes, SHA-256 `36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757`). The Windows x86_64 CPU/runnable proof remains recorded at `qa/evidence-bundles/lfm2-2.6b-q8-phase1-promotion-20260810/`. The Apple M4 macOS 26.5 arm64 receipt at `qa/evidence-bundles/lfm2-2.6b-q8-macos-metal-20260810-head-d31e5cb0/` independently asserts the resident-Metal execution plan, 96/96 short greedy token IDs, the exact 512-token prompt plus 8/8 generated IDs/text against pinned llama.cpp b9632 (`acd79d603`), and API/Models-page/WebUI non-streaming plus 128-ceiling SSE smoke. The 512-token/8-token oracle checks are reference-only; raw `/v1/completions` and tools remain typed fail-closed. Sampling beyond deterministic greedy, context above 512, neighboring rows, production throughput, CUDA, other Apple hardware, broad platform portability, and broader LFM2 support remain unclaimed.

The Phi-3 promotion is limited to `Phi-3-mini-4k-instruct-Q8_0.gguf` (4,061,221,376 bytes, SHA-256 `0ac8ee48aeebf7d1b354691fd1e29e91c32ad88bbad10ad45ac880dcd4372a47`) on Windows x86_64 using the CPU-reference prefill/decode lane. The receipt at `qa/model-qualification/phi3-mini-windows-support-20260811.json` records successful API load/readiness and exact agreement with pinned llama.cpp `acd79d603` for generated IDs `[3681, 29889, 13, 32001, 3869]`. Bounded context packs, performance, tools, neighboring Phi files, Linux, and macOS remain unclaimed.

### Multimodal image chat

Seven hash-pinned PrismML Bonsai GGUFs are supported on Apple Silicon Metal and Windows x86_64 CUDA: 4B Q1/Q2/PQ2, 8B Q1/Q2, and 27B Q1/Q2. Both 27B rows support multimodal PNG/JPEG input in browser chat and OpenAI-compatible Chat Completions when paired with the Qwen3-VL projector.

The **Arch** column reports what each GGUF declares in `general.architecture`, so it is not uniform across this family: the 4B and 8B files declare `qwen3` and only the 27B files declare `qwen35`. The two labels bind different engines, so the split is real rather than a typo.

The desktop **Models** page downloads the projector automatically with either 27B model. For a CLI installation, place `Ternary-Bonsai-27B-mmproj-Q8_0.gguf` beside the model GGUF or set `CAMELID_MMPROJ`, then run `camelid serve` normally. See [COMPATIBILITY.md](../COMPATIBILITY.md) for the exact artifacts and validated scope.

### Embeddings and reranking

The exact Nomic Embed Text v1.5 Q8_0 row supports OpenAI-compatible `/v1/embeddings`, Matryoshka dimensions, cosine-similarity reranking through `/v1/rerank`, and optional in-memory semantic retrieval for Workspace. The encoder currently runs on CPU; other embedding families and quantizations fail closed. See the [embedding API guide](architecture/EMBEDDINGS.md) for loading and request examples.
