# ORNITH 9B — Phase 0 Reconnaissance Receipt

**Date:** 2026-06-29 · **Box:** Windows x86_64 (MSVC), RTX 3060 **Laptop 6 GB**, 15.74 GiB RAM
**Author lane:** ORNITH_9B_BRINGUP_CONDUCTOR Phase 0 (NO engine code written)
**Status:** Phase 0 COMPLETE. **Premise refuted — STOP for scope decision before Phase 1.**

---

## TL;DR

The brief's load-bearing assumption — *"9B = Qwen-3.5 lineage, reuses the `qwen3` arch path,
minimal loader deltas → Runnable"* — is **REFUTED**. Ornith-1.0-9B is arch **`qwen35`**, a
**hybrid State-Space-Model (SSM / linear-attention) + sparse full-attention** multimodal model.
Camelid has **no SSM / linear-attention compute path of any kind**. This is not a bring-up; it is a
new-architecture engine build comparable to (arguably larger than) the MoE work the brief deferred.

One upside correction to the brief: the **oracle is NOT blocked** — the pinned llama.cpp `acd79d6`
already defines `LLM_ARCH_QWEN35` with SSM tensors. So *if* Camelid implements the arch, the
Supported/parity lane is open.

---

## 0.1 Arch identity — GROUND TRUTH

Source: HF `deepreinforce-ai/Ornith-1.0-9B/config.json` (base safetensors) +
HTTP-range read of the first 3 MB of `deepreinforce-ai/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q8_0.gguf`
(header only — **full file NOT downloaded**).

- **GGUF `general.architecture = qwen35`** (HF `model_type = qwen3_5`,
  `architectures = ["Qwen3_5ForConditionalGeneration"]`).
- Dense text decoder, **32 blocks**, hidden 4096, intermediate 12288, RMS-norm eps 1e-6,
  **untied** embeddings, vocab **248320** (BPE / gpt2 tokenizer, pre=`qwen35`).
- Attention: 16 Q heads, **4 KV heads** (GQA), **head_dim 256**, `attn_output_gate: true`
  (**gated attention output**), **partial RoPE `partial_rotary_factor = 0.25`**, rope_theta 1e7,
  **interleaved mRoPE** (`mrope_section [11,11,10]`, `rope.dimension_sections`), max_pos 262144.

### The blocker — hybrid SSM layers
`layer_types` = repeating **[linear, linear, linear, full] × 8** (`full_attention_interval = 4`):
**24 linear-attention (SSM) layers + 8 full-attention layers.** GGUF confirms the SSM metadata block:
`qwen35.ssm.conv_kernel` (=4), `.state_size`, `.group_count`, `.time_step_rank`, `.inner_size`.
HF config: `linear_conv_kernel_dim 4`, `linear_key_head_dim 128`, `linear_num_key_heads 16`,
`linear_value_head_dim 128`, `linear_num_value_heads 32`, `mamba_ssm_dtype float32`.
→ A gated-delta-net / Mamba-style **conv1d + selective state recurrence** lane.

### Also present (text-only inference can ignore, but token plumbing remains)
- **Vision tower** (`qwen3_5_vision`, 27-layer ViT, image/video tokens) — it is a VLM.
- **MTP head** (`mtp_num_hidden_layers: 1`) — multi-token-prediction extra head.

### Loader-delta verdict vs Camelid's Qwen3 path: **NOT a claim. Net-new architecture.**
Camelid is a pure-transformer engine. `grep` for mamba/ssm/linear_attention/qwen3_5 in `src/`
returns only **label strings** (`phi_falcon_mamba_others` capability-group id, a `"mamba"` family
label in `model.rs:1715`) — **zero compute path**. Implementing `qwen35` requires, at minimum:
SSM conv1d + selective-scan recurrence lane · gated attention output · partial + sectioned-mRoPE ·
hybrid layer scheduler · MTP-head handling. This is multi-week engine R&D, not "minimal deltas".

## 0.2 Chat template diff
Deferred — secondary to the arch blocker. ChatML + reasoning channel as briefed; `chat_template`
KV present in GGUF. To be diffed in Phase 2 *if* the arch build is authorized.

## 0.3 Tool-call / reasoning byte capture
Deferred — requires a running model; `ollama` is not installed on this box and no engine path exists.

## 0.4 Oracle availability (G-ORACLE) — **YES (correction to brief)**
Pinned comparator: llama.cpp `acd79d6` (build 9632) at `<home>\llama.cpp`.
`src/llama-arch.cpp` defines `LLM_ARCH_QWEN35 = "qwen35"` and `LLM_ARCH_QWEN35MOE`, plus
`LLM_TENSOR_SSM_ALPHA // qwen3.5`. The arch is wired in the pin → **the Supported/parity lane is
NOT oracle-blocked**. (Forward-pass smoke against the actual GGUF still to be run to fully certify
G-ORACLE=YES, but the arch is present, contra the brief's prediction of NO.)

## 0.5 Asset decision — Q8_0 directly available, no conversion
Official `deepreinforce-ai/Ornith-1.0-9B-GGUF` ships, downloadable directly:
`Q4_K_M 5.63 GB · Q5_K_M 6.47 GB · Q6_K 7.36 GB · Q8_0 9.53 GB · bf16 17.92 GB`.
**Recommended bring-up quant: Q8_0** (decouples from in-flight K-quant work, no convert step needed).
**Not yet downloaded** — pending the scope decision below.

## Hardware reality (correction to brief)
Brief assumes RTX 3060 **12 GB**; this box is RTX 3060 **Laptop 6 GB** + 15.74 GiB RAM.
Q8_0 (9.5 GB) does **not** fit resident on 6 GB → realistic first light is CPU or VRAM+host offload
(like the existing 8B Qwen3 row), which *compounds* the build: the SSM lane is needed on CPU first.

---

## Decision (STOP — Tim's call, precedent: "Q4_K AVX2 already shipped" stopped at Phase 0)

The brief authorized a **bring-up**. Recon shows a **new-architecture engine build (SSM hybrid lane)**.
Not a no-op — the work is real and the oracle exists — but it is far larger than billed. Options:

- **A. Authorize the qwen35 SSM engine build** (multi-week): SSM/linear-attn lane → load+coherence
  (Q8_0, CPU/offload on this 6 GB box) → template/reasoning/tool-lift → parity vs the `acd79d6`
  oracle (open!) → 4/4 agent eval → Supported lane. Honest but large.
- **B. Re-scope to a thin Runnable probe only**: download Q8_0, drive it through the pinned
  llama.cpp oracle directly (proves the asset + oracle), document `qwen35` as *recognized but
  unimplemented in Camelid*, defer the engine. Cheap, honest, no false "supported" claim.
- **C. Park** behind the existing K-quant / agent campaigns; revisit when a bigger (12 GB+) host is
  available, since this box can't hold Q8_0 resident anyway.

**No engine code, no 9.5 GB download, and no commit have been made.** (Repo is mid-work on
`eval/qwen3-tool-capable-rung3` with unrelated dirty files; this receipt is written, not yet
committed/branched, pending the call.)
