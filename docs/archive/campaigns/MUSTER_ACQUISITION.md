# MUSTER Phase 2 — Acquisition & anchoring

Campaign: [`MUSTER_CONDUCTOR.md`](./MUSTER_CONDUCTOR.md) §5. Executed 2026-07-16 on the Gate 1-signed tree (`main` merge `b80788c9`). Every SHA-256 below is the anchor **every later artifact must repeat verbatim**; upstream verification used the Hugging Face tree API (`lfs` size + sha256) — no model was run. Raw sidecars live under the gitignored Phase 2 run dir.

M-B5 and M-B6 exited the pipeline at this phase with sealed HOLD receipts: [`qa/muster/HOLD-ornith-1.0-9b-bf16.json`](../../../qa/muster/HOLD-ornith-1.0-9b-bf16.json), [`qa/muster/HOLD-ornith-1.0-9b-IQ3_XXS.json`](../../../qa/muster/HOLD-ornith-1.0-9b-IQ3_XXS.json) (named blockers, oracle-side controls, exit conditions). COMPATIBILITY.md's Ornith non-claim now points at both receipts.

## Anchors (live rows)

| Row | Exact file | Size (B) | SHA-256 | Source | License | Acquired |
|---|---|---|---|---|---|---|
| M-A1 | `gemma-3-1b-it-Q8_0.gguf` | 1,069,306,368 | `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5` | `ggml-org/gemma-3-1b-it-GGUF` — **upstream exact match verified 2026-07-16** (size + full sha, live tree); also matches the catalog literal and the committed runnable receipt's `gguf_sha256` | gemma (repo not gated) | pre-campaign; re-anchored this phase |
| M-A2 | `Phi-3-mini-4k-instruct-Q8_0.gguf` | **4,061,221,376** | `0ac8ee48aeebf7d1b354691fd1e29e91c32ad88bbad10ad45ac880dcd4372a47` | `bartowski/Phi-3-mini-4k-instruct-GGUF` — single upload series 2024-04-29, never re-uploaded; local hash matches the live tree exactly | MIT (not gated) | **2026-07-16 via `camelid pull phi3_mini`** |
| M-B1 | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 807,693,984 | `6a74661014a3e2f139871f81e6cec852c489a627d169de503a3c0434a10c503d` | **UNRESOLVED — see below** | Llama 3.2 Community License (inherited from the base model; no repo to cite) | on disk since 2026-07-12; no acquisition record |
| M-B2 | `ornith-1.0-9b-Q6_K.gguf` | 7,359,259,072 | `33b6f6a3e3f05078438e12df8a4b55c8acf78ceadcc639d2af1cf35a026e8387` | `deepreinforce-ai/Ornith-1.0-9B-GGUF` — **upstream exact match verified 2026-07-16** (size + sha, live tree); corroborates QUANT_QUALITY_TABLE's "HF pristine" | MIT (not gated) | pre-campaign; re-anchored this phase |
| M-B3 | `ornith-1.0-9b-IQ4_XS.gguf` | 5,196,440,096 | `0e267369ffbbfcdbdc50241db62a865942a155fb5dfa041f7e8518949b5df7b9` | **In-house requant** (no upstream file exists — verified: the source repo's tree carries no IQ4_XS): produced at pinned llama.cpp `acd79d6` from the upstream-verified bf16 (sha `27bc7534…`) + committed imatrix `qa/ornith/constrained-vram/imatrix_ornith_agentic.gguf` | MIT (derivative of the MIT upstream) | pre-campaign (Ornith constrained-VRAM campaign) |
| M-B4 | `qwen2.5-0.5b-instruct-q4_0.gguf` | 428,730,208 | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (official) — **upstream exact match verified 2026-07-16** (size + full sha, live tree) | Apache-2.0 | pre-campaign; re-anchored this phase |

## M-A2 stop-and-amend (conductor Amendment A-5) and Phase 2 verify results

The catalog literal and the conductor's Wave A table baked **4,061,222,688 B** for the phi3 file; the live tree shows bartowski's only upload (2024-04-29, 4 commits total, never re-uploaded) is **4,061,221,376 B**. The baked size matched no upload that ever existed. The catalog literal is corrected in this commit; the anchor above is the live-verified pair. `camelid pull phi3_mini` resolved and downloaded against the live Hub size as designed.

Inspect results against the recon's Phase 2 verify list (metadata-only):

- **B-SWA blocker RESOLVED for this exact file:** the GGUF carries **no `phi3.attention.sliding_window` key** (April-2024 export predates llama.cpp's phi3-SWA change), so the pinned oracle will not apply SWA and Camelid's lack of phi3 SWA does not constrain the parity envelope **for this file**. Any other/newer phi3 GGUF reopens the blocker.
- Dims confirmed: 32 layers, embedding 3072, heads 32/32 (MHA), rope dim 96, ffn 8192, ctx 4096, `file_type` 7 (Q8_0); tensor mix Q8_0 linears + F32 norms.
- Tokenizer confirmed SPM (`tokenizer.ggml.model = "llama"`), `add_bos=true`, bos 1, eos 32000, `<|end|>` present in vocab (the live turn-end control token, covered by Camelid's additive EOG set).
- **Template divergence recorded for the Phase 3 renderer gate** (quoted verbatim from the GGUF): the template maps **both `user` and `system` roles to `<|user|>`** and emits `<|assistant|>\n` after every user/system turn; assistant messages render as bare `{content}<|end|>\n`. The in-tree `render_phi3_prompt` emits `<|system|>` for system turns and appends a trailing `<|assistant|>\n` unconditionally. Byte-identical for plain alternating user/assistant chats; **divergent for system-bearing and trailing-assistant shapes** — exactly what the Phase 3 shapes pack must adjudicate against the pinned oracle before any parity run.

## M-B1 provenance: unresolved (decision recorded, not blocking Phase 2)

The on-disk file matches **no current upload** of any surveyed publisher (2026-07-16, live trees):

| Candidate | Size (B) | SHA-256 (prefix) |
|---|---|---|
| on-disk file | **807,693,984** | **`6a746610…`** |
| bartowski (single series, 2024-09-25) | 807,694,464 | `6f85a640…` |
| unsloth | 807,694,368 | `3f5a2242…` |
| lmstudio-community | 807,690,688 | `f7ede428…` |
| MaziyarPanahi | 807,694,432 | `e4650dd6…` |
| QuantFactory | 807,694,080 | `f3cdd84d…` |
| second-state | 807,694,176 | `26bac8ef…` |

The file's metadata carries no producer marker (no `general.quantized_by`, no `quantize.*` keys); it appeared on disk 2026-07-12 with no acquisition record. Most plausible origin: a local `llama-quantize` run, undocumented. **Default path (taken):** the row proceeds SHA-anchored as a local file, and its eventual contract evidence must say "local file, upstream provenance unresolved" — never name a repo. **Alternative for Tim:** replace with the canonical bartowski upload — that is a *different exact file* (different SHA), i.e. a roster substitution, and the chip-visibility observation transfers only after a fresh load. Flag it at the next touchpoint if preferred; Phase 4 parity is equally valid either way since support is per exact file.

## Phase 2 exit state

- 6 live rows anchored (5 upstream-verified exact, 1 SHA-anchored local with unresolved upstream); 2 rows HOLD-sealed.
- Next: per-row Phase 3/4 in the signed order — M-B1 → M-A1 → M-A2 → M-B3 → M-B2 → **M-B4 pending the engine-bite authorization** (three-item bite per `MUSTER_RECON.md` §Gate 1 decisions; not yet given — the row stays anchored-but-parked until Tim rules).
