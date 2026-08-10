# BASALT eval protocol — pre-registered (Phase 0, before any NVFP4 quality number exists)

Status: **PRE-REGISTERED 2026-07-16.** No NVFP4 file of the pilot model exists yet; no NVFP4
quality number of any kind has been produced. After this file merges, changes are permitted
only by Tim, only before Phase 3 runs, and only via an explicit amendment note in this file.
Adjusting thresholds, packs, or comparison rows after seeing a result is a campaign-failing
act (MUSTER_CONDUCTOR.md:93 discipline applies verbatim).

Companion: `BASALT_CONDUCTOR.md` §5 (Phase 0 item 5), §6 (defaults). Deviations from the §6
defaults are marked **[AMENDS §6]** with rationale and require Tim's sign-off at G0.

> **Amendment record — G0 SIGNED 2026-07-16 (Tim):** both §6 deviations accepted (lane-native
> packs §3; 80% sanity guard §5.2), the §1 format-isolated row design accepted (recon T7),
> and the D-B decision set accepted as recommended (DECISIONS.md D17). No further changes
> are permitted except by Tim before Phase 3 runs, via a new amendment note here.
>
> **Amendment 2 — SIGNED 2026-07-16 (Tim, in-session), quantize-source change for RAM
> safety:** the pin's quantizer stages whole tensors through an f32 buffer plus a same-size
> output buffer (src/llama-quant.cpp:216-218, 1225-1227; both persistent); on the pilot,
> `per_layer_token_embd` (2.82 G elements) drives every BF16-source leg to ~22.5 GB of
> anonymous commit vs 21.7 GB total commit headroom on the 15.7 GB host. Accordingly:
> (a) the gated `-mm` pair is produced **from the Q8_0 baseline file** (the embeddings,
> norms, and every other same-type tensor copy through the mmap with zero staging,
> byte-identical to the baseline — stricter isolation than the original design; the sole
> exceptions are the 84 `inp_gate`/`proj` F32→Q8_0 conversions disclosed below, identical
> across the pair; `--allow-requantize` is now load-bearing);
> (b) the `Q4_K_M` rows are produced from the Q8_0 source with `--token-embedding-type
> q8_0` (covers both embedding tensors; **disclosed deviation**: embeddings Q8_0 instead of
> the ftype-default Q6_K); (c) **`NVFP4-all` is BLOCKED-HOST** — its definition requires
> re-staging the big embedding from any source; recorded with the receipt, revisit on a
> larger-RAM host. The BF16 file stays on disk as the archival source for that case. A
> known, disclosed side effect from either source: `blk.*.inp_gate/proj` (84 F32 tensors)
> quantize to Q8_0 in every produced row — identical across the `-mm` pair, so the gated
> isolation holds.
>
> **Amendment 3 addendum (signed 2026-07-16)** — the §4 phrasing-lock paragraph, verbatim:
> All quality figures in the G3 table are measured against the Q8_0 parent at matched
> 4.5 bpw. Any surfaced claim derived from this table carries the qualifier 'vs Q8_0
> parent, matched 4.5 bpw' and may not be compressed to an unqualified 'NVFP4 ≈ Q4_K_M'
> or any absolute-quality statement. Absolute-vs-BF16 characterization requires a
> separately pre-registered eval.

## 1. Pilot and rows

Pilot model: Gemma 4 E4B-it (`arch=gemma4`, 42 blocks, tied lm_head, SPM tokenizer), the
validated-lane row anchored in `tests/gemma4_forward.rs`.

The pin produces NVFP4 via the **per-tensor override path** (`--tensor-type '<regex>=nvfp4'`,
`tools/quantize/quantize.cpp:313-343` → the ggml trait-table name, case-insensitive; there is
no NVFP4 positional ftype — see `refusal_receipt.md`). The override regex therefore controls
the keep-list exactly, which enables a **format-isolated** gated comparison: two files
identical in every tensor except the matmul weight format.

Let `MM_RE` = `blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output|ffn_up|ffn_gate|ffn_down)\.weight`
(the 294 matmul weights; the Phase 2 receipt must assert from the per-tensor log that exactly
294 tensors were overridden, and record how every other tensor was treated — identical across
the two `-mm` rows by construction).

| row | gating | provenance ([Amendment 2] all produced rows from the Q8_0 baseline; data-free unless noted) |
|---|---|---|
| `Q8_0` (baseline) | reference | upstream pristine `gemma-4-E4B-it-Q8_0.gguf`, `unsloth/gemma-4-E4B-it-GGUF`, sha256 `a2232a649523c36bf530f1dc3614eb8c800645c4227390381c8b05d4d6eee05a`, 8,192,951,456 B (verify at Phase 2 download) |
| `NVFP4-mm` | **GATED** | [Amendment 2] `llama-quantize --allow-requantize --tensor-type '<MM_RE>=nvfp4' --override-kv general.file_type=int:39 <Q8_0-baseline> <out> Q8_0` |
| `Q4K-mm` | **GATED comparator** | same command with `=q4_K` and `general.file_type=int:15` — identical treatment everywhere except matmul format (uniform q4_K, 4.5 bpw, vs uniform NVFP4, 4.5 bpw); every non-matmul tensor byte-identical to the baseline in both `-mm` rows **except** the 84 disclosed `inp_gate`/`proj` F32→Q8_0 conversions, which are identical across the pair |
| `Q4_K_M-df` | report-only | [Amendment 2] `llama-quantize --allow-requantize --token-embedding-type q8_0 <Q8_0-baseline> <out> Q4_K_M` (practical shipping mixture; embeddings pinned Q8_0 — disclosed deviation from the ftype-default Q6_K) |
| `Q4_K_M-im` | report-only | as above with `--imatrix` = upstream `imatrix_unsloth.gguf_file` (4,429,024 B, sha256 `5a97d82199cd4bac77b76180620e9cef2c5108a95c880d46a312b8a6036a5753`) |
| `NVFP4-all` | **BLOCKED-HOST** [Amendment 2] | definition requires re-staging `per_layer_token_embd` (~22.5 GB commit) from any source — unrunnable on this host within the bench-memory rules; revisit on a larger-RAM box (BF16 archival source retained on disk) |

Quantization source for all produced rows **[Amendment 2]**: the Q8_0 baseline file itself
(`gemma-4-E4B-it-Q8_0.gguf`, sha256 `a2232a64…` per the table above — verify before every
quantize run). The BF16 sibling (`gemma-4-E4B-it-BF16.gguf`, sha256
`21eb0c95bad07abe57c78068d63683d61a84d5464f2809389389d8fb05b559ef`, 15,053,095,840 B) is
retained on disk **as archival source only** for the BLOCKED-HOST `NVFP4-all` case on a
future larger-RAM host. Every quantize command line and output sha256 lands in the Phase 2
bundle with the tool's full per-tensor log. Determinism: the pin quantizer was shown
byte-identical across repeat runs (refusal receipt); each produced row is quantized twice and
the sha256s must match. Exact `--tensor-type` regex dialect is verified at Phase 2 against
`parse_tensor_type`; the per-tensor log, not the regex, is the arbiter of what was hit.

Rationale (pre-registered): `NVFP4-mm` vs `Q4K-mm` is the gated comparison — it isolates the
format question (same source, same tool, same calibration budget of none, same keep-list,
same bits/weight). The two standard `Q4_K_M` rows answer the practical "what you'd actually
ship" question and expose the mixture/imatrix advantages, clearly labeled, never gated. This
amends the conductor §6 comparator naming (`Q4_K_M` → format-isolated `Q4K-mm`) — flagged for
Tim at G0 (recon open item T7). NVFP4 calibration (ModelOpt-style checkpoints with sidecar
scales) remains out of scope for v1 per `BASALT_CONDUCTOR.md` §2.

## 2. Tools (single comparator pin)

All BASALT reference legs use the repo-wide pin **llama.cpp `acd79d603` build 9632**, Windows
CPU-only build at `<pin-tools>/`:

| tool | sha256 |
|---|---|
| `llama-quantize.exe` | `055468bf00616f8dac0c4ae48fa68308dfbcdd22ec3a2eb6255b89632a6a3a4c` |
| `llama-completion.exe` | `9547c4559eed03627856587a7e7158628502923d80e5f0d445b62fbccf951ab3` |
| `llama-perplexity.exe` | `c437588c0fd07850f4d4fbcad7efcdfb751438bbd6e7ba6c12dcfe955d8ec043` |

- **`llama-cli.exe` is banned from all scripted legs** — in build 9632 it is a
  conversation-only REPL that spins infinitely on EOF (see
  `qa/evidence-bundles/basalt/phase0/incident-20260716-hard-hang.md`). Greedy reference
  generation uses `llama-completion.exe`.
- The existing gemma4 Q8_0 oracles under `qa/gemma4/oracle/` are pinned to llama.cpp
  `5d56eff` and are **not** mixed into any BASALT receipt. BASALT captures its own oracles
  with the build-9632 pin; each receipt names exactly one comparator pin.

## 3. Prompt set **[AMENDS §6: lane-native packs instead of "8 prompts × 128"]**

The §6 default (8 prompts × 128 greedy tokens, speed-column raw pack) is replaced by the
gemma4 lane's own committed packs. Rationale: gemma is SPM, and MUSTER sealed the lesson that
SPM rows cannot ride the speed-column raw pack (specials/merge-order seams — see the M-A2
HOLD receipt); the gemma4 packs are raw BOS+text completion prompts designed for exactly this
lane, with an established oracle-flag convention and committed baselines.

| pack | sha256 | prompts | greedy tokens |
|---|---|---|---|
| `qa/gemma4/prompt_packs/basic_v1.json` | `d9d54f5745d00734eff2be85b64c4f2c48c0b67342e4a761a21d696058c1fc07` | 5 | 120 |
| `qa/gemma4/prompt_packs/deep_v1.json` | `adcfef002af7f2e6810e8320fdd730ff983bffe66f19479e000877ad8fdf0609` | 4 | 200 |

Total: **9 prompts, 320 greedy continuation positions.** Optional report-only depth leg:
`context_2048_v1.json` (`de7bf900efc14b1f873651c4b35fb4ffe22919826cc5f925cc80def84456ad14`).
Decode is greedy (`--temp 0`) everywhere; all receipts `reproducible: true`; determinism
spot-check = one repeated capture (×2) per engine per row.

## 4. G3(a) — token parity, CPU, cross-engine

Camelid NVFP4 CPU greedy vs pin `llama-completion` CPU greedy, **same NVFP4 file**, all 9
prompts at full pack budgets. Reference flags follow the gemma4 oracle convention:
`--no-repack -fa off -ctk f32 -ctv f32 -ub 1` (Camelid's gemma4 KV cache is f32; the
conductor §2 mention of "kv_f16" does not apply to this pilot — see BASALT_RECON.md errata).

Target: token-identical. Any divergence is handled by the MUSTER probe-and-attribute
discipline (top-k logprob capture at the divergent position, nat gap, oracle-side
cross-backend control); widening tolerance or shrinking the pack is prohibited.

## 5. G3(b) — quality table

### 5.1 Primary (gated) metric: teacher-forced top-1 agreement vs the Q8_0 baseline

1. Capture the baseline: Camelid `Q8_0` row, greedy, all 9 prompts → 320-token continuations
   (committed to the Phase 3 bundle as token-ID sequences).
2. For each produced row R ∈ {NVFP4-mm, Q4K-mm, Q4_K_M-df, Q4_K_M-im} ([Amendment 2]:
   NVFP4-all is BLOCKED-HOST and excluded until produced on a capable host):
   teacher-force the baseline continuation through R (prompt + forced tokens fed one step at
   a time) and record R's argmax token at every continuation position.
3. `agreement(R)` = (positions where R's argmax equals the baseline's token) / 320, in
   percentage points to one decimal.

Harness note (pre-registered): this requires a small forced-decode mode in the Phase 3
harness — a decode loop that feeds a supplied token instead of sampling, and records
per-position argmax + logits. No engine math changes; the forward path is untouched.

### 5.2 GO rule (§6 threshold unchanged; comparator renamed per §1)

**GO iff `agreement(NVFP4-mm)` ≥ `agreement(Q4K-mm)` − 2.0 percentage points.**

Sanity guard: if `agreement(Q4K-mm)` < 80.0, the comparison basis is suspect (tool or
harness fault is likelier than a format fault) — STOP and escalate to Tim instead of applying
the rule in either direction. KL and perplexity are reported alongside and gate nothing in
v1; the report-only rows' agreement numbers likewise gate nothing.

### 5.3 KL (report-only)

Exact KL(baseline ‖ R), natural log, computed from **full logit vectors** at each of the same
320 teacher-forced positions (the forced-decode harness dumps full logits; vocab 262144 —
dumps are temporary, the bundle keeps per-position top-32 excerpts plus the mean/percentile
KL values). Reported: mean, median, p95, max, and the position/prompt of the max.

### 5.4 Perplexity (report-only)

Corpus: `qa/ornith/constrained-vram/heldout_coding.txt`, sha256
`460634c23b5a6ddeeaa325b4a461c44c569e753f4064a2af20290f36f35aaedf`, full file. Convention:
llama-perplexity semantics, `n_ctx` 2048, stride `n_ctx/2`, BOS once, second-half scoring.
Two instruments where possible: pin `llama-perplexity.exe` and Camelid's `src/quality`
Perplexity (pinned to the same convention). If the Camelid instrument cannot drive the gemma4
runtime path without engine changes, pin-side ppl alone is reported and the gap disclosed.
Rows: all five (Q8_0 baseline included; NVFP4-all excluded per Amendment 2, BLOCKED-HOST).

## 6. Execution hygiene (binding)

- Free-RAM check before every model load: model size + 3 GB free, else do not run. The Q8_0
  and BF16 legs (~8.2 / ~15 GB files) on this 15.7 GB box are the tight ones: single-engine,
  two-phase capture only (`--reference-out` / `--reference-in` pattern — engines never
  co-resident), ambient-use check first.
- Every scripted run: explicit timeout + kill-by-PID; after every leg, sweep for leftover
  `llama*`/`camelid` processes (kill by PID; never blanket taskkill — desktop sidecar).
- No cargo builds concurrent with any capture leg.
- Perf numbers, if any appear in Phase 3 outputs, are incidental and make no claims — perf
  belongs to Phase 4 with SIROCCO/WDDM hygiene.

## 7. Outputs

Phase 3 bundle `qa/evidence-bundles/basalt/phase3/` contains: quantize command lines + output
sha256s (from Phase 2), baseline continuation token IDs, per-row agreement/KL/ppl tables, the
G3 GO/NO-GO application with the rule quoted verbatim, parity receipts per §4, and the ×2
determinism spot-checks. The quality table is reported with all pre-registered rows even if
the answer is unflattering; a NO-GO terminates in a postmortem note, not threshold shopping.
