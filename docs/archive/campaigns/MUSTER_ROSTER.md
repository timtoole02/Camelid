# MUSTER Roster — Phase 0 (Gate 0)

Campaign: [`MUSTER_CONDUCTOR.md`](./MUSTER_CONDUCTOR.md). Phase 0 enumeration performed live against a fresh `--features cuda` release build of clean public `main` @ `cd528cacf6d707e3678155e53804cac7b5f2cfa3` (2026-07-15).

**Status: awaiting Gate 0 — Tim reviews, strikes or adds rows; the struck roster is then frozen for the campaign.**

## Environment fingerprint (§2)

| Axis | Value |
|---|---|
| Repo head | `cd528cac` (`Merge pull request #455 from karan68/feat/iquant-iq4xs`), branch `main`, tracked tree clean |
| rustc / cargo | 1.95.0 / 1.95.0 |
| node | v24.16.0 |
| CUDA toolkit | 12.9 (V12.9.86) |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, driver 576.83, compute 8.6, tensor cores |
| CPU / RAM | 16 logical cores, AVX2+AVX512F+FMA; 15.7 GiB RAM |
| OS | Windows 11 Home Insider Preview, build 26220 |
| Oracle pin | llama.cpp `acd79d603` (build 9632) — binary SHA-256/build flags/backend recorded per capture manifest (§2) |

## Enumeration sources (all live at head `cd528cac`)

- `GET /api/capabilities` — **26** `model_compatibility` rows, of which **21** supported-family (`supported_current_gate` ×1 TinyLlama + `supported_exact_row_smoke` ×20); **zero** rows match `gemma3` or `phi3`. Matches `ledger/camelid-ledger.json` (26 `model_rows`) exactly.
- `GET /api/models/local` — per-file `admitted` / `admission_reason` / `oracle_qualified` / `runnable_receipt_present` / `lane_class` for all 15 on-disk GGUFs (static scan; **no model loads performed this phase**).
- `GET /api/models/catalog` — 15 curated entries, `next_cursor: null`; both Wave A entries present (`oracle_qualified: true`).
- Models-directory scan — 15 GGUFs, size + SHA-256 each (full hashes in §Full SHA-256 anchors).
- Cross-checks: README `### Experimental lanes` (README.md:207-217), `COMPATIBILITY.md` at-a-glance table, contract literal `src/api/mod.rs:2873-3998`, catalog literal `src/api/mod.rs:17062-17262`.

Chip-surface legend (per `frontend/src/lib/chatGate.js:15-16`, `frontend/src/lib/modelLanes.js:22-27`): **chat chip** = `Experimental — unverified` renders on every turn once the file is loaded + generation-ready with no supported contract row; **page chip** = `NotAnchoredRow` (`LaneRows.jsx:140`) renders in the Models list regardless of load state; **catalog chip** = predicted-lane chip in Get-models browse (`CatalogLaneBrowse.jsx:38-42`).

## Roster (proposed — pending Gate 0 strikes)

| MUSTER id | Exact file | SHA-256 | Arch | Quant | Size (B) | Current lane | Chip-visible | Prior evidence | Wave | Initial read |
|---|---|---|---|---|---|---|---|---|---|---|
| **M-A1** | `gemma-3-1b-it-Q8_0.gguf` | `b205840c…` | gemma3 | Q8_0 | 1,069,306,368 | `experimental_implemented`; Models page **Oracle-qualified** (admitted, `oracle_qualified=true`, no cached receipt) | **Y** — chat chip on load; catalog chip `Experimental · runnable` | `qa/runnable/gemma3-parity.json` (HF-reference greedy parity, 4 fixtures, `all_greedy_match=true`, SHA-anchored to this exact file — verified this phase) | A | Strongest candidate: on disk, size matches catalog literal, oracle-qualified runnable combo (gemma3/Q8_0/SPM); Phase 1 decides runnable-serve vs optimized-lane vehicle; SPM merge-order caveat must be checked for gate prompts. |
| **M-A2** | `Phi-3-mini-4k-instruct-Q8_0.gguf` | — (not on disk; anchor at Phase 2) | phi3 | Q8_0 | 4,061,222,688 (catalog literal) | Catalog-only (`bartowski/Phi-3-mini-4k-instruct-GGUF`, curated, `oracle_qualified=true`, fit advisor: `wont_fit` on current free RAM) | **Y** — catalog chip; chat chip expected on load | **None** | A | Needs ~4 GB pull; phi3 in both lanes incl. fused-QKV expansion (`model.rs:1697-1714`) and oracle-qualified combo (phi3/Q8_0/SPM); chat-template renderer existence is the Phase 1 question; fit advisor abstains on GPU — CPU-lane promotion legitimate. |
| **M-B1** | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | `6a746610…` | llama (BPE) | Q4_K_M | 807,693,984 | `experimental_implemented`; Models page **NotAnchoredRow** (admitted, not oracle-qualified — llama/BPE not a smoke combo) | **Y — verified live this phase**: serve auto-restored this file to generation-ready during enumeration (`/v1/models` showed it loaded) | None for this file (siblings: 1B Q8_0 + 3B Q4_K_M/Q5_K_M + 1B IQ4_XS all supported rows) | B | Prime candidate on the Llama-3.2-3B K-quant raw-decode precedent (`scripts/raw-decode-parity.mjs` unmodified, GPU-resident); sibling Q8_0 row carries the chat surface; frontend artifact gate keeps rows cleanly separated. |
| **M-B2** | `ornith-1.0-9b-Q6_K.gguf` | `33b6f6a3…` | qwen35 | Q6_K | 7,359,259,072 | `experimental_implemented`; **NotAnchoredRow** (admitted, not oracle-qualified) | Y — page chip now; chat chip expected on load (not load-verified this phase) | None (siblings Q8_0/Q4_K_M/Q3_K_M supported) | B | Runnable-serve vehicle per sibling precedent; 7.4 GB exceeds 6 GB VRAM → CPU lane or VRAM+host split (Q4_K_M at 5.6 GB was the resident ceiling); COMPATIBILITY.md:35 currently names Q6_K in the Ornith do-not-claim — promotion would move that boundary. |
| **M-B3** | `ornith-1.0-9b-IQ4_XS.gguf` | `0e267369…` | qwen35 | IQ4_XS | 5,196,440,096 | `experimental_implemented`; **NotAnchoredRow** (admitted — IQ4_XS newly covered by #455) | Y — page chip now; chat chip expected on load (not load-verified this phase) | None for this file; IQ4_XS decode/kernel infrastructure from PR #455 (Llama-3.2-1B row) | B | qwen35×IQ4_XS composition unproven; #455's unquantified near-tie systematic offset caveat applies; candidate for runnable/CUDA lane per sibling precedent. |
| **M-B4** | `qwen2.5-0.5b-instruct-q4_0.gguf` | `7671c0c3…` | qwen2 | Q4_0 | 428,730,208 | `experimental_implemented`; **NotAnchoredRow** (admitted, not oracle-qualified) | Y — page chip now; chat chip expected on load (not load-verified this phase) | None | B | Smallest candidate; qwen2 in both lanes; ChatML template likely reusable from the Qwen3 renderer (verify byte-exactness vs GGUF metadata); COMPATIBILITY.md:79 locked Qwen2.5-family readiness wording constrains promotion language (row is 0.5B, distinct from the planned 7B row). |
| **M-B5** | `ornith-1.0-9b-bf16.gguf` | `27bc7534…` | qwen35 | BF16 | 17,920,696,512 | `experimental_implemented` but **`admitted=false`** — runnable lane rejects `unsupported quant BF16 in tensor output.weight`; `chat_capable=true` via optimized lane (unproven) | Page chip Y; chat chip **unknown** — no load attempted; 17.9 GB is over-RAM (15.7 GiB total) | None | B | Expected HOLD: runnable vehicle rejects BF16; optimized-lane qwen35 end-to-end is unproven; over-RAM load on this host is WRAITH territory. Propose Tim either strikes it or accepts it as an expected-HOLD row. |
| **M-B6** | `ornith-1.0-9b-IQ3_XXS.gguf` | `0be488ed…` | (unparseable) | IQ3_XXS | 3,938,165,280 | `lane_class=unsupported` — GGUF parse fails: `tensor token_embd.weight has unknown or removed GGML type Unknown(21)`; `chat_capable=false` | Page chip Y (NotAnchoredRow renders for any unanchored scan entry); chat chip **N — cannot load, ever, at this head** | None | B | Fails Wave B criteria (a)/(b) — can never be generation-ready, so the chat chip is unreachable. Propose STRUCK at Gate 0, or keep as an instant HOLD receipt ("IQ3_XXS wire type unimplemented"). |

## Full SHA-256 anchors (models-directory scan, 2026-07-15)

Roster rows:

- `gemma-3-1b-it-Q8_0.gguf` — `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5` (matches `qa/runnable/gemma3-parity.json` `gguf_sha256` exactly)
- `Llama-3.2-1B-Instruct-Q4_K_M.gguf` — `6a74661014a3e2f139871f81e6cec852c489a627d169de503a3c0434a10c503d`
- `ornith-1.0-9b-Q6_K.gguf` — `33b6f6a3e3f05078438e12df8a4b55c8acf78ceadcc639d2af1cf35a026e8387`
- `ornith-1.0-9b-IQ4_XS.gguf` — `0e267369ffbbfcdbdc50241db62a865942a155fb5dfa041f7e8518949b5df7b9`
- `qwen2.5-0.5b-instruct-q4_0.gguf` — `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`
- `ornith-1.0-9b-bf16.gguf` — `27bc753487eed85539c3aef63dd602b79cd060401b928c9ff7d30d5556eca260`
- `ornith-1.0-9b-IQ3_XXS.gguf` — `0be488edfb63d7bd112a15b947d8a24fcb184a6678cddb6cc23d0e828b9aa7ae`

Non-roster on-disk files, mapped to existing supported rows (SHA cross-checked against ledger prefixes where the ledger carries them):

- `Llama-3.2-1B-Instruct-Q8_0.gguf` — `3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1` → `llama32_1b_instruct_q8_0` (artifact-gated basename match)
- `Llama-3.2-3B-Instruct-Q8_0.gguf` — `f34112a11b7dad74ab517dedf6dcf00d624c9adac2dc0c72c719ca0478554ef2` → `llama32_3b_instruct_q8_0` (**canonical** `f34112a1…` ✓)
- `Qwen3-0.6B-Q8_0.gguf` — `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` → `qwen3_0_6b_instruct_q8_0`
- `Qwen3-4B-Q8_0.gguf` — `8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300` → `qwen3_4b_instruct_q8_0`
- `ornith-1.0-9b-Q8_0.gguf` — `d0e4bebaa8b3450c62090df1408f2ee5ccb2094f9c610ffde564a654483d4f37` → `Ornith 1.0 9B`
- `ornith-1.0-9b-Q4_K_M.gguf` — `2711bf1ef034fa39eb899f793fe63bbb0aac21ebdacbcbe09406b5600ad5188f` → `ornith_1_0_9b_q4_k_m` (ledger prefix `2711bf1e` ✓)
- `ornith-1.0-9b-Q3_K_M.gguf` — `16f54df50e44bcaed854941835e595e60a12db48d3b2248af2a1959fc91b6eaa` → `ornith_1_0_9b_q3_k_m` (ledger prefix `16f54df5` ✓)
- `gemma-4-26B_q4_0-it.gguf` — `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5` → `gemma4_26b_a4b_it_q4_0` (out of scope, §1.3)

## Out-of-scope confirmations (§1.3)

- `gemma-4-26B_q4_0-it.gguf` maps to the supported distributed row; its single-node SSER CUDA lane annotation stays out of MUSTER.
- The `planned_exact_row_candidate` files (`Qwen2.5-7B-Instruct-Q8_0.gguf`, `gemma-2-9b-it-Q8_0.gguf`) are **not** physically present → they stay out per §1.3. (M-B4 is a different file/row of the Qwen2.5 family and enters Wave B on its own merits.)
- README Experimental-lanes entries for DiffusionGemma and the Gemma 4 E4B Q4_0 CUDA lane are designed labels on supported rows — untouched.

## Observations recorded for Tim (no MUSTER action taken)

1. **IQ4_XS docs drift — RESOLVED post-roster:** `llama3_2_1b_instruct_iq4_xs` (`supported_exact_row_smoke`, PR #455) was present in the contract + ledger but absent from the README supported-models table, the README quant-tier table, and the COMPATIBILITY.md at-a-glance table at roster time. Fixed on `main` via PR #457 (merge `e18b756d`, 2026-07-15): all doc surfaces plus the contract's missing `supported_quantization` IQ4_XS item, ledger regenerated. Recorded here because the roster's original observation predates the fix.
2. **Catalog/evidence wording mismatch:** the `llama32_1b_instruct_q8_0` catalog entry pulls from `unsloth/Llama-3.2-1B-Instruct-GGUF` (src/api/mod.rs:17064-17076) while the contract row's evidence text says "the exact bartowski … GGUF" (src/api/mod.rs:3097). Filename matches; provenance wording doesn't.
3. **Serve auto-restore makes the chip live at boot:** `serve` restored the previous session's model (`Llama-3.2-1B-Instruct-Q4_K_M.gguf`, an unanchored row) to generation-ready with no operator action — which is how M-B1's chat chip was verified live. Behavior noted, not judged.
4. **Ledger provenance head is stale by design:** `ledger/camelid-ledger.json` provenance records `source_head e5932cce`; the drift check excludes provenance per its own note.
5. **Pre-existing privacy-audit findings — RESOLVED post-roster:** at roster time `node scripts/audit-evidence-bundle-privacy.mjs` reported `finding_count: 3`, all in the June stream-options bundle's `phase3/proxy.log` (operator home paths in one logged request body). Fixed via PR #458 (merge `9c48810f`, 2026-07-15): the log — which turned out to be untracked/gitignored, so the leak was local-disk only, never public — was redacted in place with a tracked `REDACTION_NOTE.md`; the audit now reports `finding_count: 0` repo-wide. MUSTER bundles must still land at `finding_count: 0` for their own files (§2).

## Gate 0 sign-off

- [ ] Tim: strike/add rows above (M-B5 and M-B6 have proposed dispositions inline). On sign-off, the struck roster is frozen; the sign-off is recorded in `STATUS.md` under the MUSTER heading, and Phase 1 (per-row recon, `MUSTER_RECON.md`) begins.
