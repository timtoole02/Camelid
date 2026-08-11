# Conductor: MUSTER — Experimental-Lane Clearance (Windows/CUDA)
**Status: executable conductor. Recon gate first; code lands only behind gates.**
**Comparator pin: llama.cpp `acd79d603` (build 9632) — standing pin policy, do not chase upstream mid-mission. (Gemma-4-family rows pin `5d56eff`; none are expected in scope.)**
**Execution host: Tim's Windows/CUDA machine, Tim-authorized maintainer lane, clean public `main` checkout. Agent runtime: Claude Fable 5 in Claude Code, one phase per session unless a gate says otherwise.**
---
## 0. Mission
Every model row on this machine that renders the **`Experimental — unverified`** Evidence Chip must exit the experimental lane through one of exactly two doors:
1. **PROMOTED** — a supported exact-row contract entry, earned with the standard promotion evidence set (parity + API/WebUI smoke + synchronized contract/ledger/docs), sealed and pushed.
2. **HOLD** — a committed, named blocker receipt explaining precisely why the row cannot be promoted yet, with the experimental-lane docs updated to say so honestly.
There is no third door. **Relabeling is not promotion.** The chip is a symptom; the contract row is the cure.
### 0.1 What the chip actually is (verified)
- `frontend/src/lib/chatGate.js` — `experimentalUnlocked = runtimeReady && !contractSupported`. The chip renders whenever a model is loaded and generation-ready but has **no supported contract row** in `/api/capabilities`.
- `frontend/src/components/models/LaneRows.jsx:140` (`NotAnchoredRow`), `CatalogLaneBrowse.jsx:295`, `ActiveModelBar.jsx:13`, `MessageTurn.jsx:56` — the four chip surfaces.
- The contract source of truth is the static `CapabilitiesResponse` literal in `src/api/mod.rs` (per CAIRN Amendment 1). `ledger/camelid-ledger.json` is its derived canonical form via `scripts/extract-capabilities-to-ledger.mjs`; `scripts/check-ledger-drift.mjs` re-derives from code in CI (`.github/workflows/ci.yml:146-150`) and fails on drift.
Therefore: **the only legitimate way to clear the chip is to add a supported exact-row contract entry backed by evidence, then sync ledger + COMPATIBILITY.md + README.md + STATUS.md in the same commit.** Touching the chip logic, `chatGate.js`, or `EvidenceChip` states is FORBIDDEN in this campaign.
### 0.2 The bar, stated precisely
The promotion target per row is **`supported_exact_row_smoke`** — the identical bar used by the Mistral-7B promotion (head `d7b1699`), the Qwen3 dense rows, the Llama 3.2 3B K-quant rows, and the Ornith rows:
- exact file anchored (SHA-256, source repo, size);
- greedy token-AND-text parity vs the pinned oracle on the row's prompt pack at 1/5/50 tokens (near-tie policy in §6.3);
- API + WebUI promotion smoke green (`scripts/model-promotion-smoke-bundle.mjs` with expectations asserted);
- contract row + ledger + docs synchronized, evidence bundle sealed under `qa/evidence-bundles/`, committed and pushed.
**This campaign does NOT target `FULL_SUPPORT_BLOCKER_MATRIX.md` green.** Every promoted row lands with `full_support_status: blocked_pending_normalized_full_support`, exactly like the 24 existing `supported_exact_row_smoke` rows in the ledger. Full-support normalization (portability, production throughput, model-native context, durable current-head reruns) is a separate campaign and MUSTER must not borrow its language. If any doc, commit message, or LinkedIn draft produced during MUSTER says "fully supported," replace it with the exact contract status string.
### 0.3 Non-negotiable framing
The campaign objective is **"every roster row has a sealed verdict,"** not "every roster row is promoted." A NO-GO with a committed receipt is a successful phase outcome (precedent: the Ornith split-precision NO-GO receipt, `RECEIPT_ITEM5_acceptance_economics.json`; STAMPEDE Phase 6). An agent that cannot earn parity for a row files a HOLD receipt and moves on. An agent that edits an oracle, fixture, tolerance, or expectation to make a row pass has failed the campaign and the incident goes in `STATUS.md` (see §9, Forbidden Actions — this codebase has a scar here: the frontend smoke-test fixture inversion).
---
## 1. Scope
### 1.1 Wave A — repo-named experimental rows (in scope, fixed)
These two are the catalog entries the README (§ Experimental lanes) and the catalog literal (`src/api/mod.rs:17236`, `:17249`) name explicitly as having **no supported row**:
| MUSTER id | Catalog id | Exact file | Arch | Quant | Size | Prior evidence |
|---|---|---|---|---|---|---|
| **M-A1** | `gemma3_1b_it_q8_0` | `gemma-3-1b-it-Q8_0.gguf` (`ggml-org/gemma-3-1b-it-GGUF`) | `gemma3` | Q8_0 | 1,069,306,368 B | Runnable-lane greedy-parity receipt vs the HF reference: `qa/runnable/gemma3-parity.json` (SHA-anchored, `all_greedy_match=true`, 4 fixtures). Engine-parity fixture evidence only — **not** a support surface. |
| **M-A2** | `phi3_mini_4k_instruct_q8_0` | `Phi-3-mini-4k-instruct-Q8_0.gguf` (`bartowski/Phi-3-mini-4k-instruct-GGUF`) | `phi3` | Q8_0 | 4,061,222,688 B | **None.** No committed receipt of any kind. |
Both architectures are in `model.rs`'s implemented dense-decoder set AND in the runnable lane's `COVERED_ARCHITECTURES` (`src/runnable/admit.rs`), so two promotion vehicles exist per row (§4, Phase 1 decides which).
### 1.2 Wave B — live-discovered local rows (in scope, enumerated at Gate 0)
Any additional local GGUF on this machine that (a) admits/loads, (b) reports `generation_ready=true`, and (c) has no supported contract row — i.e., anything else actually showing the chip in the app. Typical candidates: neighboring quants of covered families (LLaMA/SPM Q4_0/Q5_0 files run "unverified in the experimental lane only" per the contract notes at `src/api/mod.rs:3566`), Ornith neighboring quants (Q6_K / IQ3_XXS / IQ4_XS / bf16), any HF-browse installs. **The Phase 0 roster artifact is authoritative — the agent must not promote, or write recon for, a row that is not on the committed roster.**
### 1.3 Explicitly OUT of scope
Do not touch these; each is a lane annotation on a row that already has a contract entry, or has its own campaign:
- Gemma 4 E4B-It **Q8_0 Windows-CUDA lane** (experimental lane of a supported row; first-token-argmax gate only) — its promotion needs a token-for-token CUDA bundle like the E2B one, a different mission.
- Gemma 4 E4B-It **Q4_0 mixed-QAT CUDA lane** (documented argmax-stable, not token-for-token — a known fp-reassociation frontier, not an evidence gap).
- Gemma 4 26B-A4B QAT **single-node SSER CUDA lane** (experimental annotation on the distributed-supported row).
- **DiffusionGemma 26B-A4B** (already "Supported (experimental) via the dedicated diffusion lane" — a designed label, not an evidence gap).
- **Mixtral-8x7B** (`active_validation_partial_runtime`; has its own blocker campaign).
- The `planned_exact_row_candidate` ledger rows (Qwen2.5 7B, Gemma 2 9B) **unless** their files are physically present and chip-visible at Gate 0, in which case they enter Wave B like anything else.
If Tim wants any of these folded in, he flips them in by editing this section — the agent never widens scope on its own.
---
## 2. Standing pins, hosts, and hygiene
- **Oracle:** llama.cpp `acd79d603` (build 9632), the standing pin. Build/locate `llama-server` from that exact commit; record binary SHA-256, build flags, and backend (CPU vs CUDA) in every manifest. If a row-specific pin doc exists (pattern: `REFERENCE_PIN_*.md`), it wins for that row.
- **Environment fingerprint** (Phase 0 artifact): `git rev-parse HEAD` (must be clean public `main`), `rustc -V` (floor 1.87+ per the checked-in toolchain files), `nvidia-smi` (GPU model, driver, VRAM), CUDA toolkit version, Windows build, `node -v`. All CUDA claims are GPU/driver/CUDA-version specific — the fingerprint is what scopes them.
- **Single-model runs, always.** The oracle is captured **alone** and its server killed before Camelid starts. Use the two-phase mode of `scripts/raw-decode-parity.mjs` (`--reference-out` to capture the oracle to a file and exit; `--reference-in` to compare live Camelid against the committed capture). Engines are never co-resident. This is the established discipline from the Qwen3 context bundles — keep it.
- **Artifacts:** raw runs under `target/muster-<row>-<UTC>-head-<sha12>/` (gitignored); scrubbed manifests + checksums published under `qa/evidence-bundles/<row>-support-promotion-<UTC>-head-<sha>/`. Before commit: `node scripts/audit-evidence-bundle-privacy.mjs` must report `finding_count: 0`, and `scripts/check-public-scrub.sh` must pass. No private host paths, home paths, or operator-only commands in anything tracked.
- **CI:** evidence-only or Windows-lane commits may use the `[ci-os: windows]` commit-message tag (`.github/workflows/ci.yml:24`) to run the Windows leg only; the `ci-gate` aggregator must still be green before any promotion commit merges. The ledger schema + drift checks (`ci.yml:146-150`) are the CAIRN tripwire — a promotion commit that fails them is by definition wrong.
- **Builds:** `cargo build --release --features cuda` for the GPU-capable binary; the deterministic CPU lane (`--deterministic` serve flag, per `RECEIPTS.md`) is the correctness reference on this host.
---
## 3. Phase 0 — Roster & seal (GATE 0: Tim signs the roster)
**No code. No downloads beyond what already exists locally.**
1. Build current `main`, start `serve`, and enumerate from the **live instance**: `/api/capabilities` (contract rows), the models page lane classification (Supported / Oracle-qualified / `NotAnchoredRow`), the pull catalog, and a directory scan of the models dir (filename, size, SHA-256 of every GGUF present).
2. Cross-check against `ledger/camelid-ledger.json` (26 rows at time of writing; 24 `supported_exact_row_smoke`-family, plus planned/active-validation rows) and the README Experimental-lanes table.
3. Emit `MUSTER_ROSTER.md`: one table — MUSTER id, exact filename, SHA-256, arch, quant, size, current lane, chip-visible (Y/N), prior evidence paths, wave (A/B), and a one-line initial read. Wave A rows M-A1/M-A2 appear even if not currently on disk (they are catalog-listed and chip-visible in catalog browse).
4. Emit the environment fingerprint (§2) into the same doc.
5. Commit `MUSTER_ROSTER.md` + this conductor. **STOP. Gate 0 = Tim reviews the roster and strikes or adds rows.** The struck roster is frozen for the campaign.
**Gate 0 exit criteria:** roster committed, fingerprint recorded, Tim's sign-off noted in `STATUS.md` under a `MUSTER` heading.
---
## 4. Phase 1 — Per-row recon (GATE 1: recon dossiers, no code)
For each roster row, a recon section in `MUSTER_RECON.md`, grounded in `file:line` citations (house style — see `RUNNABLE_LANE_RECON.md` as the template). Answer, per row:
1. **Vehicle:** which lane can carry the promotion on this host?
   - **Optimized lane** (`model.rs` dense-decoder path — `gemma3` and `phi3` are both in the implemented set): does the forward actually run this arch end-to-end today, and on which backends (CPU / `cuda_resident_q8` / VRAM+host offload)? What does the static execution plan report for the file?
   - **Runnable serve lane** (`CAMELID_RUNNABLE_SERVE=1`, precedent: all three Ornith promotions): is the arch runnable-served? What is the perf reality (pure-f32 lane)?
   - Pick ONE vehicle per row and defend it. CPU-lane promotion is fully legitimate for the support bar — CUDA residency is a stretch lever (§8), never a gate.
2. **Tokenizer:** family (SPM vs BPE per `admit.rs` groupings), whether the existing tokenizer path covers it, and whether any SPM merge-order caveat from `SPM_MERGE_ORDER_CONDUCTOR.md` applies to this row's gate prompts.
3. **Chat template:** does a renderer for this family exist in-tree, and is it byte/token-exact against the GGUF metadata template? (My clone-time grep found no gemma3/phi3-specific renderer — VERIFY, don't assume.) If missing: the promotion can be scoped as a **raw-decode** row (precedent: Llama 3.2 3B Q4_K_M/Q5_K_M promoted on `scripts/raw-decode-parity.mjs` with "per-quant API/WebUI/serve smoke remains a follow-up" — but note those rows had sibling Q8_0 rows carrying the chat surface; a first-of-family row promoted raw-decode-only must say so in its contract `tested_context` and README wording) OR the renderer gets built in Phase 3 with a template-shapes pack (precedent: Qwen3's hardcoded ChatML renderer + `qa/prompt-packs/` shape packs).
4. **Quant coverage:** eager vs lazy-wire dequant status for the row's tensor types (the lazy set is gated to formats with committed dequant-parity evidence — `src/tensor/wire_dequant.rs:47-51`).
5. **Fit:** VRAM/RSS estimate on the fingerprinted card; expected serving plan.
6. **Oracle plan:** llama.cpp `acd79d603` runs this arch? Which prompt pack (existing `qa/prompt-packs/` / `qa/speed/prompts.json`, or a new family pack committed BEFORE oracle capture)?
7. **Blockers:** anything that predicts HOLD, stated now.
**Gate 1 exit criteria:** recon committed; per-row vehicle decisions recorded; Tim skims and may re-order the waves. Rows whose recon already proves HOLD (e.g., oracle cannot run the arch at the pin) file their HOLD receipt here and exit the pipeline honestly.
---
## 5. Phase 2 — Acquisition & anchoring
Per row: fetch via `camelid pull <catalog-alias>` where a catalog entry exists (verify the alias — README shows `camelid pull gemma3_1b`; the catalog id is `gemma3_1b_it_q8_0`), else document the exact HF source. Record: repo id, filename, size (must match the catalog literal where one exists), SHA-256, license note (M-A1 is `gemma`-licensed — note any gating), download date. Emit `MUSTER_ACQUISITION.md` rows + per-file `sha256` sidecars in the run dir. **The SHA-256 recorded here is the one every later artifact must repeat verbatim — support is per exact file, and the Llama-3.2-3B canonical-vs-prior-upload split is the cautionary tale: evidence does not survive a re-upload.**
---
## 6. Phase 3 — Engine/renderer work (only what recon demanded) & Phase 4 — Parity certification
### 6.1 Phase 3 rules (code, gated)
- Only the work items the recon dossier named. Each lands with unit gates in-tree (pattern: the per-kernel bit-exact GEMV tests; the `#[ignore]`-gated GPU parity tests that need hardware).
- Any new renderer ships with a template-shapes pack committed under `qa/prompt-packs/` and a byte/token-exact gate before it is used in a parity run.
- Determinism sanity on the CPU lane before any oracle comparison: two back-to-back runs of the gate prompts must be byte-identical (the deterministic lane's reduction-order stability is what makes receipts meaningful — `RECEIPTS.md`).
- New family harness scripts (`scripts/chat-parity-gemma3.mjs`, `scripts/chat-parity-phi3.mjs`) are modeled on `scripts/chat-parity-qwen3.mjs`; raw-decode rows use `scripts/raw-decode-parity.mjs` unmodified.
### 6.2 Phase 4 — the certification run
Per row, in order, all inside one `target/muster-<row>-.../` run root:
1. **Oracle capture, alone:** pinned `llama-server`, exact GGUF, prompt pack, greedy (`temperature: 0`), capture via `--reference-out` (or the family harness's equivalent). Kill the oracle.
2. **Camelid runs:** the chosen vehicle/lane, same file, same pack; 1/5/50-token depths; `--reference-in` comparison. Prompt-token parity AND generated token+text parity both recorded.
3. **Receipts:** emit at least one sealed parity receipt on the deterministic CPU lane where the vehicle allows (`camelid verify-receipt` round-trip must PASS — the receipt is per-request evidence, NOT the promotion itself, per `RECEIPTS.md` rule 1; MUSTER uses it as an integrity spot-check inside the bundle).
4. **Verdict per pack:** `all_pass` → green. Divergences → §6.3. Anything else → HOLD receipt.
### 6.3 Near-tie policy (fixed; do not renegotiate per row)
A non-identical position is admissible ONLY if probed and attributed under the established cross-backend tolerance discipline (precedent: Ornith Q4_K_M — every flip probed to a ≤0.33-nat soft position where the oracle's own backends also flip; Llama-3.2-3B Q4_K_M — documented benign near-ties with logprob gaps stated). The bundle must contain the probe artifacts, the nat/logprob gap, and the oracle-side control. **"Looks coherent" is not attribution. Widening a tolerance, shrinking a pack, or swapping a prompt to convert FAIL→PASS is a campaign-failing act.** If the flips cannot be attributed, the row HOLDs with the divergence receipt committed as-is.
---
## 7. Phase 5 — API/WebUI promotion smoke, and Phase 6 — Contract/ledger/docs sync
### 7.1 Phase 5 (per row, after parity green)
```
node scripts/model-promotion-smoke-bundle.mjs `
  --model "<models-dir>\<exact-gguf>" `
  --model-id <row-id> `
  --out-dir "target/muster-<row>-<UTC>/api-webui" `
  --expect-compatibility-row <row-id> `
  --expect-compatibility-status "<exact contract status text>" `
  --expect-contract-supported true `
  --expect-webui-chat enabled
```
This asserts the whole loop: `/api/models/load`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, frontend readiness, and — the point of the campaign — **the chip flip**: the row must now render as supported, and a `NotAnchoredRow`/unverified turn for it must no longer be reachable. Capture stdout/stderr/summary in the bundle. (Rows promoted on the raw-decode precedent run the smoke to whatever surface their contract wording claims — the contract text and the smoke expectations must agree exactly.)
### 7.2 Phase 6 (one commit per row, atomic)
1. Add the contract row to the `CapabilitiesResponse` literal in `src/api/mod.rs` — copy the field discipline of the nearest precedent row (Mistral for a plain Q8_0 chat row; Llama-3.2-3B Q5_K_M for raw-decode; Ornith for runnable-serve). `full_support_status: blocked_pending_normalized_full_support`; `full_support_blockers` filled honestly; `frontend_readiness_gate` states the exact green-when condition including SHA prefix and any required env flags.
2. `node scripts/extract-capabilities-to-ledger.mjs` → `node scripts/check-ledger-schema.mjs` → `node scripts/check-ledger-drift.mjs`. All green locally before commit.
3. Same commit: README supported-models table row (+ delete/annotate its Experimental-lanes entry), `COMPATIBILITY.md` row (public claim / checked boundary / **Do not claim** column — write the non-claims first), `STATUS.md` MUSTER log entry, evidence-bundle manifest + checksums under `qa/evidence-bundles/`.
4. Privacy scrub (§2) green. Push. CI (`ci-gate`) green — the CAIRN drift job is the proof the surfaces agree.
**Wording rule for every touched surface:** support is the exact row only; nothing spreads to neighboring sizes, quants, templates, contexts, or the family. The "Do not claim" column is load-bearing — a promotion PR with a thin one is not done.
---
## 8. Phase 7 — Seal, verdicts, and stretch levers
1. `MUSTER_VERDICTS.md`: the roster table re-emitted with final verdict per row — **PROMOTED** (contract row id + bundle path + head) / **HOLD** (blocker receipt path + one-line cause) / **STRUCK** (Gate 0). Every roster row appears. This file is the campaign's definition of done.
2. `STATUS.md` closing entry: waves, dates, heads, and the one-paragraph honest summary (wins AND holds — the STAMPEDE Phase 6 register).
3. **Stretch levers — record, never gate:** bounded-context ladder buckets (512→8192, contiguous-ladder rule: hold non-contiguous passes like the Qwen3-4B-Q4_K_M 2048 near-tie precedent), CUDA-resident lane certification, `tool_capable` via a `camelid.agent_eval/v1` battery PASS on the exact file, decode tok/s (recorded, "NOT head-to-head" framing unless both engines ran the same backend on this host). Any stretch item that lands gets its own bundle and its own contract field update — never folded silently into the promotion claim.
---
## 9. Forbidden actions (campaign-failing)
1. Editing any oracle capture, fixture, prompt pack, tolerance, or smoke expectation **after** seeing a failing result, to make it pass. (The frontend fixture-inversion incident is why this is rule #1.)
2. Touching `chatGate.js`, `EvidenceChip`, `LaneRows.jsx`, or any readiness copy to change what renders, except the sanctioned effect of a new contract row.
3. Adding a contract/ledger/README/COMPATIBILITY row whose named evidence paths do not exist in the same commit.
4. Any fail-open construct in gating code (the `mem::replace(mode, Done)` class of bug from the PR #419 review). Gates fail closed.
5. Co-resident engine runs during capture; unpinned oracle builds; "same family" evidence inheritance; reusing prior-upload evidence for a different SHA.
6. Private host/home paths or operator-only commands in tracked files; skipping the privacy audit.
7. Widening scope beyond the frozen roster, or narrating a HOLD as anything other than a HOLD.
## 10. Agent-run mechanics
- One phase per Claude Code session; open each session by reading this conductor, `MUSTER_ROSTER.md`, and the previous phase's artifacts. Close each session by writing the `STATUS.md` MUSTER log line and the next session's entry point.
- Gates 0 and 1 end with **STOP for Tim**. Phases 2–7 run gate-to-gate per row; rows are independent after Gate 1 and may be executed in any order within a wave (Wave A before Wave B).
- Long CUDA/parity runs: prefer `[ci-os: windows]`-tagged commits for evidence-only pushes; never push a red `ci-gate`.
- If a session discovers a fact that contradicts this conductor (a script flag changed, a pin doc exists for the family, the catalog alias differs), the conductor is amended in the same commit as the discovery — the conductor is the contract, and a stale contract is a CAIRN violation in miniature.
---
## Amendment log (§10 — discoveries land here in the same commit)
- **A-1 (2026-07-15, Phase 0):** row-count correction. The live contract and ledger at `cd528cac` hold **26 rows: 20 `supported_exact_row_smoke` + 1 `supported_current_gate` (TinyLlama) + 1 `active_validation_partial_runtime` (Mixtral) + 2 `planned_exact_row_candidate` + 1 `planned_phase_10` + 1 `planned_beyond_named_certified_rows`**. §0.2's "24 existing `supported_exact_row_smoke` rows" and §3.2's "24 `supported_exact_row_smoke`-family" overcount. The bar, process, and scope are unchanged.
- **A-2 (2026-07-16, Phase 1):** §4.3's "clone-time grep found no gemma3/phi3-specific renderer" is half wrong: a phi3 renderer EXISTS in-tree (`render_phi3_prompt`, `src/api/mod.rs:12714-12730`, detector-routed before the tinyllama look-alike; gated today by exactly one unit literal, no shapes pack). gemma3 truly has none (its template falls to the non-exact role-colon fallback). M-A2's Phase 3 renderer work is therefore pack+gate retro-fit, not new renderer code.
- **A-3 (2026-07-16, Phase 1):** §6.2 harness reality. `scripts/raw-decode-parity.mjs --prompts-file` accepts ONLY a bare JSON array (strings or `{prompt}`/`{text}` objects) or `{prompts:[<strings>]}`; `qa/speed/prompts.json` (`{columns:[…]}`) crashes it — raw-decode rows derive an array-form pack (France BOS probe + the speed columns, the 3B-precedent set) and COMMIT it before oracle capture. Additionally, the script's `--stop`/`--variant`/`--proof-chain` defaults are Llama-3/K-quant-specific: every non-Llama-3/non-K-quant row MUST pass all three explicitly or the emitted parity JSON carries wrong-row labels and wrong stop-stripping.
- **A-4 (2026-07-16, Phase 1, no-change finding):** §7.1 verified valid as written — all four `--expect-*` flags exist in `scripts/model-promotion-smoke-bundle.mjs` and are implemented in `frontend/scripts/smoke.mjs`. Caveat recorded: the expectations assert only on the frontend leg (`--skip-frontend` would silently drop them; §7.1 does not use it).
- **A-5 (2026-07-16, Phase 2, stop-and-amend):** M-A2 size correction. The §1.1 Wave A table and the catalog literal baked **4,061,222,688 B** for `Phi-3-mini-4k-instruct-Q8_0.gguf`; the live HF tree (verified via lfs metadata) shows bartowski's only upload series (2024-04-29, never re-uploaded) is **4,061,221,376 B**, sha256 `0ac8ee48aeebf7d1b354691fd1e29e91c32ad88bbad10ad45ac880dcd4372a47`. The baked size matched no upload that ever existed. Catalog literal corrected in the same commit; the M-A2 anchor is the live-verified size+sha.
---
*MUSTER: assemble the whole herd, inspect every animal, brand only the ones that pass — and write down exactly why the rest didn't.*
