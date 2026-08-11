# Conductor: SPM Merge-Order Reconciliation (TinyLlama × Mistral × llama.cpp)

**Status: recon dossier + mission brief. No code lands with this document.**
**Comparator pin: llama.cpp `acd79d6` (build 9632) — per standing pin policy, we do not chase upstream mid-mission.**

## Problem statement

Camelid currently ships **three different SPM piece-tokenization algorithms** on the
SentencePiece (no-merges) lanes, none of which is llama.cpp's algorithm:

| path | algorithm | who hits it |
|---|---|---|
| `parse_special=true` segments (all chat prompts since `03039b7a`) | score-merge pair loop + greedy re-tokenization of unresolved runs + `▁▁` exclusion (`merge_spm_symbols_by_score` + `encode_spm_segment`) | TinyLlama marker gate, Mistral instruct gate |
| `parse_special=false` (raw completion text) | greedy longest-piece over the whole piece (`encode_piece_greedy`) | `bench-generate`, runnable-lane oracle prompts |
| rank-based BPE models | `bpe_registry.merge_symbols` | gemma4 family etc. — **out of scope here** |

The May-5 TinyLlama certification measured the greedy path (chat prompts did not
parse specials for SPM then). The May-8 Mistral fix built the score-merge path and
gated it on the Mistral pack only. The June-5 seam fix (`03039b7a`) correctly routed
SPM chat prompts through `parse_special=true` — and thereby swapped the algorithm
under the TinyLlama gate without re-running it. Result, measured 2026-07-01 on
pristine main `a323abb0` vs llama-server `acd79d6`:

- `trailing-spaces` prompt: Camelid tokenizes "camelid" as `[3949,295,333]`; llama.cpp
  (and May-4 Camelid, and the May-5 certified bundle) say `[2996,492,29881]`.
- `special-chars` prompt (🦙 emoji): same class — May-Camelid == today-llama.cpp,
  today-Camelid differs.
- `longer` prompt: Camelid generation is byte-stable May→today; the generated-text
  mismatch is upstream llama.cpp generation drift between the May-era Ubuntu pin and
  `acd79d6`. **Reference to be regenerated from the pinned build at re-cert; upstream
  drift recorded here as a note, not chased.**

Artifacts: `<home>\cam-attnblk\target\attn-blocked-dot-work\flag{off,on,off-main}\tinyllama-pack\`
plus the May bundle `qa/evidence-bundles/tinyllama-broader-template-context-perf-rss-20260505T044519Z-head-864e07b51f36/`.

## Reference semantics: llama.cpp SPM at `acd79d6` (normative for both gates)

`src/llama-vocab.cpp`, `llm_tokenizer_spm_session::tokenize` (lines 96–238):

1. Split the fragment into UTF-8 char symbols (doubly-linked list).
2. Seed a **priority queue** with every adjacent pair whose concatenation is a vocab
   token. Queue key: **higher token score first; on score ties, lower left symbol
   index first** (`llm_bigram_spm::comparator`, line 96–101: `l.score < r.score ||
   (l.score == r.score && l.left > r.left)`).
3. Pop repeatedly with **lazy invalidation** (skip if either side was merged or the
   recorded size no longer matches), merge right symbol into left, then push only the
   two new neighbor pairs `(prev,left)` and `(left,next)`.
4. Every successful `try_add_bigram` records `rev_merge[text] = (left,right)`.
5. Final pass `resegment`: for each surviving symbol, emit its vocab id if the text
   is a token; otherwise **recursively split via `rev_merge`**; if no rev_merge entry
   exists, emit **per-byte fallback tokens**.
6. There is **no `▁▁` special-casing anywhere** — multi-space pieces like `▁▁` are
   ordinary vocab tokens and merge normally.
7. Fragment boundaries: special-token partitioning happens upstream
   (`tokenizer_st_partition`); whitespace escaping and `add_space_prefix` are applied
   per fragment there, not inside the merge loop.

Selection rule 2+3 is equivalent to "repeatedly merge the currently-valid adjacent
pair with (max score, min left index)". The load-bearing differences from Camelid are
**not** the selection rule — they are items 4–6.

## What each Camelid commit actually changed (source-verified)

All four May-8 commits are the same-day Mistral campaign (`e1f0cc01` 11:13 →
`fa7efc08` → `5e989e61` 16:14 → `24e18106` 16:39); the June commit is the seam fix.

- **State 0 = certified May-4 head `864e07b5`**: for SPM models, `encode_piece`
  did **no control-token splitting at all** and ran `encode_piece_greedy` (longest
  vocab piece wins; ties → higher score) over the whole rendered prompt. `</s>` came
  out as token 2 only because greedy longest-match hit the `</s>` vocab entry as
  text. This is what the May-5 five-prompt certificate measured. It agreed with
  llama.cpp on those five prompts empirically — greedy longest-piece coincides with
  SPM bigram merges on common English, not by construction.
- **`e1f0cc01` (May 8): why Mistral required a change.** Mistral v0.3 instruct
  prompts need `[INST]`/`[/INST]` parsed as control tokens and llama.cpp's `▁`
  separator (id 29473) emitted after `[INST]` — the reference sequence is
  `[1, 3, 29473, …, 29473, 4]`. Whole-text greedy cannot produce that: it has no
  concept of control-token boundaries or dummy-prefix insertion. This commit
  introduced `parse_special` in `encode_piece`, control-token splitting for SPM, and
  `should_insert_dummy_after_control` (with an explicit `[INST]` special case).
- **`5e989e61` (May 8): first segment algorithm.** Once prompts were split at control
  tokens, the segments needed piece tokenization; without it, segments after control
  tokens fell to char-by-char unknown fallback (commit comment: "they can need a
  single vocab piece such as 'Hello'"). Implemented Viterbi best-path over vocab
  scores — which is unigram-SPM semantics, **not** llama.cpp's greedy bigram-merge —
  and introduced the `▁▁` exclusion hack.
- **`24e18106` (May 8, 25 min later): second segment algorithm.** Replaced Viterbi
  with `merge_spm_symbols_by_score` (pair-merge: max score, leftmost tie) plus
  greedy re-tokenization of unresolved runs, and added three broader Mistral prompt
  expectations including tokenizer stress words ("Fact alpaca MSTR mstr CMLD checksum
  gamma llama") — i.e. the Viterbi attempt demonstrably mismatched llama.cpp on rare
  words and was tuned until the **Mistral** pack passed. The **TinyLlama** pack was
  not re-run (its lane still bypassed this code entirely).
- **`03039b7a` (June 5): the seam flip.** `chat_prompt_parse_special()` (introduced in
  the qwen3 ChatML window) returned true only for BPE, so SPM chat prompts had
  regressed to encoding `</s>` as literal text `(829,29879,29958)`; this commit made
  chat prompts parse specials for every tokenizer model, verified on an 18-token
  TinyLlama prompt + 5 greedy tokens. Correct fix, insufficient verification breadth:
  it routed every SPM chat prompt into the May-8 segment algorithm, whose merges
  diverge from llama.cpp on exactly the word classes the 18-token smoke didn't
  contain ("camelid", 🦙).

## Exact behavioral deltas of current main vs llama.cpp `acd79d6`

1. **Unresolved-run handling.** llama.cpp resegments a non-vocab symbol by recursing
   into its own recorded merge tree (`rev_merge`), or falls back per byte. Camelid
   concatenates ALL adjacent unresolved symbols into one string and re-tokenizes it
   with `encode_piece_greedy` — a different algorithm that can merge across llama.cpp's
   byte-fallback boundaries and picks longest-piece instead of merge-tree pieces.
2. **`▁▁` exclusion.** Camelid refuses any candidate containing `▁▁` in both the
   merge loop and greedy; llama.cpp merges multi-space pieces normally (`▁▁` is vocab
   id 259 in llama-family vocabs). Every prompt with ≥2 consecutive spaces diverges
   structurally — note this becomes live the moment PR #356's harness fix makes the
   trailing-spaces prompt actually carry its spaces.
3. **Merge provenance.** Camelid's post-merge emission checks `token_to_id` per final
   symbol; llama.cpp emits the id the merge already established. Combined with (1),
   Camelid's output is not guaranteed to equal its own merge tree.
4. **Dummy-prefix seam.** Camelid inserts `▁` after control tokens via
   `should_insert_dummy_after_control` heuristics ([INST] special-case, `rest`
   inspection); llama.cpp derives the equivalent from fragment-level whitespace
   escaping + `add_space_prefix` in `tokenizer_st_partition`. These agree on the
   gated Mistral prompts today; the equivalence has never been audited generally.
5. **Raw-text path (latent).** `parse_special=false` SPM text still uses whole-piece
   greedy — a third algorithm. llama.cpp uses the same bigram-merge for raw text.
   The runnable-lane HF-bit-parity receipts for (llama, Q8_0, SPM) were earned on
   this greedy path; any unification must not invalidate them silently.

## Mission (for the eventual fix — recon deliverable ends here)

**Verdict on the central question: one algorithm satisfies both lanes.** llama.cpp
passes both the TinyLlama marker gate and the Mistral instruct gate with a single
SPM implementation; the strongest evidence possible that per-family tokenizer lanes
are unnecessary. Camelid should port `llm_tokenizer_spm` exactly — bigram priority
queue (score desc, left-index asc, lazy invalidation), `rev_merge` resegmentation,
per-byte fallback, no `▁▁` exclusion — behind the existing control-token seam.

Phases for the implementation mission (all gated, no reverts in the interim):

- **P0 — oracle harness.** Byte-level differential tokenizer fuzz: Camelid `encode`
  vs `llama-tokenize` (pinned `acd79d6`) over (a) both certified packs, (b) a
  generated corpus hitting rare words, emoji, byte-fallback, multi-space runs,
  control-token adjacency, and (c) the runnable-lane oracle prompts. Record the
  failing corpus BEFORE changing code — it is the acceptance baseline.
- **P1 — exact port.** Implement llama.cpp's queue+rev_merge algorithm as THE SPM
  segment tokenizer for `parse_special=true`. Kill the `▁▁` exclusion and the
  greedy-unresolved fallback on this path.
- **P2 — seam audit.** Prove `should_insert_dummy_after_control` reproduces
  llama.cpp's fragment semantics on the fuzz corpus, or replace it with
  partition-level whitespace handling.
- **P3 — raw-path decision.** Either unify `parse_special=false` onto the same
  algorithm (then re-earn the runnable-lane HF parity receipts) or explicitly
  document the greedy raw path as a separate contract. Do not change it silently.
- **P4 — acceptance.** BOTH gates on the pinned comparator: TinyLlama five-prompt
  50-token pack (with PR #356's harness fix, trailing-spaces expectations
  regenerated from the pinned build) AND the Mistral reference pack
  (`encodes_mistral_real_prompts_like_llama_cpp_when_available` + the broader pack).
  Then the re-certification order from the 2026-07-01 rulings: re-certify flag-off →
  rerun blocked-dot flag-on parity → promotion decision on
  `BACKENDINFERENCE_ATTENTION_F32_BLOCKED_DOT`.

## Non-goals

- No reverts of `e1f0cc01`/`5e989e61`/`24e18106`/`03039b7a` (Mistral stays green).
- No comparator bump; `acd79d6` stays pinned.
- Rank-based BPE lanes (gemma4/qwen) untouched.
- No README/COMPATIBILITY/STATUS claims until P4 receipts exist.
