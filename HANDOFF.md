# Handoff — speculative decoding campaign

Branch `campaign/mtp-3x`. **Unpushed.**
Worktree `/Volumes/Untitled/Camelid-3x`. The EAGLE benchmark and experimental
v2/MMA K-quant lanes remain opt-in. Phase 0's SPEC_GPU helper does auto-arm on
eligible Metal speculative requests unless explicitly disabled.

## Latest result: Llama 3.2 3B on mini2

The 100 tok/s stretch target was cleared, then doubled:

| workload | context | tok/s | acceptance | verdict |
| --- | ---: | ---: | ---: | --- |
| recurring four-token structure, width 16 (3-run mean) | 115 | **201.67** | 100% | lossless |
| same structure, width 16 + packed attention | 1,819 | **161.58** | 100% | lossless |
| favorable raw, linear learned EAGLE | 46 | **56.09** | 82.72% | lossless |
| mixed JSONL suffix + learned-tree fallback | 485 | **52.88** | 36.36% offered | lossless |
| agentic learned-tree pack | 1,078 | **42.47** | 29.91% offered | lossless |

The 201/161 rows are an honestly labelled recurrence ceiling, not general chat:
the model-free suffix lane emits 15.83 target-verified tokens per round. The
short result is stable across three runs (`201.62..201.70`) with identical
plain/spec streams. The 1,819-token run traced every verifier layer through the
new row-dimensional attention path and reproduced its pre-batch plain/spec/draft
arrays exactly.

The three changes are lazy EAGLE-head maintenance across suffix streaks
(`b59a1596`), packed F16 verifier attention (`794c6a55`, extended to k=16 by
`ee33ea68`), and a two-SIMDgroup V4 K-quant width-16 kernel (`8201475d`). Width
16 was not a learned-tree win in the unpromoted notebook control (37.31 tok/s);
keep the receipted N8/K4/X4 fallback and use N16 only for high-confidence suffix
chains.

The 201.67 ceiling and raw/mixed/agentic rows use source `aeeacfa1`, version
`v0.6.1-308-gaeeacfa1`, binary sha256
`e509fc61d602a7a467ffe44b0b0d97d3dc1e779034b2f37e2a4566566cd990a0`.
The deepest 1,819-token row uses source `ee33ea68`, version
`v0.6.1-309-gee33ea68`, sha256
`9e34084423e16c212746dc9666a151c860ed378a650825494bf81d06efa6bb4d`.
Safety behavior introduced by `a7f8db6d` prohibits wide V4 in the standard
resident Llama prefill path exercised by this benchmark, and k=16 packed
attention fails closed above position 2,048. The prefill guard is not a global
claim about unrelated windowed or architecture-specific surfaces.

The first 4k width-16 stress attempt produced no receipt; mini2 subsequently
reported `This system is locked` to SSH, consistent with a reboot/local-login
lock. Do not rerun it. Locally unlock and inspect mini2 first, then copy the new
receipts from `~/camelid-overnight/receipts/`. Full method, hashes, receipt names,
and scope: `qa/speculative/LLAMA32_3B_MINI2_WILD.md`.

This is a learned EAGLE sidecar plus model-free suffix decoding, **not native
MTP**. No usable pinned Llama-3.2-3B native MTP/Medusa weights were found.

## Original 8B result

Llama-3-8B **Q4_K_M**, 4137-token agentic prompt, one machine (mini2), one prompt;
each row differs from the one above by a single change.

| | plain | speculative | verify round |
| --- | --- | --- | --- |
| default lane (origin/main) | 7.13 | 4.72 | 1528 ms |
| + K-quant v2 / MMA verify | 8.22 | 9.55 | 389 ms |
| + suffix drafter on the chain | 8.22 | 11.41 | 621 ms |
| **+ kv16 split-K attention** | **13.51** | **30.68** | **215 ms** |

**7.13 -> 30.68 tok/s = 4.30x.** Speculation went from a 0.66x *regression* to
2.27x. Plain decode alone went **1.89x** on this exact 8B model, prompt, and
machine. The routing fix structurally applies to K-quant decode whether it
speculates or not, but that measured factor is not a cross-model claim. All arms
`lossless=true`.

## How to reproduce

```bash
BIN=<release camelid>
env CAMELID_METAL_RESIDENT_DECODE=1 CAMELID_METAL_RESIDENT_PREFILL=1 \
    CAMELID_METAL_WIRE=1 CAMELID_METAL_WIRE_NSG8=1 CAMELID_METAL_F32Y=1 \
    CAMELID_METAL_NOCOPY=1 CAMELID_METAL_KQUANT=1 CAMELID_SPEC_TREE=0 \
    CAMELID_KQUANT_V2=1 CAMELID_KQUANT_MMA=1 \
  "$BIN" bench-speculative Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
    --drafter suffix --draft-tokens 7 --prompt-file qa/speculative/harness/prompts/agentic_4k.txt \
    --max-tokens 96 --warmup
```

Harness and prompts: `qa/speculative/harness/`. Raw records: `qa/speculative/receipts/`.
Narrative and method: `qa/speculative/PHASE0.md`.

## The three changes that matter

1. **kv16 primaries admitted to split-K attention** (`src/metal.rs`,
   `encode_attention`). The gate read `v2 && !kv16`, and every K-quant model gets
   an F16 KV primary, so every K-quant model fell through to a slow attention
   kernel. The stated reason (a kv16 primary keeps no mirrors) was wrong:
   `kv_scatter_kv16` and `kv_scatter_f32` write the same `dst` with the same
   `half(src)`, so the primary already IS the mirror. **This single edit is both
   the 4.30x and the losslessness fix.**
2. **`SpeculativeDrafter::Suffix`** — suffix drafting flattened to a chain so it
   rides the batched verify instead of the 13x-more-expensive tree path.
   `CAMELID_SPEC_DECODE=suffix`, or `--drafter suffix`.
3. **Phase 0 spine** — SPEC_GPU auto-arm, streaming speculation across all three
   decode loops, greedy-only admission. Without these the lane never fires on
   real traffic.

## State of each thread

**Exact measured speed lane — DONE.** Committed, receipted, and gated; no broad
support claim follows from the single-model result.

**v1 K-quant losslessness — FIXED.** The default lane emitted a different token
stream than its own plain decode, deterministically at index 58 (5/5 both ways on
a same-binary A/B). The kv16 split-K admission fixed it.

**Llama-3.1 promotion — parity ANSWERED, packs NOT RUN.**
`qa/model-qualification/llama3_1_8b_q8_0-parity-tolerance-decision.md` has the
measurement: against the exact pinned oracle (`acd79d603` = b9632) on the
sha256-verified artifact, the top-10 token set is identical and max disagreement
is 0.071 nats, against a 0.031-0.077 nat margin. It is a tie, not a bug (contrast
qwen35: rank 70, 8.4-19.3 nats). **Remaining for promotion:** bounded-context 512
pack, API/WebUI load receipts, and sign-off on the proposed gate (top-10 set
equality, max |delta| <= 0.15 nats).

**Historical Llama 3.2 3B EAGLE-3 baseline — superseded by the latest result
above; still not native MTP or serving integration.** The learned EAGLE-3
benchmark is pinned to the exact target
`Llama-3.2-3B-Instruct-Q4_K_M.gguf` (sha256
`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`,
2,019,377,696 bytes) and the matching Thoughtworks EAGLE3 head at revision
`02d343789b502a3edfe351bdd4537a44affb98cd` (`model.safetensors` sha256
`c0713251464a9b6b5fcf9fb229587bbe59b6fd1521027aef32101d11b9ebbdaf`,
486,297,280 bytes). It captures resident target taps `[2, 14, 25]`, keeps both
the learned head and target resident on Metal, and enforces losslessness and
head/target cache-watermark gates.

The pre-tree/raw-prompt baseline at gamma 3 was
**30.681 -> 41.536 tok/s = 1.354x**, with **87.18% offered-draft acceptance**,
**3.615 emitted tokens/round**, and **26 resident / 0 CPU verify rounds** over
96 generated tokens. Source is `9626148a`; the deployed binary is sha256
`dac2fb1cc83b2c63f2926b082ba8a6639c3da1804fa493523f26f024fd2be411`.
The K/V-only authoritative-row optimization passed a full A/B: its draft ids,
EAGLE ids, and plain ids exactly match the full-compute control, while throughput
rose from 36.830 to 41.536 tok/s.

Do not generalize that raw row. The new `--chat` mode uses the same no-tools
renderer as `/v1/chat/completions`; its best arm is gamma 1 at
**30.649 -> 26.085 tok/s = 0.851x**, despite a plausible 59.32% first-draft
acceptance. On the final provenance build (`758b565f`, binary SHA-256
`616b3094b242d3bdf4c2f55ca68312d86e96f88223433fc58b86e24523c728cc`),
the 556-token code context measured EAGLE `0.877x` and suffix `0.915x`. At the
original 4,137-token depth, linear gamma-3 EAGLE is effectively break-even to a
small loss (**24.077 -> 23.014 tok/s = 0.956x**, 55.14% acceptance), while
suffix gamma 7 reaches **22.015 -> 43.910 tok/s = 1.995x** with 84.62%
acceptance. All four streams are lossless and fully resident, with full
binary/model/prompt/token-array provenance. So EAGLE is functionally correct
and can win on the exact short raw completion, but linear top-1 is not a broad
chat accelerator. That result motivated the dynamic-tree, lazy-head, packed
attention, and width-16 campaign summarized at the top of this handoff.

The exact matching head declares training sequence length **1024**; it does not
declare a runtime cap. The old Llama 3.1 / 2048 account was about a different
artifact and must not be carried forward. This remains a benchmark-only EAGLE-3
path, not a native MTP implementation and not integrated into serving. Full
method and receipts:
[`qa/speculative/EAGLE3_LLAMA32_3B.md`](qa/speculative/EAGLE3_LLAMA32_3B.md).

## Next lever for speed

For the 3B hybrid, the remaining general-output limiter is acceptance, not
verifier width. The unpromoted N16 notebook control emitted only 3.80
tokens/round and fell to 37.31 tok/s; N8/K4/X4 is the receipted fallback. A
genuinely better matching head or
native target MTP weights are needed to lift ordinary chat. For recurrence,
decode is already 201.67 tok/s short / 161.58 at 1,819 tokens; prompt prefill,
not decode, dominates request throughput at depth.

Plain V4 decode is still roughly 22 tok/s because the structurally identical
single-token path pads to eight columns. The established V3 plain lane is around
41.5 tok/s. A faster V4-compatible single-token kernel is the clean next kernel
lever, but it must preserve the V4 arithmetic universe used by wide verification.

The following attribution is the earlier 8B campaign context:

Re-attribution after the fix: attention is now **1.9 ms/verify-column, down from
52.8**; rope, scatter and argmax measure zero. The remaining ~25 ms/col is inside
the **K-quant MMA GEMV**. Separately, plain decode sits at **55% of the ~120 GB/s
wall** (66 GB/s), and the mc-gemv campaign's own microbench shows even a
math-free weight-stream probe reaching only 89 GB/s at these shapes — so roughly
1.6x is unclaimed in the decode kernel itself. Use
`CAMELID_VERIFY_ABLATE=<stage>` (measurement-only, outputs garbage) to attribute.

**2026-08-29 mini2 follow-up:** the obvious store/barrier/reload removal was a
real microbenchmark win (Q4 -6.7%, Q6 -28.0%) but failed the warmed 4k lossless
gate at output token 0 or 11, including after stronger barriers and after
isolating Q4. It was fully reverted. Half-staging Q6 regressed; paired packed
loads were noise. Do not repeat the direct-fragment route from the synthetic
result alone. The now-preserved live benchmark is in `qa/speculative/kbench/`;
full receipts and the rejection analysis are in
`qa/speculative/KQUANT_MMA_OVERNIGHT.md`.

## Traps worth knowing

- `metal_spec_verify_bit_identical` **skips silently** unless the fast-stack
  gates are armed, so it can go green without touching the intended K-quant
  path. It now prints `kv_format`; run it with `CAMELID_METAL_KV_DTYPE=f16`,
  and use `metal_tree_verify_kv16_primary_path_runs` for the full primary/tree
  and packed-attention counters. Filters go AFTER `--`.
- A K-quant micro/unit identity pass is still not a promotion gate. The direct
  fragment candidate passed both but diverged only in the warmed 4137-token
  model run. Require two warmed 96-token exact runs after those fast gates.
- The old “tree verify is structurally dead; keep `!kv16`” conclusion is now
  obsolete. `5edd26a6` safely admits F16 primaries, and `794c6a55` batches
  row-dimensional split-K attention. Linear and branching trees pass staged and
  direct bit-identity tests; the full primary proof reports the batch counters.
  Keep the new path opt-in with `CAMELID_METAL_ATTN_BATCH_K=1` until promotion.
- Width-16 packed attention is fail-closed above position 2,048. The first 4k
  stress attempt produced no receipt and left mini2 at its local-login lock.
  Unlock and inspect the machine before any deeper rerun.
- macOS ships **openrsync**: it silently no-op'd large transfers here, twice,
  including with `--checksum`. Use `tar | ssh`, and verify with SHA-256 and size.
- mini2's Thunderbolt link-local address **rotates**; resolve `bridge0` fresh
  each session. Over WiFi it is 1 MB/s vs 400 MB/s over TB.
- The CPU reference lane needs `CAMELID_LAZY_Q8_0_LINEAR=1` for an 8B Q8_0 or it
  refuses at the 6.44 GB materialization cap.
- Health is `/v1/health` with `generation_ready` — `loaded_now` alone is not
  enough, and `/api/health` serves the UI placeholder.
