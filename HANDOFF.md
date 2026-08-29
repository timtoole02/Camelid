# Handoff — speculative decoding campaign

Branch `campaign/mtp-3x`, 28 commits ahead of `origin/main` (b8951e24). **Unpushed.**
Worktree `/Volumes/Untitled/Camelid-3x`. Nothing here is on by default.

## The result

Llama-3-8B **Q4_K_M**, 4137-token agentic prompt, one machine (mini2), one prompt;
each row differs from the one above by a single change.

| | plain | speculative | verify round |
| --- | --- | --- | --- |
| default lane (origin/main) | 7.13 | 4.72 | 1528 ms |
| + K-quant v2 / MMA verify | 8.22 | 9.55 | 389 ms |
| + suffix drafter on the chain | 8.22 | 11.41 | 621 ms |
| **+ kv16 split-K attention** | **13.51** | **30.68** | **215 ms** |

**7.13 -> 30.68 tok/s = 4.30x.** Speculation went from a 0.66x *regression* to
2.27x. Plain decode alone went **1.89x**, which lands for any K-quant workload at
depth whether it speculates or not. All arms `lossless=true`.

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

**Speed — DONE and safe.** Committed, receipted, gated.

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

**MTP / EAGLE-3 — JUSTIFIED, NOT BUILT.** Suffix drafting hits **0% acceptance**
on creative writing and on 60-unrelated-words, where speculation becomes a
0.80-0.91x *loss*. That prose gap is what a trained head fills. Artifacts are
downloaded and validated: `eagle3-llama31-8b.safetensors` (MIT, 15 tensors,
`fc [4096,12288]` 3-tap fusion, 32k draft vocab + d2t/t2d) and both 3.1 GGUFs.
Two risks before funding it: the head is trained for Llama-3.**1** (not the 3.0
row we measure), and `max_position_embeddings: 2048` while everything we care
about is at 4137 tokens.

## Next lever for speed

Re-attribution after the fix: attention is now **1.9 ms/verify-column, down from
52.8**; rope, scatter and argmax measure zero. The remaining ~25 ms/col is inside
the **K-quant MMA GEMV**. Separately, plain decode sits at **55% of the ~120 GB/s
wall** (66 GB/s), and the mc-gemv campaign's own microbench shows even a
math-free weight-stream probe reaching only 89 GB/s at these shapes — so roughly
1.6x is unclaimed in the decode kernel itself. Use
`CAMELID_VERIFY_ABLATE=<stage>` (measurement-only, outputs garbage) to attribute.

## Traps worth knowing

- `metal_spec_verify_bit_identical` **skips silently** unless the fast-stack
  gates are armed, and no test in the tree sets kv16 — so it can go green without
  touching the K-quant path. It now prints `kv_format`; run it with
  `CAMELID_METAL_KV_DTYPE=f16` to exercise the primary. Filters go AFTER `--`.
- The tree verify is ~350 ms/verify-column vs the chain's 27 and is NOT worth
  chasing: admitting kv16 there changed nothing (352 vs 349). Its cost is
  structural. `encode_attention_tree` must KEEP its `!kv16` guard — that dispatch
  unwraps the mirrors and panics on a primary.
- macOS ships **openrsync**: it silently no-op'd large transfers here, twice,
  including with `--checksum`. Use `tar | ssh`, and verify with md5/size.
- mini2's Thunderbolt link-local address **rotates**; resolve `bridge0` fresh
  each session. Over WiFi it is 1 MB/s vs 400 MB/s over TB.
- The CPU reference lane needs `CAMELID_LAZY_Q8_0_LINEAR=1` for an 8B Q8_0 or it
  refuses at the 6.44 GB materialization cap.
- Health is `/v1/health` with `generation_ready` — `loaded_now` alone is not
  enough, and `/api/health` serves the UI placeholder.
