# Handoff — speculative decoding campaign

Branch `campaign/mtp-3x`. **Unpushed.**
Worktree `/Volumes/Untitled/Camelid-3x`. The EAGLE benchmark and experimental
v2/MMA K-quant lanes remain opt-in. Phase 0's SPEC_GPU helper does auto-arm on
eligible Metal speculative requests unless explicitly disabled.

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

**Llama 3.2 3B EAGLE-3 — BENCHMARK BUILT; NOT NATIVE MTP OR SERVING
INTEGRATION.** The learned EAGLE-3 benchmark is pinned to the exact target
`Llama-3.2-3B-Instruct-Q4_K_M.gguf` (sha256
`6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`,
2,019,377,696 bytes) and the matching Thoughtworks EAGLE3 head at revision
`02d343789b502a3edfe351bdd4537a44affb98cd` (`model.safetensors` sha256
`c0713251464a9b6b5fcf9fb229587bbe59b6fd1521027aef32101d11b9ebbdaf`,
486,297,280 bytes). It captures resident target taps `[2, 14, 25]`, keeps both
the learned head and target resident on Metal, and enforces losslessness and
head/target cache-watermark gates.

The final committed raw-prompt result at gamma 3 is
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
chat accelerator; bounded top-k tree EAGLE is the next credible speed lever.

The exact matching head declares training sequence length **1024**; it does not
declare a runtime cap. The old Llama 3.1 / 2048 account was about a different
artifact and must not be carried forward. This remains a benchmark-only EAGLE-3
path, not a native MTP implementation and not integrated into serving. Full
method and receipts:
[`qa/speculative/EAGLE3_LLAMA32_3B.md`](qa/speculative/EAGLE3_LLAMA32_3B.md).

## Next lever for speed

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
  gates are armed, and no test in the tree sets kv16 — so it can go green without
  touching the K-quant path. It now prints `kv_format`; run it with
  `CAMELID_METAL_KV_DTYPE=f16` to exercise the primary. Filters go AFTER `--`.
- A K-quant micro/unit identity pass is still not a promotion gate. The direct
  fragment candidate passed both but diverged only in the warmed 4137-token
  model run. Require two warmed 96-token exact runs after those fast gates.
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
