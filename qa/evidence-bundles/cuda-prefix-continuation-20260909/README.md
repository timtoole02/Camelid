# CUDA prefix continuation — multi-turn A/B, 2026-09-09

Receipt for `perf(cuda): continue a resident prefill from the KV it already holds`.

## What was measured

A growing conversation, which is the shape the change exists for: every turn
re-sends the whole history, so turn *k*'s prompt is turn *k-1*'s prompt plus the
assistant's reply plus one new user message. A single-prompt benchmark cannot show
this effect at all.

- Model: Llama 3.2 3B Instruct Q8_0, fully resident (all 28 layers in VRAM).
- Host: RTX 3060 Laptop (6 GiB, compute 8.6), Windows/WDDM, 16 logical cores.
- Binary: one `--release` build of the branch. **Both arms are the same binary**;
  the arm is `CAMELID_CUDA_PREFIX_CONTINUATION=1` vs `=0`.
- 5 turns per arm, prompt growing 1437 → 1590 tokens, `temperature: 0`,
  `max_tokens: 16`, non-streaming `/v1/chat/completions`.
- Design: **ABBA** (on-1, off-1, off-2, on-2) with a cool-to-≤55 °C gate between
  arms. Sequential GPU A/Bs on this laptop have manufactured ~1.8× "wins" out of
  thermal drift alone, so the interleaving and the cooldown are part of the
  method, not ceremony. One server per arm, because the setting is read from the
  process environment and the resident engine is process-global.

Harness committed here: `ab.sh` (ordering + cooldown), `run-arm.sh` (one arm),
`bench-turns.mjs` (the growing conversation). Raw per-turn records: `turns.jsonl`.
Engine trace: `trace.txt`. Host banner: `host.txt`.

## Result

Request wall time in ms (client-side, includes decode of ~8 tokens):

| turn | prompt tok | on-1 | on-2 | off-1 | off-2 | off/on |
|-----:|-----------:|-----:|-----:|------:|------:|-------:|
| 1 | 1437 | 10344 | 10254 | 10328 | 10334 | 1.00 |
| 2 | 1474 |   959 |   971 | 10759 | 10747 | 11.14 |
| 3 | 1512 |   985 |   984 | 11153 | 11128 | 11.32 |
| 4 | 1550 |  1018 |  1078 | 11429 | 11771 | 11.07 |
| 5 | 1590 |  1018 |  1000 | 11744 | 12008 | 11.77 |

**Follow-up turns: 11.3× (mean 11342 ms → 1002 ms).**

**Turn 1 is the built-in control and it did not move** (10299 ms ON vs 10331 ms
OFF, 0.3%). Nothing is resident on the first turn, so the code path is identical;
an arm difference there would have meant the pairing was measuring something else.

**The flatness matters more than the ratio.** OFF degrades as the conversation
grows — 10759 → 11744 (off-1) and 10747 → 12008 (off-2), +9% and +12% over four
turns, because it re-prefills a longer history every time. ON is flat: 959 → 1018
and 971 → 1000. The OFF arm's cost is a function of conversation length; the ON
arm's is a function of what the user just typed.

## The mechanism, not just the number

`CAMELID_RESIDENT_TRACE=1` shows the reuse directly rather than leaving it to be
inferred from timings (`trace.txt`):

```
[resident-cuda] prefix continuation: reused 1436 of 1473 positions, prefilled 37
[resident-cuda] prefix continuation: reused 1473 of 1511 positions, prefilled 38
[resident-cuda] prefix continuation: reused 1511 of 1549 positions, prefilled 38
[resident-cuda] prefix continuation: reused 1549 of 1589 positions, prefilled 40
```

The OFF arm emits none of these lines, by construction.

Prefill time is the whole difference — decode is unchanged:

| turn | prefill ON | prefill OFF |
|-----:|-----------:|------------:|
| 1 (cold) | 10113 ms | 10030 ms |
| 2 | 706 ms | 10453 ms |
| 3 | 731 ms | 10876 ms |
| 4 | 721 ms | 11112 ms |
| 5 | 782 ms | 11497 ms |

## Output identity

Two independent checks, at different levels.

**Kernel level, bit-exact.** `prefill_then_decode_matches_sequential` prefills half
a prompt, continues over the reused rows, and asserts the resulting logits match a
full prefill's **to the bit** (`assert_same_bits`, not a tolerance) on both the
serial and batched paths. It ran on this device in 30 s — it did not early-return
on a missing device.

**End-to-end, served.** The same 5-turn conversation was replayed through
`/v1/chat/completions` on both arms with the replies recorded
(`bench-replies.mjs`, `replies-on.txt` vs `replies-off.txt`). They are
**byte-identical at every turn**:

```
turn 1 prompt_tokens=1437 reply="I have received your notes."
turn 2 prompt_tokens=1474 reply="I have noted the additional notation."
turn 3 prompt_tokens=1512 reply="I have recorded the new notation."
turn 4 prompt_tokens=1550 reply="I have added the notation to the list."
turn 5 prompt_tokens=1590 reply="I have updated the notation."
```

`diff replies-on.txt replies-off.txt` is empty. This is the check that matters for
a user: continuing from reused KV answered exactly as re-prefilling from scratch
did, through the real server, across a growing conversation — not just in a
kernel harness. (`prompt_tokens` also matched across all four timing arms, which
is the same signal more weakly: each follow-up prompt embeds the previous reply,
so a divergence would have shifted the counts.)

## Caveat

One host, one model, one prompt shape. The ratio depends on how much of the prompt
is shared: it is large here because a chat turn appends ~40 tokens to a ~1500-token
history, which is the normal case for chat and agent loops and the case the change
targets. A workload that sends unrelated prompts every time reuses nothing and
should see turn-1 behaviour throughout — no gain, and no loss.
