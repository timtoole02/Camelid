# CAMELID_FLASH_PREFILL — measured A/B, 2026-09-10 (NEGATIVE RESULT)

The first isolated flash-on vs flash-off measurement of the **current** kernel.
It is **slower** on this device and it **breaks its documented token-parity claim**
at 6k context.

Nothing here changes a default: the flag was already off and stays off. This exists
so the next person does not spend a day rediscovering it, and because the repo
currently carries no evidence isolating this flag at all.

## Why there was no prior evidence

The measured wins associated with flash prefill belong to a kernel that **no longer
exists**:

| commit | what happened |
|---|---|
| `34f36b1e` | shipped flash prefill (Phase P M1) — 2.2x/3.1x/11.5x @ 6.6k/8.8k/12.2k |
| `a964411f` | const-k=8 unroll, +9-18%, byte-identical |
| `2b7debee` | **replaced** it with a different design (dynamic chunked online softmax) |
| `a653b4ac` | **deleted** the original 3-pass kernels |

`improvements.md` reports "20.95x acceleration" and "TTFT 2,140 ms -> 70.8 ms", but
those are **GPU-resident vs CPU**, not flash-on vs flash-off. They do not measure
this flag.

`bdfc5e97` ("make the flash prefill claims true") had already corrected three false
claims in this feature, and left this question open verbatim:

> head_dim 128 costs 71 -> 89 registers for that, which is 3 -> 2 blocks/SM at 256
> threads, and **only a real prefill A/B can say whether the spill or the occupancy
> was the more expensive one**.

This is that A/B. The occupancy loss costs more than the spill it removed.

## Performance: slower, and worse with context

Llama 3.2 3B Q8_0 fully resident, RTX 3060 Laptop (sm_86), one release build, arm
set only by `CAMELID_FLASH_PREFILL`. **ABBA** (off, on, on, off) with a cool-to-55C
gate. Every prompt is unique at token 0 so prefix continuation finds nothing to
reuse — these are genuine cold prefills.

Prefill time from the engine trace (`prefill-trace.txt`):

| prompt tok | off-1 | off-2 | on-1 | on-2 | flash vs off |
|-----------:|------:|------:|-----:|-----:|-------------:|
| 1424 | 10347 | 10472 | 11546 | 11529 | **1.11x SLOWER** |
| 3025 | 25694 | 26260 | 30361 | 30366 | **1.17x SLOWER** |
| 6024 | 70428 | 73762 | 88294 | 88242 | **1.22x SLOWER** |

The pairing is sound: the two `on` arms agree within 0.1%, and the arm run **last**
(off-2) matches the arm run **first** (off-1), so thermal drift is cancelled rather
than assumed away — the standing hazard on this laptop, where drift has previously
manufactured a 1.8x "win" out of nothing.

The penalty grows with context, which is the opposite of what a flash kernel is for.

## Token parity: holds at 1.4k, breaks at 6k

Three distinct prompts at two lengths, greedy (`temperature: 0`), 48 max tokens
(`parity-off.txt` vs `parity-on.txt`):

| context | token-identical |
|---|---|
| 1,429 tok | **3 of 3** |
| 6,029 tok | **1 of 3** (`notes` and `story` diverge; `log` matches) |

Example divergence at 6,029 tokens:

```
off: "...and do not contain any coherent text that can be summarized in a single sentence."
on:  "...and there is no clear text or summary to provide."
```

This is deterministic, not flaky: both `on` arms of the timing sweep produced the
same divergent reply, and both `off` arms produced the same baseline reply.

That contradicts the documented contract, which says the path "preserves greedy
token-parity". It is also the failure mode the repo has recorded before: an f32
re-association applied **per layer** compounds across layers and flips greedy
tokens, and only a final projection can tolerate one.

The divergence is itself the proof the kernel engaged — the arms differ only by the
env var, so identical binaries producing different text means the flash path ran.

## Read this as device-scoped

One device (sm_86, RTX 3060 Laptop), one model, one prompt family. The kernel was
developed and reported on sm_89 (L4 / RTX 4060 Laptop), and the register/occupancy
tradeoff `bdfc5e97` describes is architecture-sensitive, so it may well behave
differently there. What is NOT device-scoped is the parity finding: a per-layer
re-association that flips greedy tokens at 6k will do so on any device.

## Recommendation

Keep it default-off, and stop describing it as token-parity without a length bound.
Anyone wanting flash prefill on this hardware should start from `34f36b1e` (the
deleted M1 design, which was byte-identical after `a964411f`) rather than the
current kernel.
