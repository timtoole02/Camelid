# Ubuntu x86 Q8 weekend handoff — 2026-05-15

This is an experimental-lane handoff note. It records what is saved for follow-up; it is not a default-on, production, or broad benchmark claim.

## Current decision state

- **Retain / continue from:** `ubuntu-x86-q8-ffn-down-safe-design` at `f23c3f4`.
  - Adds a default-off FFN-down decode consumer separate from the old suspect owner/residual path.
  - Clean repeat vs `7034294` showed a real narrow improvement with stable output.
- **Earlier required base:** `ubuntu-x86-q8-planner-attn-consumer` at `7034294`.
  - Makes the Ubuntu x86 optimized Rust Q8 runtime path select and run with attention projection packed consumers.
  - Clean repeat beat the safe/reference Camelid path.
- **Rejected:** `ubuntu-x86-q8-qkv-fused-consumer` / combined `2688bf4` QKV fused candidate.
  - Clean repeat worsened wall/generate/layers and made Q/K/V slower.
- **Parked experimental:** `ubuntu-x86-q8-ffn-gate-up-design` at `95e65c1`.
  - Clean repeat showed only a small win; profile did not justify promoting it as the main retained step yet.
- **Rejected as performance win / neutral:** `ubuntu-x86-q8-ffn-down-kernel-next-20260515` at `a8798e0`.
  - Correctness passed, but clean repeat was effectively neutral vs `f23c3f4`.

## Key clean-repeat numbers

### `7034294` vs safe/reference

`7034294` optimized Rust Q8 path vs safe/reference Camelid, n=8, 5-run medians:

- Wall: `6.690535s` vs `8.227344s`
- Generate: `3016 ms` vs `5545 ms`
- Layers: `2939.273 ms` vs `4960.898 ms`
- Output text/checksum stable.

### `f23c3f4` vs `7034294`

FFN-down consumer on vs off, n=8, 5-run medians:

- Wall: `-171.909 ms`
- Generate: `-156 ms`
- Layers: `-155.727 ms`
- FFN-down: `-160.868 ms`
- Output stable.

### Rejected QKV fused candidate

Combined QKV fused candidate vs `f23c3f4`, n=8, 5-run medians:

- Wall: `+3.00%`
- Generate: `+6.24%`
- Layers: `+6.33%`
- Q/K/V slowed by roughly `+32% / +52% / +49%`.

### Gate-up candidate

Gate-up candidate vs `f23c3f4`, n=8, 5-run medians:

- Wall: `-13.743 ms`
- Generate: `-22 ms`
- Layers: `-20.972 ms`
- Output stable.
- Status: keep as experimental/parked pending complexity decision.

### Kernel tweak candidate

`a8798e0` vs `f23c3f4`, n=8, 5-run medians:

- Wall: `+2.836 ms`
- Generate: `-13 ms`
- Layers: `-13.179 ms`
- FFN-down: `-0.339 ms`
- Status: not enough to retain as a performance win.

## External comparison status

No clean external “beats llama.cpp/Ollama” claim has been earned yet. Same-host llama.cpp reruns were contaminated. The honest current claim is internal: the optimized Rust path now repeatably beats Camelid safe/reference on this exact experimental lane.

## Monday resume plan

1. Start from `f23c3f4` as the current best retained experimental branch.
2. Decide whether to merge/PR `7034294` + `f23c3f4` as a coherent default-off Ubuntu x86 Q8 experimental slice.
3. Do not retain QKV fused or `a8798e0` as performance wins.
4. Keep `95e65c1` gate-up parked unless code complexity is acceptable for a small gated win.
5. Next serious optimization should be a larger FFN-down / packed matmul / scheduler-amortization design, not more tiny leaf tweaks.
6. After the retained stack is settled, rerun clean same-host external comparisons against llama.cpp and Ollama with contamination guards.

