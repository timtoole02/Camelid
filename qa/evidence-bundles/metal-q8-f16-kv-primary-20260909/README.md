# Q8_0 on an F16 KV primary: making the option real, and measuring it

`results.txt` has the numbers. `q8kv-probe.sh` produces the speed table,
`kvmem.sh` the footprint.

## The exclusion

`use_attn_mm` gated the attention-as-matmul prefill behind
`(!self.kv16 || (use_kq_mm && kquant_attn_mm_prefill_enabled()))`. A Q8_0 model on an F16
primary satisfies neither arm — it is `kv16`, and it is not `use_kq_mm` — so it lost a
lane it *already had* on its F32 primary.

That is a pure loss rather than a trade. `all_q8` is admitted below in the very next
conjunct for the same reason it is admitted here: a Q8_0 model already runs this path by
default and already accepts the half-staged scores it implies. Moving it onto an F16
primary changes where the half K/V is read from, not whether the trade is taken. And the
binding was already in place — `attn_k16`/`attn_v16` take `&self.cache_k[i]` when
`self.kv16`.

## Why the first number was wrong

The first measurement put the F16 primary at **42.7 s** against 4.53 s. That is a real
observation and a misleading one: about 32 s of it was a partial prompt-prefix-cache hit
falling to the CPU dense forward — a separate defect with nothing to do with KV dtype.
Attributing it here would have condemned the wrong thing.

On a tree carrying that fix, the exclusion costs **2.24x**, which is exactly the "~2.2x"
already recorded on `KQUANT_LANE_ENGAGED`. With this clause it is **1.11x**.

## The stale half of the existing warning

That same comment says the change cost "~26% decode and ~2.2x prefill". The prefill half
reproduces. **The decode half does not** — 29.94/30.02 f32 against 29.98/30.03 f16, which
is no difference. Split-K decode attention gained kv16-primary support after the note was
written. Worth knowing, because that warning is the reason nobody revisits this.

## What is left as a decision

With the exclusion gone the F16 primary costs **~11% prefill** and **0% decode**, produces
**identical output**, and saves **597 MB** on a 3B at ~4200 positions.

This PR does not flip the default. It makes `CAMELID_METAL_KV_DTYPE=f16` a real option
instead of a 2.24x trap, and puts numbers under the choice.
