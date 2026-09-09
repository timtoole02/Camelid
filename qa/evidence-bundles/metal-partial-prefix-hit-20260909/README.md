# The prompt-prefix cache is a 15.6x regression on the Metal K-quant lane

`results.txt` has the numbers, `host.txt` the machine. `prefix-probe.sh` reproduces the
regression against an unmodified `origin/main` binary; `verify-fix.sh` is the ABBA over
this branch; `parity-control.sh` establishes the cold-prefill reference output.

## What happens

Turn 2 of any conversation carrying a system prompt shares more than
`CAMELID_PREFIX_CACHE_MIN_TOKENS` (default 16) leading tokens with turn 1, so it takes a
PARTIAL prompt-prefix-cache hit. `resume_partial_prefix_hit` rolls the cached session back
to `kv_position = prefix_len > 0` — and the batched Metal prefill declines outright on a
non-zero position, so the divergent suffix falls to the CPU dense forward.

The cache built to make repeat turns fast makes them 15.6x slower, and it only bites once
the cache actually HITS. Q8_0 escapes solely because `kv_roundtrips_through_cpu_exactly`
refuses it entry to the pool, which leaves K-quant models roughly 18x slower than Q8_0
from turn 2 onward despite carrying smaller weights.

It is also not output-neutral. Against a cold-prefill reference the resumed path returned
turn 0's answer for turns 1 and 2, matching on 1 of 3 prompts where the fix matches 3 of 3.

## The fix

`resume_partial_prefix_hit` now asks whether the batched Metal prefill would have taken
this prompt, and declines the partial resume when it would. The caller then prefills the
whole prompt on the GPU from position 0: faster, and the bit-exact reference path.

The predicate is `metal_resident_prefill_shape_admits`, factored out of
`try_metal_resident_prefill_inner` so both callers share one body and cannot drift about
what "eligible" means. The prefill keeps the `position == 0` clause; the cache asks the
same question without it.

Scope, deliberately narrow:
  * exact hits are untouched — they replay stored logits and never prefill;
  * non-Metal sessions are untouched — the predicate is false, so they resume as before
    (covered by `a_cpu_only_session_still_takes_the_partial_resume`);
  * windowed archs were already refused at this site and still are.

## What this is NOT

A floor, not a ceiling. Declining a hit still recomputes the shared prefix. The real win
is to CONTINUE the batched prefill from `p`: the scatter and attention uniforms already
exist and are hardwired to zero ("base position: prefill always starts an empty cache"),
and the MSL kernels already accept `base_position`. That is a separate change needing its
own token-identity qualification. This one only stops the bleeding, and does so on a path
whose correctness argument is "do what a first turn does".

## Reproducing

    # regression, against a stock origin/main binary
    bash prefix-probe.sh

    # the fix, ABBA, one binary both arms
    bash verify-fix.sh

    # what a cold prefill actually answers
    bash parity-control.sh

`CAMELID_METAL_PREFIX_PARTIAL_RESUME=1` restores the old behaviour so both arms come from
the same binary. It is not a tuning knob and is documented as existing only for this A/B.
