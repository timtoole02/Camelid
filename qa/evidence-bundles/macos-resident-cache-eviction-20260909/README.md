# Unloading a model freed almost none of its memory on macOS

`release_model` drops the CPU-side registries and then calls
`inference::reset_resident_caches`. That function has a real CUDA implementation — its own
doc records the leak it exists to prevent, "~4.7 GB stayed on the device ... making decode
~20x slower" — and on every non-CUDA host it was `{}`.

macOS is not a host with nothing to free. `serve` turns `CAMELID_METAL_NOCOPY` on by
default, so a model's weights ARE its page-aligned `WirePages` allocation, and the
process-global Metal buffer cache holds an `Arc` to it. That `Arc` is the surviving owner:
dropping every registry frees nothing, the pages stay resident for the life of the
process, and a reload maps the file again at a new address and takes a second full copy.

`results.txt` measures it. After two model switches a stock build holds **~8.2 GB for one
live model** on a 16 GiB machine, including two independent copies of the same 3B; the
fixed build holds **~4.1 GB**.

It also made the fit advisor lie. `model_requires_unload` tells the user "Releasing it
frees ~N GB, which should be enough" using the GGUF file size, and on macOS releasing it
recovered approximately none of it — so the retry that message recommends could not
succeed.

## The fix, and why it is safe

Sweep the no-copy map and drop entries whose backing allocation has `strong_count == 1`,
i.e. where this cache holds the only `Arc` and the model that loaded those pages is gone.

That is the entire safety argument, and it needs no ownership plumbing: an `Arc` can only
be obtained by cloning an existing one, so anything still able to reach those pages is
itself already counted. A live model reads `>= 2` and is kept. The cache mutex is held
across the sweep, and a count of 1 cannot race upwards for the same reason.

A blanket `clear()` would NOT be safe — some buffers wrap pages owned elsewhere, so
clearing could free memory out from under a live model. Refcount-directed eviction cannot
do that.

`resident_weight_eviction_frees_only_pages_no_model_still_owns` asserts both directions on
one cache — a live owner's pages survive a sweep, a dropped owner's are reclaimed and
reported — plus idempotency, so a second unload cannot double-count.

## Scope

Only `q8_wire_nocopy_buffers`. The other four permanent maps hold GPU *copies* keyed by a
source `(ptr, len)` with no owner handle to test, so nothing there can distinguish a live
entry from a dead one; reclaiming them needs a per-model owner and is a separate change.
The no-copy map is where the GB-scale weights actually live on the default `serve` path,
which is why this is the one worth doing first.

## Reproducing

    bash evict-probe.sh <path-to-camelid> <label> <port>

Run it once with a stock binary and once with this branch. Add
`CAMELID_RESIDENT_TRACE=1` to the server to see each eviction as it happens.
