# Ghost mode: memory-constrained layer-streaming execution (EXPERIMENTAL)

Ghost mode runs models that are far larger than RAM with one of two bounded-storage plans.
Dense models execute one transformer block at a time. Sparse MoE models keep the shared core
available and read only the experts selected by the current layer's router. Both deliberately
trade throughput for a strict application-owned memory ceiling. They live behind the dedicated
`ghost-run` subcommand and the `repack-ghost` tool.

## Why a custom container (`.cghost`)

GGUF groups tensors by export order, scattering one block's tensors across the file. A
layer-by-layer pass over a GGUF therefore degenerates into random reads. `repack-ghost`
rewrites the model so each block's tensors are contiguous:

```
[magic "CGHOST1\0"][u64 index_offset][pad to 16 KiB]
[pre: token_embedding (+ rope_freqs)]
[blk.0: attn_norm, attn_q, attn_k, attn_v, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down]
[blk.1: ...] ... [blk.N-1: ...]
[post: output_norm (+ output unless tied)]
[index JSON]
```

Group starts are 16 KiB aligned (Apple Silicon page size) so later phases can do no-copy
buffer mapping and page-precise `madvise` eviction. Streaming one layer is ONE sequential
`pread` of the whole group.

**v1 is a pure re-layout at source quantization.** This is what makes the correctness gate
possible: identical bytes ⇒ the streamed path must produce a byte-identical greedy token
stream vs the resident path. A mixed-quantization map (high-precision ends + ultra-low-bit
FFN interiors to hit a ~1.65 bit/param average for 70B-on-16GB) is a planned v2 axis — it is
a *quality* trade, and can never be parity-gated against a different-quant baseline. Note
that sub-2-bit formats (IQ1_S / IQ2_XXS class) are not yet supported by the runtime's
loader/kernels; that support is a prerequisite for the v2 map.

## Ghost-MoE v2: Gemma 4 26B-A4B

Gemma 4 26B-A4B has 30 layers, 128 routed experts per layer, and selects 8 experts for each
token. Its shared attention, router, dense shared-expert branch, embedding/head, and norms are
about 1.35 GB in the target 4-bit checkpoint; routed expert tensors account for roughly another
12 GB. Reading a whole layer would discard the architectural advantage of MoE.

For this row `repack-ghost` automatically writes a version-2 `moe_experts` layout instead of the
dense layout:

```text
[magic "CGHOST1\0"][u64 index_offset][pad]
[blk.0.exp.0: fused gate+up slice, down slice]
[blk.0.exp.1: fused gate+up slice, down slice]
...
[blk.29.exp.127: fused gate+up slice, down slice]
[v2 index JSON: layout, model shape, groups]
```

This expert-splicing layout is informed by
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare); Camelid keeps its existing GGUF
binding, tokenizer, router, KV cache, and quantized row kernels around the bounded expert reader.

Every expert group begins on a 16 KiB boundary and contains source-quantized bytes copied directly
from the GGUF. Repacking uses a 1 MiB scratch buffer and never materializes a complete expert
tensor. One cache miss is one sequential positioned read containing both expert projections.

At generation time:

1. The existing Gemma 4 runtime computes attention and the 128-way router from shared GGUF weights.
2. The router chooses the exact same top 8 experts as the normal path.
3. Cache hits reuse the selected expert's wire bytes. Misses `pread` one v2 expert group.
4. Existing Q4/Q8/K-quant row kernels consume the cached bytes without converting them to dense f32.
5. A model-global byte budget evicts old entries; retained expert bytes cannot exceed
   `--expert-cache-mib` regardless of the number of layers or experts visited.
6. Expert reads use normal buffered positioned I/O by default, allowing the OS page cache to
   reuse hot records. `serve --ghost-strict-cache` (or
   `CAMELID_GEMMA4_GHOST_STRICT_CACHE=1`) opts into macOS `F_NOCACHE` for a stricter memory
   ceiling at the cost of that reuse. The original expert tensors in the GGUF are excluded from
   `MADV_WILLNEED` and are never touched by the Ghost-MoE forward path.

The GGUF remains necessary for the tokenizer, metadata, and common weights. The `.cghost` is a
second on-disk artifact containing only routed experts; it does not duplicate the shared core.

New v2 repacks bind those two artifacts with a bounded sampled SHA-256 identity. Camelid samples
every routed expert and every bound common tensor, so startup reads roughly a megabyte of identity
bytes rather than hashing the full 14+ GiB GGUF. The same samples are checked lazily in each
`.cghost` record on first use. Sparse legacy v2 files do not prove artifact identity and are refused
by default. `CAMELID_GHOST_ALLOW_LEGACY_SPARSE=1` is an explicitly unsafe, shape-only recovery
override; full, non-sparse legacy GGUF files remain compatible.

The repacker can make an offset-compatible sparse common-core shadow so the runtime working pair
contains no duplicate routed-expert payload:

```bash
cargo run --release --bin repack-ghost -- \
  archives/gemma-4-26B_q4_0-it.gguf \
  --out models/gemma-4-26B_q4_0-it.cghost \
  --hot-shadow models/gemma-4-26B_q4_0-it.gguf
```

The shadow preserves tiny deterministic identity islands inside the otherwise sparse expert
ranges. It therefore proves that it matches the `.cghost` without restoring the routed weights;
it is not a valid standalone model and must always be used with that matching expert artifact.

### Run Gemma 4 26B-A4B

```bash
camelid pull gemma4_26b

cargo run --release --bin repack-ghost -- \
  models/gemma-4-26B_q4_0-it.gguf \
  --out models/gemma-4-26B_q4_0-it.cghost

cargo run --release --bin camelid -- ghost-run \
  models/gemma-4-26B_q4_0-it.gguf \
  --cghost models/gemma-4-26B_q4_0-it.cghost \
  --expert-cache-mib 1024 \
  --prompt "The capital of France is" \
  --max-tokens 32
```

On Apple silicon, the browser UI/API can use persistent routed-expert slots, fused expert kernels,
and the Q6_K tied head on Metal. This is the fastest measured profile on the tested 16 GiB M4:

```bash
CAMELID_GEMMA4_GHOST_METAL_SLOTS=1 \
CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST=1 \
CAMELID_GEMMA4_GHOST_METAL_TURBO=1 \
CAMELID_GEMMA4_GHOST_METAL_COMMON=1 \
CAMELID_GEMMA4_GHOST_METAL_CONTEXT=1024 \
CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER=80 \
CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT=1 \
CAMELID_GEMMA4_GHOST_READ_THREADS=4 \
cargo run --release --bin camelid -- serve --gpu on \
  --model models/gemma-4-26B_q4_0-it.gguf \
  --cghost models/gemma-4-26B_q4_0-it.cghost \
  --expert-cache-mib 64
```

The profile above runs the complete common core (attention, router, shared expert), the selected
Q4_0 experts, and the Q6_K tied head on Metal. With expanded slot residency this full-common lane
now measures faster than the earlier hybrid recommendation; the hybrid profile
(`CAMELID_GEMMA4_GHOST_METAL_COMMON=0`) remains available and keeps common-core math on CPU. Both
modes retain the same fail-closed CPU fallback.

Kernel selection has three tiers. With fused-fast enabled, the strict 26B Q4_0 projections admit a
32-lane SIMDgroup row kernel that keeps the comparator's increasing-block f32 fold.
`CAMELID_GEMMA4_GHOST_METAL_TURBO=1` additionally admits reassociated-summation kernels: identical
per-block integer dots, but per-lane f32 accumulation folded by a cross-lane `simd_sum`, with four
output rows batched per SIMD group so the shared activation is read once per four weight rows. The
turbo tier trades the ordered comparator's exact f32 summation order for roughly twice the
effective kernel bandwidth; on the acceptance prompt it produced greedy token IDs identical to the
ordered kernels. Devices or shapes that miss admission fall back tier by tier to the scalar
ordered kernel.

`CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT=1` copies the ~600 MB Q6_K tied table into an owned
Metal allocation at load instead of the default file-backed no-copy mapping. The default costs
nothing at load but its clean pages are evictable, so heavy expert paging can silently turn the
per-token head sweep into SSD refaults; the resident copy pins it at the cost of one load-time
copy and ~600 MB of anonymous memory. The copy is the same aligned window at the same buffer
offset, so every consumer — including the opt-in MTP device chain
(`CAMELID_GEMMA4_MTP_DEVICE_CHAIN=1`) — is raw-bit identical over either backing. If the owned
allocation fails, the head logs one line and continues on the file-backed no-copy mapping.

`CAMELID_GEMMA4_GHOST_METAL_STATS=1` prints one compact per-generation slot hit/I/O summary.
`CAMELID_GEMMA4_GHOST_READ_THREADS` controls concurrent positioned reads (default 4, range 1–8).
`--ghost-strict-cache` asks macOS not to retain `.cghost` reads in its file cache; this can reduce
duplicate memory pressure on a 16 GiB unified-memory machine, but should be benchmarked on the
target system because it gives up OS cache reuse.

### Windows CUDA serve lane

On Windows, the same sparse-shadow and `.cghost` pair can serve through the Gemma 4 CUDA runtime.
The common core and KV cache remain resident on the GPU. Selected Q4_0 routed experts are read from
`.cghost` on a cache miss and promoted into a bounded VRAM expert cache; the Q6_K tied head uses the
existing CUDA head kernel. The router still determines the same expert order before any records are
loaded. This lane is experimental until a Windows parity and memory receipt is committed.

```powershell
$env:CAMELID_GEMMA4_GHOST_CUDA = '1'
$env:CAMELID_GEMMA4_GHOST_CUDA_CACHE_EXPERTS = '1000'
$env:CAMELID_GEMMA4_GHOST_CUDA_RESERVE_MIB = '160'

.\camelid.exe serve --gpu on `
  --model models\gemma-4-26B_q4_0-it.gguf `
  --cghost models\gemma-4-26B_q4_0-it.cghost `
  --expert-cache-mib 1024
```

The Windows default remains the parity-checked CPU/storage lane. Set
`CAMELID_GEMMA4_GHOST_CUDA=1` to opt into the experimental CUDA route; it is not a support or
token-parity claim. `CAMELID_GEMMA4_GHOST_CUDA_CACHE=0` disables persistent expert residency.
`CAMELID_GEMMA4_GHOST_CUDA_CACHE_EXPERTS` caps resident routed experts, while
`CAMELID_GEMMA4_GHOST_CUDA_RESERVE_MIB` preserves VRAM for routed scratch and driver/WDDM overhead.
Camelid also clamps the requested capacity to current free VRAM and falls back to CPU/storage if a
single routed expert cannot fit after the reserve. The runtime GPU switch can force an already
loaded lane back to CPU generation and truthful CPU health without a model reload. Starting with
`--gpu off` or `--deterministic` prevents CUDA admission in the first place.

The hot-shadow repacker marks its Windows destination sparse and deallocates routed-expert ranges
with NTFS zero-data control calls. The runtime queries the destination volume's sector size for
unbuffered reads instead of assuming 4 KiB sectors. Keep both artifacts on local NTFS storage;
copying a sparse shadow through a tool or filesystem that expands holes can restore its full logical
disk usage without changing model correctness.

In normal buffered CUDA mode, Camelid maps the immutable `.cghost` payload read-only and uploads
validated routed-expert ranges directly into fixed GPU cache arenas. This avoids allocating and
copying an intermediate expert record on every cache miss. `--ghost-strict-cache` deliberately
disables that mapping and retains the bounded positioned/unbuffered reader path.

The Windows CUDA hot path batches routed hits while a dedicated copy stream fills miss slots.
An eight-slot page-locked host ring pipelines one complete top-8 route without pinning the 12 GiB
expert payload. Q4_0 nibble bias is removed with packed byte subtraction and the exact integer dot
uses DP4A; the existing ordered f32 block fold is unchanged. Routed GeGLU and Q8_0 quantization are
fused, and one router-order kernel replaces eight weighted-accumulate launches. These optimizations
are on by default for the experimental CUDA lane. Set
`CAMELID_GEMMA4_CUDA_BATCHED_EXPERTS=0` for the serial diagnostic path or
`CAMELID_GEMMA4_CUDA_PINNED_EXPERTS=0` to bypass the page-locked transfer ring.

Between the VRAM cache and storage sits a page-locked **host expert tier**: a cacheable pinned
arena of whole `.cghost` records, auto-sized from available host RAM minus a reserve
(`CAMELID_GEMMA4_GHOST_HOST_TIER_MIB` overrides it explicitly, `0` disables;
`CAMELID_GEMMA4_GHOST_HOST_TIER_RESERVE_MIB` adjusts the default 3 GiB reserve). Auto-sizing
refuses to build a tier smaller than a quarter of the routed payload: the tier only sees the
VRAM cache's miss tail, and a small arena measured 0% hits on that stream while pinning RAM
away from the OS page cache that was otherwise absorbing the reads — strictly worse than no
tier. The explicit override still forces any size for measurement. A VRAM miss that
the tier holds costs one async DMA straight from pinned memory — the CPU never touches the bytes —
instead of a storage read. This matters because the tracked machine's NVMe delivers a flat
1.3–1.9 GB/s regardless of read queue depth, so no amount of read parallelism can serve the
~800 MB/token routed working set from disk. `CAMELID_GEMMA4_GHOST_TIER_PREFILL=1` optionally
fills the tier at load with a uniform per-layer stripe of records (mostly sequential reads).
On the tracked 16 GiB machine this measured neutral-to-negative — the prefill read evicts the OS
page cache's copy of the same payload, giving back what the extra tier hits gain — so it is off
by default; it is expected to pay off only when the tier can hold the entire routed payload.
`CAMELID_GEMMA4_GHOST_CUDA_CONTEXT` bounds the KV window (default 4096). Sliding-layer KV caches
are rings of window+1 positions (a sliding layer can never attend further back than its 1024-token
window, so older slots are reclaimed in place), which means only the 5 global layers scale with
context: the 26B row's KV costs ~20 MiB per 1024 positions on top of a ~200 MiB sliding-ring
floor, where it used to cost ~220 MiB per 1024 positions across all 30 layers. Every ~3.2 MiB
returned admits one more resident routed expert, so short-session deployments can still trade
context for hit rate — but full 4096-position context no longer forfeits the bulk of the cache.

Throughput on the tracked RTX 3060 Laptop 6 GiB (i7-11800H, 16 GiB RAM) is dominated by how much
of the 12.0 GiB routed payload is resident somewhere, which makes it strongly dependent on free
host RAM at load. With ~8–9.5 GiB of host RAM free, a 7.1 GiB pinned tier, the 160 MiB reserve,
and the KV window at 1024 (1013-expert VRAM cache), a 128-token greedy run sustained 11.99 decode
tokens/second over its second half; a 256-token run sustained 10.06 (longer sessions route to more
distinct experts, so the storage tail grows). The same binary with the tier disabled sustained
8.50 under the same conditions, and only ~4.3 when background applications held most of the
machine's RAM. The greedy token stream was byte-identical with the tier on and off (96/96), and
the standalone Q4_0 CUDA oracle remained bit-identical on 96/96 rows. With routing artificially
concentrated so the VRAM cache hit 95.9% — expert transfers essentially removed — the lane
measured 19.5–20.6 tokens/second, which bounds what any residency improvement alone can reach on
this machine. A 64 MiB reserve slowed sustained decode under WDDM memory pressure, so the 160 MiB
reserve remains the recommended setting. These are exact-machine measurements, not a general
throughput claim.

The 1 GiB default can retain a little more than one token's 240-expert routed working set on the
tracked Q4_0 row. `--expert-cache-mib 0` retains no routed expert after the current use and gives
the smallest application-owned footprint; smaller budgets may cyclically evict every expert
before the next token revisits its layer. Larger budgets trade memory for hits. The runner reports
hits, misses, evictions, bytes read, retained cache bytes, physical footprint, and peak RSS.

### Real 26B Apple-silicon Metal lane

The tracked 14,439,361,440-byte `gemma-4-26B_q4_0-it.gguf` was repacked into 3,840 expert
groups with 11.96 GiB of routed payload. The production pair is strongly fingerprinted and the
sparse GGUF shadow occupies about 1.5 GiB physically. On the strict 26B geometry, Camelid can keep
attention, router, the shared expert, the selected Q4_0 experts, the Q6_K head, and the f32 KV
cache on Metal. Persistent expert residency is configurable from 8 through 128 slots per layer
(3.2 MiB per slot per layer): 24 slots use 2.25 GiB, 64 use 6.0 GiB, 80 use 7.5 GiB.

Measured on a 16 GiB Apple M4 with both artifacts on the internal SSD, acceptance prompt
`In one sentence, explain why local AI is useful.`, greedy decoding:

- Earlier hybrid recommendation (24 slots, 1 GiB host cache, ordered kernels): 32 visible tokens
  in 8.42–9.04 s (3.54–3.80 tok/s end-to-end), reproduced at 3.9–4.1 tok/s. Slot hit rate 61.7%,
  290 MiB read from `.cghost` per routed position — decode was dominated by synchronous expert
  page-in on the token critical path.
- Current recommendation (full-common Metal, 80 slots, turbo kernels, resident head, 64 MiB host
  cache): 256 visible tokens in 20.2 s (12.7 tok/s end-to-end); steady-state per-token walls of
  50–58 ms (17–20 tok/s) once residency converges, with a 97.5% slot hit rate and 19.2 MiB read
  per routed position. The same profile with `CAMELID_GEMMA4_GHOST_STRICT_CACHE=1` measured
  9.4 tok/s over 128 tokens with lower memory pressure; prefer strict cache when other
  applications need the page cache.
- 96 slots per layer exceeded this machine's comfortable budget (swap traffic during warm-up) and
  bought almost no additional hit rate; 80 is the measured sweet spot on 16 GiB.

The end-to-end figures include the cold warm-up phase in which slots first populate; a persistent
`serve` process keeps slots resident across requests, so follow-up requests decode at the
steady-state rate from the first token. Greedy token IDs: the turbo kernels produced IDs identical
to the ordered kernels for the full acceptance run. Separately, the CPU-storage, hybrid, and
full-common lanes each diverged from one another within the first few positions on this
degenerate completion-style prompt — greedy lane-vs-lane agreement is NOT currently a stable
property of this experimental lane and earlier claims of identical 32-token IDs did not reproduce
after the artifact was regenerated. These are exact-machine, exact-prompt measurements rather
than a general throughput claim.

Current boundary: this is an experimental single-node lane. Dense ghost's `--stage-split` and
`--spec` options are refused because a routed expert cannot be prefetched until that layer's router
has produced its IDs.

## v1 runner (synchronous)

`camelid ghost-run <model.gguf> --cghost <model.cghost> --prompt ... --max-tokens N`

- Metadata, tokenizer, and the resident ends (embedding + output projection) load from the
  GGUF via the existing loaders; every transformer layer starts as an empty placeholder.
- Per chunk (prefill or one decoded token), per layer: one sequential group read into a
  reused buffer → decode to the same in-RAM storage the resident loader produces → run the
  existing CPU layer forward (`ghost_forward_one_layer`, which does not advance the KV
  position; the runner advances once per chunk) → swap the placeholder back in, dropping
  the weights. The weight working window is exactly one layer.
- Greedy sampling via the existing final-norm/logits path.
- v1 (`--sync-stream`) blocks the forward on each read. v2 (the default) double-buffers:
  a background worker reads + decodes layer N+1 while the main thread runs layer N's
  forward, handing off over a rendezvous channel so at most TWO layer windows exist at any
  instant. The worker is also primed with the next chunk before the current one finishes,
  so the disk is already rewinding to layer 0 of token N+1 during the last forwards (and
  the sampling) of token N. The trace reports the residual stall ("blocked") separately
  from forward time.
- Strict memory ceiling mode: `--evict-page-cache` sets `F_NOCACHE` on the `.cghost`
  handle so streamed pages bypass the page cache entirely. Off by default — when the model
  fits in RAM the cache is a free win; for the over-RAM models ghost targets, the cache can
  only thrash. (`posix_madvise(DONTNEED)` does not apply to this design — that is for
  mmap'd ranges, and the streamer uses positioned reads.)
- The two-node pipeline variant (each node hosts half the `.cghost` and overlaps its disk
  window with the other node's compute) is the next phase.

## Step-1 measurements (Llama-3.2-3B-Instruct Q8_0, M4 16GB)

- Repack: 30 groups, 3.18 GiB payload, largest block group 102.0 MiB (the per-layer
  streaming window), ~3 min on the external SSD.
- Runner footprint: **1.36 GiB** after loading resident ends (vs ~3.6 GiB fully resident) —
  the whole point of the mode.
- Streaming rate is storage-bound: ~38 MB/s cold on this external SSD (the same rate GGUF
  loads see there), i.e. ~2.8 s/layer. On internal NVMe-class storage the same read is
  projected at tens of milliseconds. Throughput numbers for ghost mode are therefore quoted
  per storage tier; the PoC gate is correctness + the memory ceiling, not speed.

## Task 3: the two-node ghost mesh (2026-06-03)

`distribute-master --cghost <shard>` / `distribute-worker --cghost <shard>`: each pipeline
node streams its own layer shard per token from a node-local `.cghost` (made with
`repack-ghost --layers a..b`), double-buffered, holding only the embedding (first node) or
output ends (last node) resident. The cross-node overlap falls out of chunk priming: a node
queues its next chunk during its own last layer, so its disk window runs while the OTHER
node computes and the activations cross the bridge.

Measured — Llama-2-13B Q8, two M4 16 GB minis over Thunderbolt, split 0..20 / 20..40, each
shard 6.44 GiB on the node's internal NVMe (321.4 MiB/layer window):

- **Peak RSS 1.80 GB per node** (resident pipeline needs 7.2–7.8 GB/node) — the fit story:
  the mesh runs a model whose resident form nearly fills both machines, in ~an eighth of
  the memory.
- Greedy output identical to the resident pipeline's stream.
- Steady state ~4.05 s/token = **0.24 tok/s**: master ~3.0 s (disk-bound: 6.44 GiB at
  ~2.15 GB/s; its own window cannot hide behind anything), worker ~1.04 s visible (about
  two-thirds of its window streams during its idle while the master computes), feedback
  ~5 ms. Without the cross-node overlap the same token would cost ~6 s.
- Split tuning is an open lever: the first node's window is unhideable, so giving it fewer
  layers (e.g. 14/26) should shrink the critical path until the last node's residual
  window surfaces.

This is the throughput-for-memory trade at mesh scale: per-storage-tier numbers as always —
faster NVMe (or smaller shards from a future low-bit map) moves the disk-bound term
directly.
