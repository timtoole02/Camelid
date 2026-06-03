# Ghost mode: memory-constrained layer-streaming execution (EXPERIMENTAL)

Ghost mode runs models that are far larger than RAM by executing one transformer block at a
time: only a tiny active working window (one layer's weights) plus the embedding/output ends
and the KV cache are materialized; everything else streams sequentially from disk every
token. It deliberately trades throughput for a strict memory ceiling — the inverse of the
primary inference paths, which hard-require RAM-resident weights. The two paths share
kernels but are otherwise isolated: ghost lives behind the dedicated `ghost-run` subcommand
and the `repack-ghost` tool, and touches none of the resident decode loops.

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
- Reads block the forward in v1. Double-buffered async prefetch (read layer N+1 while layer
  N computes), explicit page eviction (`posix_madvise(DONTNEED)`), and the two-node pipeline
  variant (each node hosts half the `.cghost` and overlaps its disk window with the other
  node's compute) are the planned next phases.

## Step-1 measurements (Llama-3.2-3B-Instruct Q8_0, M4 16GB)

- Repack: 30 groups, 3.18 GiB payload, largest block group 102.0 MiB (the per-layer
  streaming window), ~3 min on the external SSD.
- Runner footprint: **1.36 GiB** after loading resident ends (vs ~3.6 GiB fully resident) —
  the whole point of the mode.
- Streaming rate is storage-bound: ~38 MB/s cold on this external SSD (the same rate GGUF
  loads see there), i.e. ~2.8 s/layer. On internal NVMe-class storage the same read is
  projected at tens of milliseconds. Throughput numbers for ghost mode are therefore quoted
  per storage tier; the PoC gate is correctness + the memory ceiling, not speed.
