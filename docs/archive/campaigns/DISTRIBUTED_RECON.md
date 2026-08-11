# DISTRIBUTED_RECON.md — Phase 0 recon for the distributed parity lane

Date: 2026-06-13. Branch: `feat/distributed-parity-lane` (off `origin/main`).
Scope: read-only recon, accurate to source at this commit. No code changed.

This document answers the four Phase 0 questions from the build spec against the
**actual** source, and then records the single most important finding: **most of the
distributed *infrastructure* this lane describes already exists** — for Llama as a
working-but-unvalidated N-node pipeline, and for Gemma 4 as a parity-validated two-node
path. The real gap is the **parity receipt for the Llama distributed path**, not the
plumbing. The build order in the spec must be re-read in that light (see "Capability
inventory" below and the companion `DECISIONS.md`).

---

## 1. Layer loop location

There are **two** distinct decoder-layer loops, because the engine has two execution
lanes. Both iterate `self.weights.layers` and dispatch one layer at a time through a
clean per-layer function.

- **Single-token / per-token decode loop** — `src/inference.rs:2677-2728`, inside the
  forward path, iterating `for (layer_idx, layer) in self.weights.layers.iter().enumerate()`
  and calling the per-layer function `forward_layer_timed()` (`src/inference.rs:4159`,
  signature `fn forward_layer_timed(hidden: &CpuTensor, layer: &LlamaLayerWeights,
  params: ForwardLayerParams, kv_cache: &mut LlamaKvCache) -> Result<LlamaTimedLayerOutput>`).
  `hidden = timed.output;` threads the hidden state layer→layer. **Clean per-layer
  boundary exists.**

- **Sharded / chunk loop** — `src/inference.rs:2122-2144`,
  `forward_layer_range_from_hidden(hidden, start_pos, seq_len)`. Same iteration, but it
  **skips layers outside `self.weights.layer_range`** via `range.contains(&layer_idx)`
  (`src/inference.rs:2123-2127`), using **global** layer indices. Per-layer dispatch is
  `forward_prefill_layer_chunk_timed()`. This is the function the existing distributed
  pipeline already calls per node.

Crucial subtlety for parity: in `forward_layer_range_from_hidden`, **`seq_len == 1`
(decode) is routed to the GPU-resident decode path** (`try_resident_decode_forward`,
`src/inference.rs:2110-2117`) before the CPU layer loop is ever reached. So the "CPU
one-token path" the spec assumes is, on Apple Silicon by default, actually the
GPU-resident path. Any parity gate must pin the lane being compared (CPU vs resident),
or it is comparing two different implementations. This is the single biggest parity
hazard and is called out again in `DECISIONS.md`.

## 2. KV cache structure

- **Llama CPU KV cache** is **monolithic**: `LlamaKvCache { plan, keys: Vec<f32>,
  values: Vec<f32>, allocated_sequence_length, position }`
  (`src/inference/kv_cache.rs:75-81`). Layout is position-major across **all** layers:
  `offset = (((position * layer_count) + layer_idx) * kv_head_count + kv_head) * head_dim`
  (`src/inference/kv_cache.rs:160-171`). A node owning layers `[start,end)` cannot carve a
  contiguous sub-buffer; each position's slice spans every layer. Allocation grows the
  whole `layer_count`-sized buffer at once (`ensure_position_capacity`,
  `kv_cache.rs:114-139`; grow step `CAMELID_KV_CACHE_GROW_TOKENS`). **Not cleanly
  per-layer.** For a sharded node the simplest correct option is to size `plan.layer_count`
  to the **owned range length** (relative slots) and seed/write by relative index — which is
  exactly what the resident path already does (next bullet).

- **GPU-resident sharded KV** is already range-aware: the resident session is built over
  the owned subset — `let range = weights.layer_range.unwrap_or(0..block_count);
  let n_layers = range.len();` (`src/inference.rs:1804-1806`), "resident session is built
  over that subset (relative slots) while KV seeding uses absolute layer ids." So the
  resident lane already sizes KV to the shard, not the whole model.

- **Gemma 4 KV** is natively per-layer: `pub type Gemma4KvCache = Vec<Vec<Vec<f32>>>`
  (`src/gemma4_runtime.rs:561`), `cache[local_layer][position] = [kv_heads*head_dim]`,
  allocated one `Vec` per local layer (`empty_kv_caches`, `gemma4_runtime.rs:742-746`).
  **Cleanly shardable as-is.**

- **Position** is a single global counter per node — `LlamaKvCache.position`
  (`kv_cache.rs:80`), incremented once per token after all owned layers run
  (`forward_layer_range_from_hidden` advances `self.kv_cache.position += seq_len` at
  `inference.rs:2146`). Gemma 4 passes `pos` as a stateless parameter to `step_range`
  (`gemma4_runtime.rs:1060`). Either way, position must travel on the wire (it already
  does — see §4 protocols).

## 3. Activation boundary (shape + dtype)

- Hidden state crossing a layer boundary is **`[1, hidden]` f32** for single-token decode
  (`embedding_lookup` builds `[token_ids.len(), width]`, `src/tensor/mod.rs:1824`; RoPE
  asserts exactly `[1, width]`, `src/inference/rope.rs:114-118`). For a prefill chunk it is
  `[seq, hidden]` f32. Runtime dtype is f32 only (`RuntimeDType::F32`,
  `src/tensor/mod.rs:101-103`; `CpuTensor.data: Vec<f32>`). This matches the spec's wire
  assumption (raw little-endian f32, row-major, length = product(shape)*4).

## 4. Position / state carried between layers — and what already crosses the wire

Within a node, only the hidden vector threads layer→layer; RoPE position is read from the
shared `kv_cache.position` per layer (`inference.rs:4294-4309`), KV is written per-layer at
`offset(layer_idx, position, 0)` (`write_kv_cache`, `inference.rs:17107-17131`), residuals
are layer-local (`inference.rs:4421`, `4598`), and `rope_freqs` + `rms_norm_epsilon` are
shared read-only inputs (`ForwardLayerParams`, `inference.rs:4140-4147`). **Nothing but the
hidden state and the scalar position needs to travel between nodes.**

Across nodes, this already travels. Three wire protocols exist today:
- `src/distributed.rs:10-99` — Llama header (magic, is_prefill, seq_len, position) +
  hand-serialized tensor. **No checksums, no version field.**
- `src/cluster.rs:22-145` — generic framed activation packet (magic `0xDEAD_BEEF`, pos,
  seq_len, float_count, LE-f32 payload, bounds-checked) + token-feedback packet
  (`0xCAFE_FEED`). Tested (`cluster.rs:156-224`) but **not wired into the Llama pipeline**.
- `src/gemma4_distributed.rs:31-94` — versioned (`GEMMA4_WIRE_VERSION=1`), handshake-gated,
  **FNV-1a checksummed** activation protocol. Production-grade.

## Capability inventory vs. the spec's phases (the headline finding)

| Spec phase / artifact | State in repo | Evidence |
|---|---|---|
| Layer-range partition concept | **EXISTS** (as `Option<Range<usize>>`, not a named `LayerPartition` type) | `inference.rs:198`, `:2123-2127`; `load` materializes only owned layers, others are zero-element placeholders (`inference.rs:220`, `:457`) |
| In-process layer split | **EXISTS** functionally (`forward_layer_range_from_hidden`); **no bitwise chained-partition test** | `inference.rs:2094-2148`; tests grep below |
| Transport / wire protocol | **EXISTS ×3** (Llama, generic `cluster.rs`, Gemma4); no `Transport` trait abstraction | §4 above |
| Shard server + coordinator (TCP) | **EXISTS** for Llama (`distribute-worker`/`distribute-master`, `main.rs:1650+/1780+`) and Gemma4 (`gemma4-worker`/`gemma4-master`) | recon audit |
| Per-token pipeline across nodes | **EXISTS** (Llama N-node; Gemma4 two-node) | `main.rs` handlers; `gemma4_distributed.rs` |
| **Parity receipt for a distributed config** | **Gemma4: EXISTS & PASSES vs llama.cpp oracle. Llama: ABSENT.** | `tests/gemma4_distributed_parity.rs` asserts `distributed_greedy_matches_single_node_and_oracle`; `tests/distributed_tests.rs:67-76` asserts only **finite logits + shape**, never token-identity |
| Parity-receipt framework (single-lane) | **EXISTS** (`camelid.parity-receipt/v1`, SHA-256 signed, llama.cpp re-run verify) | `src/receipt/mod.rs`, `src/receipt/verify.rs` |
| Cluster frontend tab | not assessed (Phase 5) | — |

**Interpretation.** For Gemma 4, the spec's one-line success condition (a receipt proving
token-identity to a reference for a too-big model run across nodes) is *already
substantially met* by `gemma4_distributed_parity.rs`. For **Llama**, all the plumbing
exists but the parity gate has never been closed — the existing test proves "it ran," which
the spec explicitly rejects as not-working. So the genuinely missing work is:

1. A **bitwise in-process chained-partition test for Llama** (spec Phase 1), pinning the
   execution lane (CPU vs resident) so the comparison is apples-to-apples.
2. A **distributed parity receipt for the Llama path** in the spec's artifact schema
   (spec Phases 2–4), reusing the existing `receipt` framework rather than inventing one.
3. The **cluster frontend tab** with its experimental/unvalidated banner (spec Phase 5).

Re-implementing the transport, shard servers, or layer-range loop from scratch would
duplicate working, tested code and is explicitly not recommended.

## Existing distributed tests (what they actually assert)

- `tests/distributed_tests.rs` — `test_distributed_pipeline_parallel_inference` asserts
  logits shape `[1,16]` and finiteness only (`:67-76`); `test_network_benchmark`;
  `pipeline_load_puts_output_weights_on_last_node` (output-head ownership regression). **No
  token-identity assertion.**
- `tests/gemma4_distributed_parity.rs` — `distributed_greedy_matches_single_node_and_oracle`,
  `distributed_split_through_shared_kv_block_fails_closed`,
  `distributed_wire_version_mismatch_fails_closed`,
  `distributed_serve_runtime_streaming_matches_oracle`. **Real parity gates.**

## Open questions to resolve at Phase 1 boundary

- Which lane is the parity reference for Llama: CPU chunk path or GPU-resident decode?
  They are different implementations and `seq_len==1` silently picks resident
  (`inference.rs:2110`). Pin one.
- Per-shard Llama CPU KV sizing: monolithic `plan.layer_count` must be set to the owned
  range length (relative slots) for a sharded node, or KV memory does not "add up" cleanly.
  Confirm what the current `distribute-worker` path does for the CPU lane.
