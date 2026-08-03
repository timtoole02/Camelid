# Windows K-quant roadmap on Apple Metal

Implementation result for the five-item Windows-to-Metal roadmap. The
same-host receipt is accepted for Camelid's CLI/server appliance policy:
qualified Q4_K/Q6_K models use the Metal path automatically on Apple Silicon,
while library embedders retain conservative defaults for every *Metal* gate.
The one appliance default that is NOT library-conservative is the two-slot
cooperative streaming scheduler — see item 5.

## What landed

1. **Resident Q4_K/Q6_K decode.** Metal consumes GGUF wire weights directly,
   quantizes activations to the same Q8_K representation as the CPU/CUDA
   oracle, and dispatches Q4_K or Q6_K tiled kernels for attention, FFN, and
   the output projection. Token embedding uses format-specific GPU gather
   kernels. Mixed Q4_K_M models may switch format per tensor.
2. **Parity and performance gates.** The Q8_K quantizer is compiled separately
   with Metal fast math disabled so its scales and integer codes match the
   CPU/CUDA oracle. Unit tests cover both K-quant formats before an
   end-to-end benchmark is allowed to count.
3. **Metal-native K-quant prefill.** Batched rows use the same resident wire
   weights and a four-token tile, amortizing each weight tile across prompt
   rows. Existing Q8_0 simdgroup-matrix prefill remains unchanged.
4. **Compressed resident KV.** F16-primary and Q8_0-primary caches support
   prefill scatter and decode attention without maintaining an F32 primary
   copy. Q8 cache rows use 32-value blocks with a half scale and signed bytes.
5. **Continuous streaming batches.** The production engine can retain
   multiple streaming sessions and rotate them one token step at a time on
   its sole compute thread. A lone session retains single-request encode-ahead;
   contended rounds stop enqueueing future session-local graphs so one stream
   cannot head-of-line block another on the shared command queue. Unlike the
   Metal items above, the two-slot default is **not** macOS-scoped: it lives in
   `runtime_config` and therefore also reaches Linux/Windows embedders that
   call `EngineHandle::spawn()`. Two invariants keep that safe: exclusive
   engine work (model load/unload, non-streaming completions, resident-cache
   resets, the parity probe) is dequeued in post order rather than waiting for
   the streaming batch to drain, and the bounded engine channel is never
   drained into an unbounded local queue, so the typed `QueueFull` -> 503 is
   still reachable while streams are running. Runs driven by the CUDA resident
   engine stay run-to-completion, because that engine — unlike Metal's
   per-session `ResidentDecodeState` — is a process-global slot keyed by model
   id and cannot host two interleaved sequences.

## Rollout controls

| Variable | Values | Default | Effect |
|---|---|---:|---|
| `CAMELID_METAL_KQUANT` | `0`, `1` | `1` in the macOS CLI; library default `0` | Admit resident Q4_K/Q6_K weights and native K-quant prefill/decode. Unsupported mixes fall back. |
| `CAMELID_METAL_NOCOPY` | `0`, `1` | `1` in qualified macOS serve/bench runs | Read Q8_0/Q4_K/Q6_K weights into page-aligned storage which Metal wraps without a second upload. |
| `CAMELID_METAL_KV_DTYPE` | `f32`, `f16`, `q8` | `f16` for K-quant; `f32` otherwise | Select the resident KV primary representation. The default follows the LOADED MODEL's weights, not the `CAMELID_METAL_KQUANT` gate: a Q8_0 model keeps its F32 cache (and with it the split-K decode attention and the attention-as-matmul prefill, both of which require an F32 primary) even with K-quant admission enabled. |
| `CAMELID_METAL_KV16` | `0`, `1` | `0` | Legacy alias for `CAMELID_METAL_KV_DTYPE=f16`. |
| `CAMELID_CONTINUOUS_BATCH_SLOTS` | `1..256` | `2` | Maximum active cooperative streaming sessions. Set `1` for legacy run-to-completion scheduling. |
| `CAMELID_PREFIX_CACHE_RESIDENT` | `0`, `1` | auto (`0` on hosts with ≤8 GiB RAM, otherwise `1`) | Let a GPU-resident session mirror its KV back so the prompt-prefix cache can store it. `0` keeps this lane's CPU KV at zero bytes and accepts a full re-prefill on every repeated prompt. An explicit value overrides the host-memory policy. Only ever engages when the round trip is bit-exact (F16 resident primary + non-quantized CPU KV). |

`--deterministic` forces `CAMELID_METAL_KQUANT=0` and
`CAMELID_METAL_KV_DTYPE=f32` along with the rest of the GPU-off policy.

## Fail-closed boundaries

- Metal K-quant admission requires every resident dense projection to be
  Q8_0, Q4_K, or Q6_K with a valid aligned wire layout. Unsupported K-quants
  stay on their existing CPU/CUDA route.
- Q4_K/Q6_K input dimensions must be multiples of 256.
- Q8 KV requires a head dimension divisible by 32 and no larger than 128.
- Tree verification currently falls back when a compressed primary KV cache
  is selected; linear speculative verification supports compressed KV.
- Continuous batching applies only to streaming generation. Non-streaming
  work and management jobs remain exclusive, and run ahead of the streaming
  batch in post order rather than behind it.
- Cooperative streaming jobs seed from the prompt-prefix cache exactly as the
  run-to-completion job does — both call `stream_prompt_cache_prologue`.
- The prompt-prefix cache now also covers the GPU-resident lane.
  `store_prompt_prefix_cache` requires `cpu_kv_authoritative()`, and a resident
  prefill advances `kv_cache.position` while leaving the CPU buffers empty, so
  this lane used to store nothing at all and every repeated or growing prompt
  re-prefilled from scratch. `prepare_for_prompt_prefix_cache` mirrors the GPU
  history back at store time, and a later resume re-seeds a fresh
  `ResidentDecodeState` from it. Gated on the round trip being BIT-EXACT in both
  directions: the resident cache must be F16-primary (queried on the session's
  own engine, not the process-global format) and the CPU cache must be F32/F16
  rather than `--kv-quant q8_0|q4_0`. That means Q4_K/Q6_K models are covered
  and **Q8_0 models are not** — their F32-primary resident cache would be
  silently f16-rounded on the way out, the same hazard that makes the streaming
  path bypass this cache entirely under the CUDA resident engine.
  `CAMELID_PREFIX_CACHE_RESIDENT=0` opts out; the mirror takes this lane's CPU
  KV from zero bytes to full size and `store_prompt_prefix_cache` then clones
  it, so a cached entry costs roughly two CPU KV copies of one prompt (the pool
  holds one entry by default and `ensure_position_capacity` still enforces the
  session's KV budget).
  Measured on an Apple M4, Llama-3.2-1B-Instruct-Q4_K_M, zero configuration,
  the same 1974-token turn repeated three times: **33.23s / 33.33s / 33.54s**
  before, **33.35s / 0.50s / 0.49s** after, with byte-identical greedy
  completions across cold and cached turns.
- `/props.total_slots`, the `GET /slots` array length, and `fail_on_no_slot=1`
  all answer from one number, `EngineHandle::total_slots`. It is the cooperative
  capacity except on the CUDA resident lane, where `stream_completion` runs every
  stream exclusive and the honest answer is `1`. `fail_on_no_slot` refuses only
  when every slot is busy, so a second stream is admissible while the first is
  mid-generation; an exclusive job (model load/unload, a non-streaming
  completion, a cache reset) saturates all slots, because it owns the engine
  while it runs and no slot can produce a token until it returns. Per-slot task
  identity and per-slot progress are engine-wide values repeated on the busy
  entries, declared as `per_slot_task_identity` / `per_slot_progress` in the
  route's `unsupported` list rather than quietly implied.
- The plan only labels a model `metal_resident_kquant_runtime` when its tensor
  types are an allow-listed Q4_K/Q6_K/F32/F16/BF16 mix AND its architecture is
  one the resident dense kernels can express. Everything else — including
  tensor types this GGUF reader does not model — keeps the CPU block-dot
  label.
- Q5_K/Q2_K/Q3_K/IQ4_XS mixes are not labeled or admitted as Metal K-quant;
  they keep their existing wire-only CPU/CUDA routes.

## Post-review verification

`PERF_RECEIPTS/same-host/metal-kquant-m4-postfix-three-way-20260730.json` is an
independent re-measurement on the same M4, with three release binaries built
from the same toolchain — the merge base, this PR as first submitted, and this
PR after the review fixes — run back to back with no `CAMELID_*` variables set.
Median of five after one warmup; the full generated token-ID list is recorded
for every arm.

| Probe | merge base | PR as submitted | PR after fixes |
|---|---:|---:|---:|
| Q4_K_M, 6-token prompt, 50 tokens | 33.77 tok/s / 2.781 GB | 63.62 / 0.970 | **62.75 / 0.971** |
| Q4_K_M, 1974-token prompt, 64 tokens | 13.89 tok/s / 2.963 GB | 42.69 / 1.002 | **42.79 / 1.002** |
| Q8_0, 1974-token prompt, 64 tokens | 64.46 tok/s / 1.608 GB | 47.74 / 1.528 | **66.04 / 1.621** |

Generated token IDs are identical across all three arms in all three probes.

The Q8_0 row is why the fix pass exists: keying the F16-primary KV default off
the `CAMELID_METAL_KQUANT` env gate (which the macOS CLI now sets for every run)
rather than off the loaded model's weights moved curated Q8_0 rows onto a half
KV cache, which disables the split-K decode attention and the
attention-as-matmul prefill — 25.9% slower decode and 2.28x slower prefill at
2k context. On the K-quant lane itself the fix pass is a wash (-1.4% on the
short probe, inside this host's run-to-run spread).

## Merge gate

Run these in a release profile on the same Mac and exact GGUF:

1. `metal_kquant_resident_projection_matches_cpu_oracles`
2. `metal_q8_primary_kv_scatter_and_attention_match_dequantized_reference`
3. `metal_attention_decode_splitk_kv16_matches_cpu_reference`
4. `cooperative_jobs_interleave_one_step_per_round`
5. the full library suite
6. a cold, greedy, median-of-five `bench-generate` before/after comparison

The benchmark receipt must record the commit, host, model path and hash,
prompt, environment, raw iterations, medians, and generated token IDs. Where a
receipt records a `token_ids_sha256` digest instead of (or alongside) the raw
IDs, the recipe is `sha256(",".join(str(id) for id in output_token_ids))` over
the `output_token_ids` array that `bench-generate` emits — verifiable against
any recorded ID list.
The merge receipt records the first generated-token divergence, if any.
F16-primary must pass the predeclared confident probe; Q8-primary is explicitly
lossy and is compared against the dequantized-Q8 oracle rather than claimed as
token-identical to F16/F32. A speed tie or regression is reported as such and
leaves the new path default-off. Continuous batching additionally requires a
live two-client streaming probe: both sessions must emit all requested tokens,
alternate after admission, and complete without a shared-queue stall.
