# CUDA Continuous Batching and Model Residency

Status: implemented, explicit opt-in, exact validated envelope only.

This runtime project lets one resident CUDA engine serve multiple independent sequences and, separately, retain up to two main models on the GPU. It does not widen Camelid's model-support ledger: every model and hardware claim remains bounded by [COMPATIBILITY.md](../COMPATIBILITY.md).

## Defaults

Without these settings, Camelid retains the conservative CUDA behavior:

- one CUDA sequence;
- one resident main model;
- no paged true-batch dispatch;
- no cross-request batched prefill.

The ordinary two-stream cooperative scheduler on CPU and Metal is separate and remains controlled by `CAMELID_CONTINUOUS_BATCH_SLOTS`.

## Validated CUDA configuration

The Phase 7/8 Windows validation used:

```powershell
$env:CAMELID_CUDA_SEQUENCE_SLOTS = '8'
$env:CAMELID_CUDA_PAGED_KV = '1'
$env:CAMELID_CUDA_TRUE_BATCH = '1'
$env:CAMELID_CUDA_BATCHED_PREFILL = '1'
$env:CAMELID_CUDA_BATCHED_PREFILL_ROUND_TOKENS = '4'
$env:CAMELID_CUDA_RESIDENT_MODELS = '2'
$env:CAMELID_CUDA_RESIDENT_MAX_CONTEXT = '512'
camelid serve --gpu on --model C:\models\Llama-3.2-1B-Instruct-Q8_0.gguf
```

Accepted values are deliberately narrow:

| Setting | Accepted values | Default / invalid value |
| --- | --- | --- |
| `CAMELID_CUDA_SEQUENCE_SLOTS` | `1`, `2`, `4`, `8` | one sequence unless the legacy two-slot gate is enabled |
| `CAMELID_CUDA_PAGED_KV` | boolean `1`, `true`, `on`, `yes` | off |
| `CAMELID_CUDA_TRUE_BATCH` | boolean `1`, `true`, `on`, `yes` | off |
| `CAMELID_CUDA_BATCHED_PREFILL` | boolean `1`, `true`, `on`, `yes` | off |
| `CAMELID_CUDA_BATCHED_PREFILL_ROUND_TOKENS` | `1`, `4`, `16` | `16` |
| `CAMELID_CUDA_RESIDENT_MODELS` | `1`, `2` | `1` |
| `CAMELID_CUDA_RESIDENT_MAX_CONTEXT` | integer at least `256` | VRAM-sized runtime limit |

Sequence capacities above two require paged KV. True dynamic batching requires paged KV and `CAMELID_CUDA_TRUE_BATCH=1`. Cross-request prefill additionally requires `CAMELID_CUDA_BATCHED_PREFILL=1`. Unsupported architectures or configurations stay on existing fallback/refusal paths and are not labeled as true-batch successes.

## Safety and lifecycle

- The engine worker remains the only owner of resident GPU mutation.
- Sequence leases and generation counters isolate KV state.
- Paged append uses prepare, device allocation, kernel execution, and commit/abort transitions.
- Active resident models are pinned and cannot be evicted.
- Idle eviction is deterministic least-recently-used order.
- A model's arena identity binds its model ID and exact GGUF SHA-256 plus numerical/backend configuration.
- Per-model unload refuses with `409 model_operation_in_progress` while that model is active, then rechecks under engine ownership before registry mutation.
- The speculative drafter cache remains separate from the main-model arena.

## Observability

The aggregate arena object has these fields:

```json
{
  "capacity_models": 2,
  "resident_models": 2,
  "active_models": 0,
  "evictions": 0,
  "admission_failures": 0
}
```

It appears at:

- `GET /v1/health` and `GET /health` as `cuda_resident_arena`;
- `GET /api/runtime/memory` as `cuda_resident_arena`, with `cuda_resident` and `cuda_active` on each loaded model;
- `GET /props` under `camelid.cuda_resident_arena`;
- every `GET /slots` entry under `camelid.cuda_resident_arena`.

Prometheus exposes bounded, model-label-free metrics:

- `camelid_cuda_resident_model_capacity`
- `camelid_cuda_resident_models`
- `camelid_cuda_resident_active_models`
- `camelid_cuda_resident_model_evictions_total`
- `camelid_cuda_resident_model_admission_failures_total`

The WebUI Model Arena preserves both loaded models only when live health reports capacity two. Older servers, non-CUDA hosts, and default capacity one retain sequential replacement semantics. Analytics → Runtime memory shows aggregate and per-model residency.

## Rollback

Restore conservative behavior by stopping the server, removing the Phase 7/8 variables, and restarting it. Setting the two capacity controls to `1` is also fail-closed:

```powershell
$env:CAMELID_CUDA_SEQUENCE_SLOTS = '1'
$env:CAMELID_CUDA_RESIDENT_MODELS = '1'
$env:CAMELID_CUDA_PAGED_KV = '0'
$env:CAMELID_CUDA_TRUE_BATCH = '0'
$env:CAMELID_CUDA_BATCHED_PREFILL = '0'
```

Existing resident state is process-local and disappears on restart. No model file or conversation data is migrated by this feature.

## Validated boundary

The retained physical gate used an NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9, 8 GiB VRAM), CUDA/NVRTC 12.9, and these exact artifacts:

- `Llama-3.2-1B-Instruct-Q8_0.gguf`, SHA-256 `3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`;
- `Llama-3.2-3B-Instruct-Q5_K_M.gguf`, SHA-256 `0b94ccd04d908304cec5246a3d942b64417a423bc5c6d47c73bc557e590b5194`.

It proved A-B-A-B deterministic generation with two resident engines and zero eviction; typed active-unload refusal; target-only idle unload; survivor reuse; and exact same-model batch-eight regression. The committed Phase 9 bundle adds API/UI/normal/edge-case visual evidence. These results do not claim arbitrary models, other GPUs or drivers, context above 512 for this project gate, MoE/windowed/sharded batching, portable throughput, or default enablement.
