# Backend asks — endpoint/contract needs discovered during the frontend overhaul

The UI never stubs fake data for these; each ask names the surface that stays
guarded until the contract grows.

## 1. Sampling-parameter capability rows (Phase 2, 2026-06-12)

**Surface waiting on it:** the chat "Generation controls" drawer renders every
sampling parameter (temperature, top_p, top_k, stop, seed) as a guarded
read-only row because `/api/capabilities` `api_features` advertises no sampling
rows at all. Chat keeps sending greedy `temperature: 0` — the lane the parity
evidence covers.

**Ask:** advertise one feature row per parameter the backend actually honors on
`/v1/chat/completions`, with the usual status vocabulary, e.g.:

```json
{
  "id": "sampling_temperature",
  "status": "supported_current_gate",
  "notes": "temperature in [0,2] honored for streaming + non-streaming chat completions; evidence row-scoped to the current gate"
}
```

The frontend already resolves rows by exact id (`sampling_<param>` or
`<param>`, no resemblance matching — `lib/samplingContract.js`) and will unlock
the matching control, persist last-used values per model id, and merge the
override into the request body only while the row stays supported. No frontend
change needed when the rows appear.

**Not asked:** any claim that sampled output has parity evidence. If sampled
lanes need their own evidence categories, that belongs in the compatibility
rows, not the feature row notes.

## 2. Evidence-bundle manifest references on compatibility rows (Phase 4, 2026-06-12)

**Surface waiting on it:** the Compatibility ledger's per-row evidence checklist cites
what the contract exposes today — the `*_pack_id` identifiers (e.g.
`tinyllama-context-512-smoke-v1`). The repo's qa/evidence-bundles manifests
(README/COMPATIBILITY.md reference them by path) are not addressable from
`/api/capabilities`, so chip popovers can name a pack id but cannot cite its manifest.

**Ask:** add an optional manifest reference per evidence lane, e.g.

```json
{
  "bounded_context_512_pack": "validated_bounded_pack",
  "bounded_context_512_pack_id": "tinyllama-context-512-smoke-v1",
  "bounded_context_512_pack_manifest": "qa/evidence-bundles/tinyllama-context-512-.../manifest.json"
}
```

Repo-relative paths only (no absolute filesystem paths — the frontend will render them
as citations, and I7 keeps absolute paths out of shareable surfaces). The ledger picks
up `*_pack_manifest` fields automatically once they appear.

## 3. System memory + KV-cache cost for the response-length control (Phase 9, 2026-06-12)

**Surface waiting on it:** Settings → Response length renders its memory ceiling
marker and projected-memory gauge ABSENT (with an explanatory line) because neither
input exists on the API. The frontend will not estimate RAM client-side or invent KV
math from assumed dtypes.

**Ask (exact fields, units = bytes):**
1. `GET /api/system/memory` → `{ "total_bytes": u64, "available_bytes": u64,
   "process_rss_bytes": u64 }` (or fold the same fields into `/v1/health`).
2. On `/api/models/current` (and ideally `/v1/models` meta):
   `"kv_bytes_per_token": u64` for the loaded runtime configuration — or, if
   preferred, `"kv_cache_dtype": "f16" | "f32" | ...` so the frontend can combine it
   with the GGUF block_count / head_count_kv / key_length / value_length already
   exposed. Also useful: `"kv_cached_tokens": u32` (current cache occupancy) so the
   projection can subtract already-resident tokens.

Once present, the control renders: projected = process_rss_bytes +
(value − kv_cached_tokens) × kv_bytes_per_token vs available_bytes, labeled
"estimated", with red above available RAM and amber above 85% — formula shown in the
readout's popover.

## 4. Runnable-lane HTTP serve/generate endpoint (Models tab Gate 4, 2026-06-17)

**Surface waiting on it:** Models tab → "Compatible" lane rows (smoke-admitted, runnable
f32 lane). These models have a runnable receipt proving deterministic execution, but the
UI offers NO in-app "Use for chat" for them — only the Supported lane gets a load button
(it loads into the parity chat backend via `POST /api/models/load`). The runnable lane is
a separate generic-f32 engine with only `POST /api/models/runnable-smoke` (one-shot smoke)
and the `camelid runnable-smoke` CLI. There is no interactive serve/generate route, so the
frontend cannot — and will not fake — a chat session against the runnable lane.

**Ask (one of):**
1. `POST /api/models/runnable-generate` → `{ filename, prompt, max_tokens, ... }` returning
   `{ tokens, text }` from the runnable engine (stateless or KV-cached), OR
2. let `POST /api/models/load` accept `{ lane: "runnable" }` so a runnable-only model can be
   loaded into a runnable serving context and reuse `/v1/completions` with an explicit
   `execution_lane: "runnable"` echoed on every response (so the chat UI can label it amber,
   never copper, and keep the parity-locked Send-gate off for it).

Until then the Compatible rows stay receipt-only with an explicit "CLI only — no HTTP serve
yet" note; membership and evidence remain fully derived, nothing is invented.

## 5. Logprobs on the streaming path (Token Inspector, 2026-09-06)

**Surface waiting on it:** the Token Inspector composer toggle
(`lib/tokenInspection.js`, `components/chat/render/TokenInspector.jsx`). Because
`rich_logprobs.unsupported_modes` contains `streaming`, an inspected turn must be
sent with `stream: false`, so the user gives up token-by-token rendering for that
reply and must decide to inspect BEFORE sending. Inspecting afterwards is not an
option we are willing to take: it would mean decoding a second time, and those
numbers would describe a different generation than the one on screen.

**Ask:** carry the per-token record on the SSE path — either a `logprobs` object on
each `choices[0].delta` (one entry per emitted token), or one terminal chunk
carrying the whole `content[]` array in the same shape the non-streaming response
already uses. Either shape collapses the pre-send decision into an ordinary
disclosure of data already in hand, and the frontend keeps streaming.

**Not asked:** logprobs together with `n > 1`. Those are mutually exclusive by
contract and the UI treats the candidates surface as guarded, not pending.

## 6. Logprobs fail OPEN on four serve lanes (engine defect, 2026-09-06)

**Not a frontend ask — a backend correctness report.**

`chat_completions` short-circuits into the gemma4 (`src/api/mod.rs:18011`), runnable
(`:18063`) and DiffusionGemma (`:18077`) lanes BEFORE `validate_choice_and_logprob_fields`
runs. On those lanes a request carrying `logprobs: true` returns **HTTP 200 with the
`logprobs` key simply absent** — no error, no typed refusal. Every other invalid
logprobs combination on the dense lane is a typed 400, so this one path fails open
while its neighbours fail closed.

That matters because the missing key is indistinguishable, to a client, from a lane
that genuinely reported nothing — and the honest reading of "no distribution" is
easy to confuse with "a flat or certain distribution". The Token Inspector handles
it defensively (`inspectionAbsenceReason`, `code: 'lane_absent'`) and says the
measurement is missing rather than flat, but the engine should refuse rather than
leave every client to detect this independently.

**Ask:** move the logprobs validation ahead of lane dispatch, and return the same
typed 400 these lanes already return for other unsupported combinations — or add a
lane axis to `api_conformance` so the refusal is at least machine-readable. Note
that `unsupported_modes` today covers only the ROUTE axis (`streaming`,
`multi_choice`) and says nothing about which serve lanes can honour the request.
## 7. Host RAM total/available on an HTTP surface (Wave C, 2026-09-06)

**Surface waiting on it:** Settings → Response length still renders its memory
ceiling and projected-memory gauge ABSENT (see ask 3, 2026-06-12), and the new
System → Engine metrics panel can show resident memory and VRAM but cannot say how
close either is to the host's ceiling.

**Ask 3 is now MOSTLY satisfied** and worth re-reading: `GET /api/runtime/memory`
already returns `process_resident_bytes`, `model_weight_bytes_estimate`,
`kv_cache_bytes`, `kv_cache_entries`, `kv_cache_capacity` and a per-model breakdown
carrying `cached_tokens` — which is more than that ask requested, and enough to
derive KV bytes per token once the cache is non-empty.

**What is still missing is one field:** host total and available RAM. The engine
ALREADY COMPUTES IT — the startup hardware probe prints
`RAM 4.9 GiB free / 15.7 GiB total` — it simply is not on any HTTP surface. Neither
`/v1/health`, `/api/capabilities`, `/api/runtime/memory` nor `/metrics` carries it.

**Ask:** add `host_total_bytes` and `host_available_bytes` to
`GET /api/runtime/memory` (and/or `camelid_host_memory_total_bytes` /
`camelid_host_memory_available_bytes` gauges on `/metrics`, matching the existing
`camelid_cuda_vram_total_bytes` / `_free_bytes` pair). No new measurement is needed;
this is exposing a value the process already has.

**Not asked:** any per-model prediction. The frontend will not estimate RAM
client-side; it wants the ceiling so it can render a real proportion instead of
omitting the gauge.

## 8. The active value of runtime levers, over HTTP (Wave C, 2026-09-06)

**Surface waiting on it:** System → Active configuration reports the lane the engine
is on using `execution_plan` plus `q8_runtime`, which is honest but partial. It
cannot report the ACTIVE value of the levers that most affect output — the KV dtype,
the flash-prefill and blocked-dot gates, the speculative-decode settings — because
no endpoint discloses them by name.

**The data shape already exists in this codebase.** `eagle3_effective_env()` and
`speculative_effective_env()` build exactly such a map, and the CLI receipt structs
carry `effective_env` and `planner_env_updates` fields. They are simply never served.

**Ask:** a read-only `GET /api/runtime/config` returning the effective values of the
levers the engine actually consulted, e.g.

```json
{ "effective": { "CAMELID_KV_QUANT": "f32", "CAMELID_FLASH_PREFILL": "0" },
  "latched": ["CAMELID_ATTENTION_F32_BLOCKED_DOT"],
  "planner_managed": ["CAMELID_QWEN35_CUDA"] }
```

`latched` matters as much as the values: several gates are read once into a
`OnceLock`, so a UI must be able to say "this cannot change without a restart"
rather than implying it is editable. `planner_managed` matters because those keys
are written by the planner itself at every model load — presenting one as a user
setting would invite a user to fight the planner.

**Not asked:** any route that WRITES these. The levers are read at process start and
several are latched; a write route would be a control that silently does nothing,
which is worse than no control. If they ever become settable, that is a separate
engine change with its own evidence.
