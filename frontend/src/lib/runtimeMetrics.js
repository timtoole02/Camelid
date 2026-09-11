/* Server-reported runtime metrics, and the risk posture of the active lane.

   Two surfaces are served from this module, and they are deliberately different
   in kind.

   THE SAFE ONE reads GET /metrics — the engine's own Prometheus exposition. This
   is server truth, unlike the request timings the Telemetry view measures in the
   browser, and it carries quantities the browser cannot see at all: how often the
   prompt-prefix cache was reused, how the engine's time split between evaluating
   prompts and decoding, how deep the job queue got, real resident memory and VRAM.
   Reading it cannot change anything.

   THE QUARANTINED ONE reads the execution plan already on /v1/health and reports
   WHICH LANE the engine is currently decoding on, and whether that lane is the one
   the model's parity evidence was produced on. It offers no controls, because the
   engine has none to offer: exactly one HTTP route mutates runtime configuration
   (POST /api/runtime/gpu), and the rest of the levers are environment variables
   read once at process start — several latched in a OnceLock, so they cannot
   change after the first forward pass even if something could write them. A toggle
   here would save a value and silently do nothing, which is a worse failure than
   an honest readout.

   THE CENTRAL ARITHMETIC TRAP. Every counter below is CUMULATIVE SINCE PROCESS
   START. A rate derived from two counters is therefore a lifetime average, never a
   current rate — a server that was fast for an hour and has been slow for a minute
   still reports the hour. Nothing here calls a derived value "current", and every
   derived figure is labelled as a lifetime average at the point of use. */

/* Metrics whose absence is meaningful rather than zero. An engine build that does
   not emit one of these should show "not reported", not a confident 0. */
const KNOWN_METRICS = [
  'camelid_http_requests_total',
  'camelid_http_failures_total',
  'camelid_http_request_duration_seconds_sum',
  'camelid_generation_requests_total',
  'camelid_generation_failures_total',
  'camelid_prompt_tokens_total',
  'camelid_decode_tokens_total',
  'camelid_generation_duration_seconds_sum',
  'camelid_prompt_evaluation_duration_seconds_sum',
  'camelid_decode_duration_seconds_sum',
  'camelid_prompt_cache_hits_total',
  'camelid_prompt_cache_misses_total',
  'camelid_weight_cache_hits_total',
  'camelid_weight_cache_misses_total',
  'camelid_engine_queue_depth',
  'camelid_engine_queued_tasks',
  'camelid_engine_active_slots',
  'camelid_engine_active_generated_tokens',
  'camelid_engine_active_elapsed_seconds',
  'camelid_engine_active_stalled_seconds',
  'camelid_process_resident_memory_bytes',
  'camelid_cuda_vram_total_bytes',
  'camelid_cuda_vram_free_bytes',
]

/* Parse Prometheus text exposition.

   The engine emits no labels on any series (verified against a live instance), so
   this handles the unlabelled `name value` form and deliberately IGNORES a labelled
   sample rather than guessing how to aggregate one — silently summing across
   labels would invent a number the engine never reported. */
export function parsePrometheusText(text) {
  const values = new Map()
  const types = new Map()
  const help = new Map()
  for (const rawLine of String(text || '').split('\n')) {
    const line = rawLine.trim()
    if (!line) continue
    if (line.startsWith('#')) {
      const typeMatch = line.match(/^#\s+TYPE\s+(\S+)\s+(\S+)/)
      if (typeMatch) types.set(typeMatch[1], typeMatch[2])
      const helpMatch = line.match(/^#\s+HELP\s+(\S+)\s+(.*)$/)
      if (helpMatch) help.set(helpMatch[1], helpMatch[2])
      continue
    }
    const sample = line.match(/^([A-Za-z_:][A-Za-z0-9_:]*)\s+(.+)$/)
    if (!sample) continue
    const [, name, rawValue] = sample
    if (name.includes('{')) continue
    const value = Number(rawValue.trim())
    if (!Number.isFinite(value)) continue
    values.set(name, value)
  }
  return { values, types, help }
}

const read = (values, name) => (values.has(name) ? values.get(name) : null)

/* A ratio of two counters, or null when the denominator is zero.

   Zero attempts is NOT a 0% hit rate — it is an unknown one, and rendering 0%
   there would report a cache as failing when it was never asked. */
function ratio(hits, misses) {
  if (hits === null || misses === null) return null
  const total = hits + misses
  if (total <= 0) return null
  return hits / total
}

export function summarizeMetrics(text) {
  const { values, help } = parsePrometheusText(text)
  if (!values.size) return null

  const httpRequests = read(values, 'camelid_http_requests_total')
  const httpFailures = read(values, 'camelid_http_failures_total')
  const genRequests = read(values, 'camelid_generation_requests_total')
  const genFailures = read(values, 'camelid_generation_failures_total')
  const promptTokens = read(values, 'camelid_prompt_tokens_total')
  const decodeTokens = read(values, 'camelid_decode_tokens_total')
  const promptSeconds = read(values, 'camelid_prompt_evaluation_duration_seconds_sum')
  const decodeSeconds = read(values, 'camelid_decode_duration_seconds_sum')
  const cacheHits = read(values, 'camelid_prompt_cache_hits_total')
  const cacheMisses = read(values, 'camelid_prompt_cache_misses_total')
  const weightHits = read(values, 'camelid_weight_cache_hits_total')
  const weightMisses = read(values, 'camelid_weight_cache_misses_total')
  const vramTotal = read(values, 'camelid_cuda_vram_total_bytes')
  const vramFree = read(values, 'camelid_cuda_vram_free_bytes')

  const forwardSeconds = promptSeconds !== null && decodeSeconds !== null
    ? promptSeconds + decodeSeconds
    : null

  return {
    /* Raw, exactly as reported. */
    raw: values,
    help,
    missing: KNOWN_METRICS.filter((name) => !values.has(name)),
    counters: {
      httpRequests,
      httpFailures,
      genRequests,
      genFailures,
      promptTokens,
      decodeTokens,
    },
    live: {
      queueDepth: read(values, 'camelid_engine_queue_depth'),
      queuedTasks: read(values, 'camelid_engine_queued_tasks'),
      activeSlots: read(values, 'camelid_engine_active_slots'),
      activeTokens: read(values, 'camelid_engine_active_generated_tokens'),
      activeElapsedSeconds: read(values, 'camelid_engine_active_elapsed_seconds'),
      activeStalledSeconds: read(values, 'camelid_engine_active_stalled_seconds'),
    },
    memory: {
      residentBytes: read(values, 'camelid_process_resident_memory_bytes'),
      /* A ZERO TOTAL IS NOT A READING. `src/api/metrics.rs` writes both CUDA VRAM
         gauges UNCONDITIONALLY, and `HardwareProfile::detect()` leaves them at 0
         whenever `cuda::probe_capability()` returns None — which is every macOS
         host, always. Guarding on presence alone (both gauges ARE present, at 0)
         rendered "VRAM in use: 0 B of 0 B" on every Metal machine: a confident
         zero for a device that was never asked, which is the same mistake the
         prefix-cache rate avoids by returning null on zero attempts.

         Zero FREE is left alone — a fully committed device legitimately reports
         it. Only a zero TOTAL is impossible for a real one. */
      vramTotalBytes: vramTotal !== null && vramTotal > 0 ? vramTotal : null,
      vramFreeBytes: vramTotal !== null && vramTotal > 0 ? vramFree : null,
      vramUsedBytes: vramTotal !== null && vramTotal > 0 && vramFree !== null
        ? Math.max(0, vramTotal - vramFree)
        : null,
    },
    /* Every figure here is a LIFETIME AVERAGE over the process, never a current
       rate. Named `lifetime` so a caller cannot accidentally present it as now. */
    lifetime: {
      promptCacheHitRate: ratio(cacheHits, cacheMisses),
      promptCacheAttempts: cacheHits !== null && cacheMisses !== null ? cacheHits + cacheMisses : null,
      weightCacheHitRate: ratio(weightHits, weightMisses),
      /* The share of forward time spent evaluating prompts rather than decoding.
         On long contexts this is the number that explains where the wall time
         went, and the browser cannot measure it. */
      prefillShare: forwardSeconds !== null && forwardSeconds > 0 ? promptSeconds / forwardSeconds : null,
      promptSeconds,
      decodeSeconds,
      decodeTokensPerSecond: decodeTokens !== null && decodeSeconds !== null && decodeSeconds > 0
        ? decodeTokens / decodeSeconds
        : null,
      httpFailureRate: ratio(httpFailures, httpRequests !== null && httpFailures !== null ? httpRequests - httpFailures : null),
      generationFailureRate: ratio(genFailures, genRequests !== null && genFailures !== null ? genRequests - genFailures : null),
    },
  }
}

/* ---- The quarantined half: which lane is the engine actually on? ---- */

/* Risk classes, in the sense that matters to a user: not "is this fast" but
   "can this hand me a different answer than the evidence was produced with". */
export const LANE_RISK = {
  EVIDENCED: 'evidenced',
  UNEVIDENCED: 'unevidenced',
  UNKNOWN: 'unknown',
}

/* Backends whose decode path carries its own parity evidence in this repo. A
   backend absent from this set is not accused of being wrong — it is reported as
   not carrying an evidence claim on this surface, which is a different statement
   and the only one the frontend can support.

   THIS LIST IS A LIABILITY, and the smoke treats it as one. It is a copy of a
   fact that lives in `src/execution_plan.rs`, and the first version of it was
   itself a copy — of the display-only `METAL_BACKENDS` in `executionPlan.js`,
   which had gone stale at one entry while the planner grew to six. Carried here
   unchanged, that staleness stopped being cosmetic: three lanes that are DEFAULT
   ON macOS (`metal_resident_qwen35_runtime`, `metal_resident_qwen35_kquant_runtime`
   and `metal_resident_lfm2_runtime`, each an opt-OUT) rendered as carrying no
   evidence claim, while their CUDA counterparts rendered as evidenced. Two of the
   three are lanes whose receipts in `src/api/mod.rs` assert the backend string
   itself as the proof — the LFM2 row's evidence reads "the Mac receipt asserts
   selected_backend=metal_resident_lfm2_runtime", and the Ornith agent-eval PASS
   was earned on `metal_resident_qwen35_kquant_runtime`.

   So the smoke extracts every backend the planner can select and asserts each one
   is classified here or in UNEVIDENCED_BACKENDS below. A new lane fails the smoke
   until someone decides which it is, rather than defaulting to "unevidenced"
   silently. */
const EVIDENCED_BACKENDS = new Set([
  'cpu_reference',
  'cpu_q8_runtime_repack',
  'cpu_kquant_block_dot',
  'cuda_resident_q8_runtime',
  'cuda_resident_kquant_runtime',
  'cuda_resident_windowed_runtime',
  'cuda_resident_prism_low_bit_runtime',
  'gemma4_cpu_runtime',
  'gemma4_cuda_resident_runtime',
  'metal_resident_q8_runtime',
  'metal_resident_kquant_runtime',
  'metal_resident_lfm2_runtime',
  'metal_resident_prism_low_bit_runtime',
  'metal_resident_qwen35_runtime',
  'metal_resident_qwen35_kquant_runtime',
])

/* Lanes deliberately left OUT of the set above, so that "not listed" cannot mean
   "nobody looked". The smoke asserts this partition covers the planner. */
export const UNEVIDENCED_BACKENDS = new Set([
  /* The engine itself treats this one as unvalidated: the admission check in
     `src/api/mod.rs` gates on `selected_backend.contains("_runnable_unvalidated")`.
     Claiming evidence for it here would contradict the engine. */
  'cuda_resident_q8_runtime_runnable_unvalidated',
])

/* The GPU row, which is NOT one row with a portable meaning.

   `cuda_resident_active` is a plain `bool` on the plan, so macOS serializes it as
   `false` rather than omitting it — it survives any presence check and rendered a
   "CUDA resident: false" row on Metal hosts with no Metal row anywhere, reading as
   "the GPU is off" while a Metal resident lane was decoding.

   The second half of that row was wrong too. It pointed at Settings → GPU
   acceleration, and that control does not reach Metal: `POST /api/runtime/gpu`
   sets `cuda::set_gpu_accel_enabled` / `set_runtime_enabled`, and every reader of
   those two atoms is a CUDA or BitNet path. The planner's Metal arms gate on
   `platform.metal_available` plus per-lane opt-outs (`CAMELID_METAL_RESIDENT_DECODE`,
   `CAMELID_QWEN35_METAL`, `CAMELID_LFM2_METAL`), none of which the toggle writes.
   So on a Metal host NOTHING here is live-settable, and saying so is the honest
   answer — this panel's whole premise is that a control which does nothing is
   worse than no control.

   Host comes from the plan's own `operating_system`, falling back to the backend
   prefix for an engine too old to report it. */
function acceleratorRow(plan, backend) {
  const metalHost = plan.operating_system === 'macos'
    || (!plan.operating_system && typeof backend === 'string' && backend.startsWith('metal_'))

  if (metalHost) {
    return {
      key: 'metal_resident_active',
      label: 'Metal resident',
      value: typeof backend === 'string' && backend.startsWith('metal_') ? 'true' : 'false',
      changeable: 'restart',
      note: 'Whether weights stay on the GPU. The GPU acceleration toggle in Settings drives the CUDA lane only and does not reach this one; the planner picks it at model load.',
    }
  }

  return {
    key: 'cuda_resident_active',
    label: 'CUDA resident',
    value: plan.cuda_resident_active === null || plan.cuda_resident_active === undefined
      ? null
      : String(plan.cuda_resident_active),
    changeable: 'gpu_toggle',
    note: 'Whether weights stay on the GPU. The GPU setting reaches this; the lane still reselects on the next model load.',
  }
}

export function readActiveLane(runtime) {
  const plan = runtime?.execution_plan
  if (!plan) {
    return { present: false, risk: LANE_RISK.UNKNOWN, rows: [], supportLevel: null, reasons: [] }
  }
  const backend = plan.selected_backend || null
  const risk = backend === null
    ? LANE_RISK.UNKNOWN
    : EVIDENCED_BACKENDS.has(backend)
      ? LANE_RISK.EVIDENCED
      : LANE_RISK.UNEVIDENCED

  /* Each row names the lever, its current value, and — the part that matters —
     whether it can be changed without restarting the engine. Exactly one of these
     is live-settable; saying so plainly is the whole point. */
  const rows = [
    {
      key: 'selected_backend',
      label: 'Decode backend',
      value: backend,
      changeable: 'restart',
      note: 'Chosen by the planner from the model, the host and the GPU setting.',
    },
    {
      key: 'decode_path',
      label: 'Decode path',
      value: plan.decode_path || null,
      changeable: 'restart',
      note: 'The kernel lane each generated token goes through.',
    },
    {
      key: 'prefill_path',
      label: 'Prefill path',
      value: plan.prefill_path || null,
      changeable: 'restart',
      note: 'How the prompt is evaluated before the first token.',
    },
    acceleratorRow(plan, backend),
    {
      key: 'q8_policy',
      label: 'Q8 loader policy',
      value: runtime?.q8_runtime?.policy || null,
      changeable: 'restart',
      note: 'The effective loader path, which outranks the environment flag that requests it.',
    },
    {
      key: 'thread_count',
      label: 'Worker threads',
      value: plan.thread_count === null || plan.thread_count === undefined ? null : String(plan.thread_count),
      changeable: 'restart',
      note: 'Set at startup with --threads.',
    },
  ].filter((row) => row.value !== null)

  return {
    present: true,
    risk,
    backend,
    rows,
    supportLevel: plan.support_level || null,
    exactRow: plan.exact_model_row || null,
    /* The planner's own explanation of why it chose this lane. Rendering it beats
       any summary this module could write, because it is the engine's reasoning
       rather than the browser's guess at it. */
    reasons: Array.isArray(plan.reasons) ? plan.reasons : [],
  }
}

export const LANE_RISK_COPY = {
  [LANE_RISK.EVIDENCED]: {
    label: 'Evidenced lane',
    detail: 'The engine is decoding on a backend that carries parity evidence in this repository. That is a statement about the lane, not about this particular reply.',
  },
  [LANE_RISK.UNEVIDENCED]: {
    label: 'Lane without an evidence claim here',
    detail: 'The engine is decoding on a backend this page cannot match to a parity-checked lane. That does not mean the output is wrong — it means this surface will not claim it was produced on an evidenced path.',
  },
  [LANE_RISK.UNKNOWN]: {
    label: 'Lane not reported',
    detail: 'No execution plan is available, usually because no model is loaded. Nothing is claimed either way.',
  },
}

/* ---- Formatters ---- */

export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`
}

/* A share, rendered as a percentage — legitimate here because these ARE
   proportions of a whole, unlike a similarity score. */
export function formatShare(share) {
  if (share === null || share === undefined || !Number.isFinite(share)) return '—'
  if (share > 0 && share < 0.001) return '<0.1%'
  return `${(share * 100).toFixed(1)}%`
}

export function formatCount(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value.toLocaleString()
}

export function formatSeconds(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  if (value < 1) return `${(value * 1000).toFixed(0)} ms`
  if (value < 120) return `${value.toFixed(1)} s`
  return `${(value / 60).toFixed(1)} min`
}

export function formatRate(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${value.toFixed(1)}/s`
}
