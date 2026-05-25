#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, stat, statfs, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import os from 'node:os'
import { dirname, resolve } from 'node:path'

import { renderExpectedPrompt, resolveReferenceContext } from './lib/chat-parity-harness.mjs'

const args = parseArgs(process.argv.slice(2))
const backendBase = (args.get('backend') || process.env.CAMELID_API_BASE || 'http://127.0.0.1:8181').replace(/\/$/, '')
const llamaBase = (args.get('llama-url') || process.env.LLAMA3_LLAMA_SERVER_URL || 'http://127.0.0.1:8183').replace(/\/$/, '')
const modelPath = resolve(args.get('model') || process.env.LLAMA3_GGUF || 'models/Llama-3.2-3B-Instruct-Q8_0.gguf')
const modelId = args.get('model-id') || process.env.LLAMA3_MODEL_ID || 'llama32-3b-q8'
const rowId = args.get('row-id') || process.env.CAMELID_BENCH_ROW_ID || 'llama32_3b_instruct_q8_0'
const backendBin = resolve(args.get('backend-bin') || process.env.CAMELID_BIN || 'target/release/camelid')
const llamaServerBin = resolve(args.get('llama-server') || process.env.LLAMA3_LLAMA_SERVER || 'target/reference/llama.cpp/build/bin/llama-server')
const renderMode = args.get('render-mode') || process.env.LLAMA3_CHAT_RENDER_MODE || 'compact'
const maxTokens = parsePositiveInt(args.get('max-tokens') || process.env.CAMELID_BENCH_MAX_TOKENS || '16', 'max-tokens')
const repeats = parsePositiveInt(args.get('repeats') || process.env.CAMELID_BENCH_REPEATS || '3', 'repeats')
const warmup = parseNonNegativeInt(args.get('warmup') || process.env.CAMELID_BENCH_WARMUP || '1', 'warmup')
const out = args.get('out') || process.env.CAMELID_SAME_HOST_BENCH_OUT
const startBackend = args.get('start-backend') !== 'false'
const startLlamaServer = args.get('start-llama-server') !== 'false'
const waitMs = parsePositiveInt(args.get('wait-ms') || process.env.CAMELID_BENCH_WAIT_MS || '600000', 'wait-ms')
const explicitLlamaContext = parseOptionalPositiveInt(args.get('llama-context') || process.env.LLAMA3_LLAMA_CONTEXT, 'llama-context')
const threads = parseOptionalPositiveInt(args.get('threads') || process.env.CAMELID_BENCH_THREADS, 'threads')
const requireMarker = args.has('require-marker') || process.env.CAMELID_BENCH_REQUIRE_MARKER === '1'
const expectedMarker = args.get('expected-marker') || process.env.CAMELID_BENCH_EXPECTED_MARKER || 'CMLD-BENCH'
const expectedCamelidQ8Policy = args.get('expected-camelid-q8-policy') || process.env.CAMELID_BENCH_EXPECTED_CAMELID_Q8_POLICY || null
const uniquePrompt = args.has('unique-prompt') || process.env.CAMELID_BENCH_UNIQUE_PROMPT === '1'
const minMarkerTokens = parsePositiveInt(process.env.CAMELID_BENCH_MIN_MARKER_TOKENS || '8', 'CAMELID_BENCH_MIN_MARKER_TOKENS')

if (args.has('help') || args.has('h')) {
  console.log(usage())
  process.exit(0)
}

if (requireMarker && maxTokens < minMarkerTokens) {
  throw new Error(`--require-marker needs --max-tokens >= ${minMarkerTokens}; got ${maxTokens}. Lower budgets can truncate ${JSON.stringify(expectedMarker)} before the guard can pass.`)
}

const benchmarkMessages = [
  { role: 'system', content: 'You are Camelid benchmark mode. Reply with the exact requested text and nothing else.' },
  { role: 'user', content: 'Reply with exactly this single line and nothing else: CMLD-BENCH' },
]

const expectedPrompt = renderExpectedPrompt(benchmarkMessagesForRun('plan'), renderMode)
const estimatedPromptTokens = estimatePromptTokens(expectedPrompt)
const llamaContext = resolveReferenceContext({
  promptTokenCount: estimatedPromptTokens,
  maxTokens,
  explicitContext: explicitLlamaContext,
})
const evidenceContext = await collectEvidenceContext({ includeFileHashes: !args.has('print-plan') })
const preStartSnapshot = await captureResourceSnapshot('pre_start')

if (args.has('print-plan')) {
  const plan = buildPlan()
  printPlan(plan)
  if (out) {
    const outPath = resolve(out)
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, `${JSON.stringify(plan, null, 2)}\n`)
    console.log(`json_out=${outPath}`)
  }
  process.exit(0)
}

let backendChild = null
let llamaChild = null
let backendSpawnError = null
let llamaSpawnError = null
let backendStartupMs = null
let llamaStartupMs = null
let camelidModelLoadMs = null
let backendHealthReady = null
let backendHealthAfterLoad = null

try {
  const backendStart = performance.now()
  if (startBackend) {
    const url = new URL(backendBase)
    backendChild = spawn(backendBin, ['serve', '--addr', `${url.hostname}:${url.port || '8181'}`], { stdio: ['ignore', 'pipe', 'pipe'] })
    backendChild.once('error', (err) => { backendSpawnError = err })
    backendChild.stdout.on('data', (chunk) => process.stderr.write(`[camelid] ${chunk}`))
    backendChild.stderr.on('data', (chunk) => process.stderr.write(`[camelid] ${chunk}`))
  }

  const llamaStart = performance.now()
  if (startLlamaServer) {
    const url = new URL(llamaBase)
    const llamaArgs = ['--host', url.hostname, '--port', url.port || '8183', '-m', modelPath, '-ngl', '0', '-c', String(llamaContext), '--no-warmup']
    if (threads) llamaArgs.push('-t', String(threads))
    llamaChild = spawn(llamaServerBin, llamaArgs, { stdio: ['ignore', 'pipe', 'pipe'] })
    llamaChild.once('error', (err) => { llamaSpawnError = err })
    llamaChild.stdout.on('data', (chunk) => process.stderr.write(`[llama-server] ${chunk}`))
    llamaChild.stderr.on('data', (chunk) => process.stderr.write(`[llama-server] ${chunk}`))
  }

  backendHealthReady = summarizeCamelidHealth(await waitForJson(`${backendBase}/v1/health`, {}, 'camelid', waitMs))
  backendStartupMs = round(performance.now() - backendStart)
  await waitForJson(`${llamaBase}/health`, {}, 'llama-server', waitMs).catch((err) => {
    if (llamaSpawnError?.code === 'ENOENT') {
      throw new Error(`could not start llama-server binary ${JSON.stringify(llamaServerBin)}`)
    }
    throw err
  })
  llamaStartupMs = round(performance.now() - llamaStart)
  if (backendSpawnError?.code === 'ENOENT') {
    throw new Error(`could not start Camelid binary ${JSON.stringify(backendBin)}`)
  }

  const loadStarted = performance.now()
  await fetchJson(`${backendBase}/api/models/load`, {
    method: 'POST',
    body: JSON.stringify({ path: modelPath, id: modelId }),
  })
  camelidModelLoadMs = round(performance.now() - loadStarted)
  backendHealthAfterLoad = summarizeCamelidHealth(await fetchJson(`${backendBase}/v1/health`))

  const camelidWarmups = []
  const llamaWarmups = []
  for (let i = 0; i < warmup; i += 1) {
    camelidWarmups.push(await runCamelidStream(i, 'warmup'))
    llamaWarmups.push(await runLlamaStream(i, 'warmup'))
  }

  const beforeMeasuredSnapshot = await captureResourceSnapshot('before_measured_runs')

  const camelidRuns = []
  const llamaRuns = []
  const measuredOrder = []
  for (let i = 0; i < repeats; i += 1) {
    const camelidRun = await runCamelidStream(i, 'measure')
    camelidRuns.push(camelidRun)
    measuredOrder.push({ engine: 'camelid', label: camelidRun.label })

    const llamaRun = await runLlamaStream(i, 'measure')
    llamaRuns.push(llamaRun)
    measuredOrder.push({ engine: 'llama_cpp', label: llamaRun.label })
  }

  const afterMeasuredSnapshot = await captureResourceSnapshot('after_measured_runs')

  const guardrails = benchmarkGuardrails({
    camelidRuns,
    llamaRuns,
    camelidHealthAfterLoad: backendHealthAfterLoad,
  })

  const report = {
    schema: 'camelid.same_host_llama3_benchmark.v1',
    generated_utc: new Date().toISOString(),
    model: {
      row_id: rowId,
      model_path: modelPath,
      model_id: modelId,
      render_mode: renderMode,
    },
    method: {
      warmup,
      repeats,
      max_tokens: maxTokens,
      expected_marker: expectedMarker,
      expected_camelid_q8_policy: expectedCamelidQ8Policy,
      require_marker: requireMarker,
      unique_prompt: uniquePrompt,
      benchmark_messages: benchmarkMessages,
      estimated_prompt_tokens: estimatedPromptTokens,
      llama_context: llamaContext,
      threads: threads ?? null,
      commands: buildPlan().commands,
      outputs: buildPlan().outputs,
      bounded_metrics: boundedMetrics(),
      server_lifecycle: {
        camelid_started_by_harness: startBackend,
        llama_started_by_harness: startLlamaServer,
        camelid_startup_ms: backendStartupMs,
        llama_startup_ms: llamaStartupMs,
        camelid_model_load_ms: camelidModelLoadMs,
        camelid_model_preloaded: !startBackend,
        llama_model_preloaded: !startLlamaServer,
      },
      evidence_context: evidenceContext,
      resource_snapshots: {
        pre_start: preStartSnapshot,
        before_measured_runs: beforeMeasuredSnapshot,
        after_measured_runs: afterMeasuredSnapshot,
      },
      measured_order: measuredOrder,
      note: 'Same-host comparison using alternating streaming requests to Camelid /v1/chat/completions and llama.cpp /completion. TTFT is first non-empty streamed content chunk; token throughput is estimated from streamed content chunks, not tokenizer-ground-truth completion tokens.',
    },
    camelid: {
      base_url: backendBase,
      health: {
        startup_ready: backendHealthReady,
        after_model_load: backendHealthAfterLoad,
      },
      warmups: camelidWarmups,
      runs: camelidRuns,
      summary: summarizeRuns(camelidRuns),
    },
    llama_cpp: {
      base_url: llamaBase,
      warmups: llamaWarmups,
      runs: llamaRuns,
      summary: summarizeRuns(llamaRuns),
      binary: llamaServerBin,
    },
    comparison: compareSummaries(summarizeRuns(camelidRuns), summarizeRuns(llamaRuns)),
    guardrails,
    claim_boundary: claimBoundary(),
  }

  printHumanSummary(report)
  if (out) {
    const outPath = resolve(out)
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`)
    console.log(`json_out=${outPath}`)
  }
  if (!guardrails.passed) {
    if (requireMarker && !guardrails.marker_presence.passed) {
      throw new Error(`benchmark marker guard failed: expected ${JSON.stringify(expectedMarker)} in every measured run`)
    }
    if (expectedCamelidQ8Policy && !guardrails.camelid_q8_runtime_policy.matched_expected_policy) {
      throw new Error(`benchmark Camelid Q8 runtime policy guard failed: expected ${JSON.stringify(expectedCamelidQ8Policy)}, observed ${JSON.stringify(guardrails.camelid_q8_runtime_policy.observed_policy)}`)
    }
  }
} finally {
  backendChild?.kill('SIGTERM')
  llamaChild?.kill('SIGTERM')
}

function buildPlan() {
  const backendUrl = new URL(backendBase)
  const llamaUrl = new URL(llamaBase)
  const llamaArgs = ['--host', llamaUrl.hostname, '--port', llamaUrl.port || '8183', '-m', modelPath, '-ngl', '0', '-c', String(llamaContext), '--no-warmup']
  if (threads) llamaArgs.push('-t', String(threads))
  return {
    schema: 'camelid.same_host_llama3_benchmark_plan.v1',
    generated_utc: new Date().toISOString(),
    model: {
      row_id: rowId,
      model_path: modelPath,
      model_id: modelId,
      render_mode: renderMode,
    },
    method: {
      warmup,
      repeats,
      max_tokens: maxTokens,
      expected_marker: expectedMarker,
      expected_camelid_q8_policy: expectedCamelidQ8Policy,
      require_marker: requireMarker,
      unique_prompt: uniquePrompt,
      estimated_prompt_tokens: estimatedPromptTokens,
      llama_context: llamaContext,
      threads: threads ?? null,
      benchmark_messages: benchmarkMessages,
      bounded_metrics: boundedMetrics(),
      evidence_context: evidenceContext,
      resource_snapshots: {
        pre_start: preStartSnapshot,
      },
    },
    commands: {
      harness: `node scripts/bench-llama3-same-host.mjs --model ${shellQuote(modelPath)} --model-id ${shellQuote(modelId)} --row-id ${shellQuote(rowId)} --max-tokens ${maxTokens} --warmup ${warmup} --repeats ${repeats}${threads ? ` --threads ${threads}` : ''}${explicitLlamaContext ? ` --llama-context ${explicitLlamaContext}` : ''}${uniquePrompt ? ' --unique-prompt' : ''}${requireMarker ? ` --require-marker --expected-marker ${shellQuote(expectedMarker)}` : ''}${expectedCamelidQ8Policy ? ` --expected-camelid-q8-policy ${shellQuote(expectedCamelidQ8Policy)}` : ''}${out ? ` --out ${shellQuote(resolve(out))}` : ''}`,
      camelid_serve: startBackend ? `${shellQuote(backendBin)} serve --addr ${shellQuote(`${backendUrl.hostname}:${backendUrl.port || '8181'}`)}` : 'not started by harness (--start-backend=false)',
      llama_server: startLlamaServer ? [shellQuote(llamaServerBin), ...llamaArgs.map(shellQuote)].join(' ') : 'not started by harness (--start-llama-server=false)',
      camelid_load_request: `POST ${backendBase}/api/models/load {"path":${JSON.stringify(modelPath)},"id":${JSON.stringify(modelId)}}`,
      camelid_measure_request: `POST ${backendBase}/v1/chat/completions stream=true max_tokens=${maxTokens} temperature=0`,
      llama_cpp_measure_request: `POST ${llamaBase}/completion stream=true n_predict=${maxTokens} temperature=0 cache_prompt=false`,
    },
    outputs: {
      stdout: [
        'camelid_ttft_ms=<mean first non-empty streamed content chunk over measured runs>',
        'camelid_decode_tok_s=<mean estimated streamed chunks per second after first content>',
        'camelid_ms_tok=<mean estimated milliseconds per streamed content chunk after first content>',
        'llama_cpp_ttft_ms=<same metric for llama.cpp>',
        'llama_cpp_decode_tok_s=<same metric for llama.cpp>',
        'llama_cpp_ms_tok=<same metric for llama.cpp>',
        'camelid_q8_policy=<Camelid /v1/health q8_runtime.policy after model load>',
        'camelid_backend_generate_ms=<mean Camelid backend generate timing when CAMELID_STREAM_TIMING_DIAGNOSTICS=on>',
        'camelid_backend_first_content_ms=<mean Camelid backend first-content timing when CAMELID_STREAM_TIMING_DIAGNOSTICS=on>',
        'json_out=<absolute path when --out is set>',
      ],
      json: 'Full machine-readable report at --out, schema camelid.same_host_llama3_benchmark.v1.',
      guardrail: expectedCamelidQ8Policy
        ? `marker_presence=${expectedMarker}; camelid_q8_policy=${expectedCamelidQ8Policy}; pass/fail is recorded under guardrails and can be enforced with --require-marker plus --expected-camelid-q8-policy`
        : `marker_presence=${expectedMarker}; pass/fail is recorded under guardrails and can be enforced with --require-marker`,
      lifecycle: 'Report records server startup timing, model-load timing, warmups, and whether servers were started by the harness or reused preloaded.',
    },
    claim_boundary: claimBoundary(),
  }
}

function boundedMetrics() {
  return [
    'first_byte_ms: first network byte from each streaming response',
    'first_event_ms: first parsed SSE event',
    'first_content_ms / TTFT: first non-empty streamed content chunk',
    'total_elapsed_ms: full streaming response wall time',
    'completion_tokens_estimate: count of non-empty streamed content chunks, not tokenizer-ground-truth tokens',
    'decode_tok_per_s and ms_per_token_after_first: derived from completion_tokens_estimate after first content',
    'marker_presence: exact expected marker observed in measured output text, optionally enforced with --require-marker',
    'camelid_q8_runtime_policy: /v1/health snapshot after model load records the active Q8 runtime policy, lazy-vs-retained booleans, and optional expected-policy guard',
    'camelid_backend_generate_ms and camelid_backend_first_content_ms: opt-in backend timings when CAMELID_STREAM_TIMING_DIAGNOSTICS=on',
    'camelid_backend_q8_calls and q8 timing counters: opt-in Q8 scheduler diagnostics when Camelid Q8 scheduler telemetry is also enabled; call count includes single-projection, fused gate/up, FFN-down decode, and route-table counters',
    'resource_snapshots: host memory/load/storage snapshots before start, before measured runs, and after measured runs',
    'server_lifecycle: Camelid/llama-server startup timing, model-load timing, reuse/preloaded status, and warmup behavior',
  ]
}

function claimBoundary() {
  return `Same-host benchmark snapshot only for exact row ${rowId} with the provided GGUF, prompt, max-token budget, host, binaries, and thread settings. It is bounded timing evidence only: it does not widen support, portability, production-throughput, model-native/larger-context, neighboring-row, broad Llama-family, 1B, or Mixtral claims unless a separate row-specific evidence bundle records those exact conditions.`
}

function printPlan(plan) {
  console.log(`schema=${plan.schema}`)
  console.log(`row_id=${plan.model.row_id}`)
  console.log(`harness_command=${plan.commands.harness}`)
  console.log(`camelid_serve=${plan.commands.camelid_serve}`)
  console.log(`llama_server=${plan.commands.llama_server}`)
  console.log(`outputs=${Object.keys(plan.outputs).join(',')}`)
  console.log(`claim_boundary=${plan.claim_boundary}`)
}

function usage() {
  return `Usage: node scripts/bench-llama3-same-host.mjs --model <GGUF> --model-id <id> --row-id <compat-row> [options]

Purpose:
  Repeatable same-host Camelid vs llama.cpp streaming benchmark for one exact Llama-family row.

Key options:
  --backend <url>                 Camelid API base. Default: CAMELID_API_BASE or http://127.0.0.1:8181
  --llama-url <url>               llama-server base. Default: LLAMA3_LLAMA_SERVER_URL or http://127.0.0.1:8183
  --model <path>                  GGUF path. Default: LLAMA3_GGUF or models/Llama-3.2-3B-Instruct-Q8_0.gguf
  --model-id <id>                 API model id. Default: LLAMA3_MODEL_ID or llama32-3b-q8
  --row-id <compat-row>           Compatibility row recorded in output. Default: llama32_3b_instruct_q8_0
  --max-tokens <n>                Completion budget. Default: 16
  --warmup <n>                    Warmup runs per engine. Default: 1
  --repeats <n>                   Measured runs per engine. Default: 3
  --threads <n>                   Optional llama-server CPU threads.
  --llama-context <n>             Optional llama-server context; otherwise bounded from prompt + max tokens.
  --expected-marker <text>        Marker checked in measured output. Default: CMLD-BENCH.
  --expected-camelid-q8-policy <policy>
                                  Optional /v1/health q8_runtime.policy guard after model load.
  --require-marker                Fail the run after writing output unless every measured output contains the marker.
  --unique-prompt                 Add a per-run request id while preserving the expected marker; helps avoid cache-shaped timing artifacts.
  --start-backend=false           Reuse an already-running Camelid server.
  --start-llama-server=false      Reuse an already-running llama-server.
  --out <path>                    Write the JSON report or --print-plan JSON.
  --print-plan                    Print exact commands/outputs/metric bounds without starting servers.

Example:
  CAMELID_BIN=target/release/camelid \\
  LLAMA3_LLAMA_SERVER=target/reference/llama.cpp/build/bin/llama-server \\
  node scripts/bench-llama3-same-host.mjs \\
    --model /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \\
    --model-id llama32-3b-q8-throughput \\
    --row-id llama32_3b_instruct_q8_0 \\
    --max-tokens 16 --warmup 1 --repeats 3 --threads 8 \\
    --expected-camelid-q8-policy lazy_q8_linear_default_or_auto_retain \\
    --out target/bench-llama32-3b-same-host.json

Outputs:
  stdout summary keys: camelid_ttft_ms, camelid_decode_tok_s, camelid_ms_tok,
  llama_cpp_ttft_ms, llama_cpp_decode_tok_s, llama_cpp_ms_tok, camelid_q8_policy,
  camelid_backend_first_content_ms, camelid_backend_generate_ms,
  camelid_backend_q8_calls, json_out.
  Backend timing fields are populated only when Camelid is run with CAMELID_STREAM_TIMING_DIAGNOSTICS=on.
  JSON report schema: camelid.same_host_llama3_benchmark.v1.

Claim boundary:
  Bounded same-host timing evidence only. This does not promote production throughput,
  portability, 1B, Mixtral, neighboring-row, or broad-family support without separate
  row-specific evidence.`
}

async function runCamelidStream(idx, phase) {
  const label = `camelid-${phase}-${idx + 1}`
  const started = performance.now()
  const response = await fetch(`${backendBase}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: modelId,
      messages: benchmarkMessagesForRun(label),
      max_tokens: maxTokens,
      stream: true,
      temperature: 0,
    }),
  })
  return consumeSseResponse({ response, started, label })
}

async function runLlamaStream(idx, phase) {
  const label = `llama-${phase}-${idx + 1}`
  const started = performance.now()
  const response = await fetch(`${llamaBase}/completion`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      prompt: renderExpectedPrompt(benchmarkMessagesForRun(label), renderMode),
      n_predict: maxTokens,
      temperature: 0,
      stream: true,
      cache_prompt: false,
    }),
  })
  return consumeSseResponse({ response, started, label })
}

function benchmarkMessagesForRun(label) {
  if (!uniquePrompt) return benchmarkMessages
  return [
    benchmarkMessages[0],
    {
      role: 'user',
      content: `Reply with exactly this single line and nothing else: ${expectedMarker}\nRequest id: ${label}. The request id is for measurement only; still reply with exactly ${expectedMarker}.`,
    },
  ]
}

async function consumeSseResponse({ response, started, label }) {
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${label} failed with HTTP ${response.status}: ${text.slice(0, 400)}`)
  }
  const reader = response.body?.getReader()
  if (!reader) throw new Error(`${label} returned no body`)
  const decoder = new TextDecoder()
  let buffer = ''
  let content = ''
  let firstByteMs = null
  let firstEventMs = null
  let firstContentMs = null
  let doneAtMs = null
  let chunkCount = 0
  let completionTokens = 0
  let backendTiming = null

  const nowMs = () => performance.now() - started

  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    if (firstByteMs === null) firstByteMs = nowMs()
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''
    for (const eventText of parts) {
      if (!String(eventText).trim()) continue
      if (firstEventMs === null) firstEventMs = nowMs()
      const dataLines = String(eventText).split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart())
      for (const data of dataLines) {
        if (!data || data === '[DONE]') continue
        let payload
        try {
          payload = JSON.parse(data)
        } catch {
          continue
        }
        const delta = payload?.choices?.[0]?.delta?.content
          ?? payload?.content
          ?? payload?.choices?.[0]?.text
          ?? ''
        if (payload?.camelid?.stream_timing_diagnostics) {
          backendTiming = payload.camelid.stream_timing_diagnostics
        }
        if (delta) {
          if (firstContentMs === null) firstContentMs = nowMs()
          content += delta
          chunkCount += 1
          completionTokens += 1
        }
      }
    }
  }
  doneAtMs = nowMs()
  const decodeWindowMs = firstContentMs === null ? null : Math.max(doneAtMs - firstContentMs, 0)
  return {
    label,
    text: content,
    first_byte_ms: round(firstByteMs),
    first_event_ms: round(firstEventMs),
    first_content_ms: round(firstContentMs),
    total_elapsed_ms: round(doneAtMs),
    completion_tokens_estimate: completionTokens,
    chunk_count: chunkCount,
    backend_timing: backendTiming,
    backend_generate_ms: round(Number(backendTiming?.timings_ms?.generate)),
    backend_first_content_ms: round(Number(backendTiming?.timings_ms?.first_content)),
    backend_q8_gemm_compute_us: round(Number(backendTiming?.q8_schedule?.q8_gemm_compute_us)),
    backend_q8_pack_us: round(Number(backendTiming?.q8_schedule?.activation_quantize_pack_us)),
    backend_q8_calls: q8ScheduleCallCount(backendTiming?.q8_schedule),
    decode_tok_per_s: decodeWindowMs && completionTokens > 0 ? round((completionTokens / decodeWindowMs) * 1000) : null,
    ms_per_token_after_first: decodeWindowMs && completionTokens > 0 ? round(decodeWindowMs / completionTokens) : null,
  }
}

function summarizeRuns(runs) {
  const avg = (field) => round(average(runs.map((run) => run[field]).filter(Number.isFinite)))
  return {
    count: runs.length,
    avg_first_byte_ms: avg('first_byte_ms'),
    avg_first_event_ms: avg('first_event_ms'),
    avg_ttft_ms: avg('first_content_ms'),
    avg_total_elapsed_ms: avg('total_elapsed_ms'),
    avg_backend_generate_ms: avg('backend_generate_ms'),
    avg_backend_first_content_ms: avg('backend_first_content_ms'),
    avg_backend_q8_calls: avg('backend_q8_calls'),
    avg_backend_q8_gemm_compute_us: avg('backend_q8_gemm_compute_us'),
    avg_backend_q8_pack_us: avg('backend_q8_pack_us'),
    avg_decode_tok_per_s: avg('decode_tok_per_s'),
    avg_ms_per_token_after_first: avg('ms_per_token_after_first'),
    avg_completion_tokens_estimate: avg('completion_tokens_estimate'),
  }
}

function compareSummaries(camelid, llama) {
  const pct = (a, b) => Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? round(((a - b) / b) * 100) : null
  return {
    ttft_delta_pct_vs_llama_cpp: pct(camelid.avg_ttft_ms, llama.avg_ttft_ms),
    total_elapsed_delta_pct_vs_llama_cpp: pct(camelid.avg_total_elapsed_ms, llama.avg_total_elapsed_ms),
    decode_tok_per_s_delta_pct_vs_llama_cpp: pct(camelid.avg_decode_tok_per_s, llama.avg_decode_tok_per_s),
    ms_per_token_after_first_delta_pct_vs_llama_cpp: pct(camelid.avg_ms_per_token_after_first, llama.avg_ms_per_token_after_first),
  }
}

function printHumanSummary(report) {
  const c = report.camelid.summary
  const l = report.llama_cpp.summary
  console.log(`camelid_ttft_ms=${c.avg_ttft_ms}`)
  console.log(`camelid_decode_tok_s=${c.avg_decode_tok_per_s}`)
  console.log(`camelid_ms_tok=${c.avg_ms_per_token_after_first}`)
  console.log(`llama_cpp_ttft_ms=${l.avg_ttft_ms}`)
  console.log(`llama_cpp_decode_tok_s=${l.avg_decode_tok_per_s}`)
  console.log(`llama_cpp_ms_tok=${l.avg_ms_per_token_after_first}`)
  console.log(`camelid_q8_policy=${report.camelid.health?.after_model_load?.q8_runtime?.policy ?? 'unknown'}`)
  console.log(`camelid_backend_first_content_ms=${c.avg_backend_first_content_ms}`)
  console.log(`camelid_backend_generate_ms=${c.avg_backend_generate_ms}`)
  console.log(`camelid_backend_q8_calls=${c.avg_backend_q8_calls}`)
}

function benchmarkGuardrails({ camelidRuns, llamaRuns, camelidHealthAfterLoad }) {
  const runStatus = (runs) => runs.map((run) => ({
    label: run.label,
    contains_expected_marker: String(run.text || '').includes(expectedMarker),
    nonempty_streamed_output: Boolean(String(run.text || '').trim()) && (run.completion_tokens_estimate || 0) > 0,
  }))
  const camelid = runStatus(camelidRuns)
  const llamaCpp = runStatus(llamaRuns)
  const markerPassed = [...camelid, ...llamaCpp].every((item) => item.contains_expected_marker && item.nonempty_streamed_output)
  const observedPolicy = camelidHealthAfterLoad?.q8_runtime?.policy ?? null
  const q8PolicyPassed = expectedCamelidQ8Policy
    ? observedPolicy === expectedCamelidQ8Policy
    : true
  return {
    expected_marker: expectedMarker,
    expected_camelid_q8_policy: expectedCamelidQ8Policy,
    require_marker: requireMarker,
    marker_presence: {
      camelid,
      llama_cpp: llamaCpp,
      passed: markerPassed,
    },
    camelid_q8_runtime_policy: {
      observed_policy: observedPolicy,
      observed_lazy_q8_linear: camelidHealthAfterLoad?.q8_runtime?.lazy_q8_linear ?? null,
      observed_retain_q8_blocks: camelidHealthAfterLoad?.q8_runtime?.retain_q8_blocks ?? null,
      matched_expected_policy: expectedCamelidQ8Policy ? q8PolicyPassed : null,
      note: expectedCamelidQ8Policy
        ? 'The harness exits non-zero after writing output if Camelid /v1/health reports a different q8_runtime.policy than the expected matrix row.'
        : 'Observed Camelid /v1/health q8_runtime policy is recorded for same-host matrix evidence even when no explicit policy guard is requested.',
    },
    passed: markerPassed && q8PolicyPassed,
    note: requireMarker
      ? 'The harness exits non-zero if any measured run omits the expected marker; empty measured output is always treated as a failed guardrail.'
      : 'Marker presence and non-empty streamed output are recorded for deterministic-output hygiene; empty measured output always fails the recorded guardrail even when marker enforcement is not requested.',
  }
}

function summarizeCamelidHealth(health) {
  return {
    loaded_now: Boolean(health?.loaded_now),
    generation_ready: Boolean(health?.generation_ready),
    active_model_id: health?.active_model_id ?? null,
    q8_runtime: health?.q8_runtime ?? null,
  }
}

async function collectEvidenceContext({ includeFileHashes }) {
  const modelInfo = await fileEvidence(modelPath, { includeHash: includeFileHashes })
  const backendInfo = await fileEvidence(backendBin, { includeHash: includeFileHashes })
  const llamaInfo = await fileEvidence(llamaServerBin, { includeHash: includeFileHashes })
  return {
    repository: {
      head: gitOutput(['rev-parse', 'HEAD']),
      branch: gitOutput(['rev-parse', '--abbrev-ref', 'HEAD']),
      status_short: gitOutput(['status', '--short']) || '',
    },
    host_class: hostClass(),
    binaries: {
      camelid: backendInfo,
      llama_server: llamaInfo,
      node: process.version,
    },
    model_artifact: modelInfo,
    privacy_note: 'Host name, user name, and home-relative local paths are intentionally not recorded; scrub full artifacts before publication if absolute paths appear in commands.',
  }
}

function gitOutput(argsList) {
  try {
    const result = spawnSync('git', argsList, { cwd: process.cwd(), encoding: 'utf8' })
    if (result.status !== 0) return null
    return result.stdout.trim()
  } catch {
    return null
  }
}

function hostClass() {
  const cpu = os.cpus()?.[0] || {}
  return {
    os_type: os.type(),
    platform: os.platform(),
    release: os.release(),
    arch: os.arch(),
    cpu_model: cpu.model || null,
    cpu_count: os.cpus()?.length || null,
    total_memory_gib: round(os.totalmem() / 1024 / 1024 / 1024),
  }
}

async function fileEvidence(path, { includeHash }) {
  const resolved = resolve(path)
  const info = {
    path: resolved,
    exists: false,
    size_bytes: null,
    sha256: includeHash ? null : 'not_computed_in_plan_mode',
  }
  try {
    const metadata = await stat(resolved)
    info.exists = metadata.isFile()
    info.size_bytes = metadata.size
    if (includeHash && info.exists) info.sha256 = await sha256File(resolved)
  } catch (error) {
    info.error = error.code || error.message
  }
  return info
}

function sha256File(path) {
  return new Promise((resolvePromise, rejectPromise) => {
    const hash = createHash('sha256')
    const stream = createReadStream(path)
    stream.on('data', (chunk) => hash.update(chunk))
    stream.on('error', rejectPromise)
    stream.on('end', () => resolvePromise(hash.digest('hex')))
  })
}

async function captureResourceSnapshot(label) {
  const memory = {
    total_gib: round(os.totalmem() / 1024 / 1024 / 1024),
    free_gib: round(os.freemem() / 1024 / 1024 / 1024),
    process_rss_mib: round(process.memoryUsage().rss / 1024 / 1024),
  }
  const linuxMemory = await readLinuxMeminfo()
  const storage = await filesystemSnapshot(process.cwd())
  return {
    label,
    captured_utc: new Date().toISOString(),
    loadavg_1m_5m_15m: os.loadavg().map(round),
    memory,
    linux_meminfo: linuxMemory,
    storage,
  }
}

async function readLinuxMeminfo() {
  try {
    const text = await readFile('/proc/meminfo', 'utf8')
    const getKiB = (key) => {
      const match = text.match(new RegExp(`^${key}:\\s+(\\d+)\\s+kB`, 'm'))
      return match ? Number(match[1]) : null
    }
    const toGiB = (kib) => Number.isFinite(kib) ? round(kib / 1024 / 1024) : null
    return {
      mem_available_gib: toGiB(getKiB('MemAvailable')),
      swap_total_gib: toGiB(getKiB('SwapTotal')),
      swap_free_gib: toGiB(getKiB('SwapFree')),
      dirty_mib: Number.isFinite(getKiB('Dirty')) ? round(getKiB('Dirty') / 1024) : null,
    }
  } catch {
    return null
  }
}

async function filesystemSnapshot(path) {
  try {
    const fs = await statfs(path)
    const blockSize = Number(fs.bsize || 0)
    const totalBytes = Number(fs.blocks || 0) * blockSize
    const availableBytes = Number(fs.bavail || 0) * blockSize
    return {
      path: resolve(path),
      total_gib: round(totalBytes / 1024 / 1024 / 1024),
      available_gib: round(availableBytes / 1024 / 1024 / 1024),
      used_pct: totalBytes > 0 ? round(((totalBytes - availableBytes) / totalBytes) * 100) : null,
    }
  } catch (error) {
    return { path: resolve(path), error: error.code || error.message }
  }
}

async function waitForJson(url, init, label, timeoutMs) {
  const started = Date.now()
  for (;;) {
    try {
      return await fetchJson(url, init)
    } catch (error) {
      if (Date.now() - started >= timeoutMs) throw new Error(`${label} did not become ready within ${timeoutMs}ms: ${error.message}`)
      await sleep(500)
    }
  }
}

async function fetchJson(url, init) {
  const response = await fetch(url, {
    ...init,
    headers: {
      'content-type': 'application/json',
      ...(init?.headers || {}),
    },
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${url} failed with HTTP ${response.status}: ${text.slice(0, 400)}`)
  }
  return response.json()
}

function estimatePromptTokens(text) {
  const normalized = String(text || '').trim()
  if (!normalized) return 0
  const pieces = normalized.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu) || []
  return Math.max(1, Math.round(Math.max(pieces.length, normalized.length / 4)))
}

function average(values) {
  if (!values.length) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function q8ScheduleCallCount(q8Schedule) {
  if (!q8Schedule || typeof q8Schedule !== 'object') return null
  const direct = [
    q8Schedule.i8mm_single_projection_calls,
    q8Schedule.i8mm_fused_gate_up_calls,
    q8Schedule.ffn_down_decode_consumer_taken,
    q8Schedule.ffn_down_vnni_decode_taken,
  ]
  let directTotal = 0
  for (const value of direct) {
    const number = Number(value)
    if (Number.isFinite(number)) directTotal += number
  }
  const routes = q8Schedule.projection_routes ?? q8Schedule.output_projection_by_route
  let routeTotal = 0
  if (routes && typeof routes === 'object') {
    for (const route of Object.values(routes)) {
      const calls = Number(route?.calls)
      if (Number.isFinite(calls)) routeTotal += calls
    }
  }
  const total = Math.max(directTotal, routeTotal)
  return total > 0 ? round(total) : null
}

function round(value) {
  return Number.isFinite(value) ? Math.round(value * 100) / 100 : null
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[i + 1]
    const value = inline ?? (next && !next.startsWith('--') ? argv[++i] : 'true')
    parsed.set(key, value)
  }
  return parsed
}

function parsePositiveInt(value, label) {
  const parsed = Number.parseInt(String(value), 10)
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`--${label} must be a positive integer, got ${value}`)
  return parsed
}

function parseNonNegativeInt(value, label) {
  const parsed = Number.parseInt(String(value), 10)
  if (!Number.isInteger(parsed) || parsed < 0) throw new Error(`--${label} must be a non-negative integer, got ${value}`)
  return parsed
}

function parseOptionalPositiveInt(value, label) {
  if (value === undefined || value === null || value === '') return null
  return parsePositiveInt(value, label)
}

function sleep(ms) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms))
}

function shellQuote(value) {
  const text = String(value)
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) return text
  return `'${text.replaceAll("'", `'"'"'`)}'`
}
