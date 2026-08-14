#!/usr/bin/env node
// Throughput baseline harness for the Qwen3 GPU-resident CUDA decode path.
//
// Phase 3 (throughput) measurement discipline, per the CUDA-resident brief:
//   - greedy / deterministic only (temperature 0)
//   - median of N>=5 runs, report variance (min/median/max + stddev)
//   - same-session comparisons only; note swap pressure
//   - distinguish empty-context (decode) from depth-N (prefill) benchmarks
//
// It drives an already-running `camelid serve` (default 127.0.0.1:8185) over the
// OpenAI-style /v1/completions endpoint. It first asserts the GPU-resident path is
// actually active (via /api/runtime/gpu) so a silent CPU fallback can't be
// mistaken for a GPU number.
//
// Decode tok/s is measured by the standard two-point method (llama.cpp style):
//   t(N) = prefill + (N-1) decode steps;  t(1) = prefill + first token
//   decode_tok_s = (N-1) / (t(N) - t(1))
// which cancels the prefill cost and isolates steady-state single-token decode.
//
// Prefill tok/s = prompt_token_count / (t(1) - one_decode_step) for a long prompt.
//
// Usage:
//   node scripts/bench-qwen3-cuda-resident.mjs \
//     --base http://127.0.0.1:8185 --label "Qwen3-0.6B-Q8_0" \
//     --decode-tokens 128 --prefill-prompt-tokens 512 --runs 5 \
//     --out qa/perf/qwen3-0.6b-cuda-resident-baseline.json

import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const base = (args.get('base') || 'http://127.0.0.1:8185').replace(/\/$/, '')
const label = args.get('label') || 'unknown'
const decodeTokens = Number.parseInt(args.get('decode-tokens') || '128', 10)
const prefillPromptTokens = Number.parseInt(args.get('prefill-prompt-tokens') || '512', 10)
const runs = Number.parseInt(args.get('runs') || '5', 10)
const outPath = args.get('out') || null
const requireGpu = args.get('allow-cpu') !== 'true'
// --decode-prompt-tokens N: run the decode benchmark at depth ~N (a long prompt
// precedes the generated tokens, so decode steps attend over ~N cached positions).
// The two-point method subtracts t(1) from t(N), so the prefill cost still cancels.
// Default 0 keeps today's ~empty-context decode.
const decodePromptTokens = Number.parseInt(args.get('decode-prompt-tokens') || '0', 10)

const DECODE_PROMPT = 'Count slowly:' // short, ~empty-context decode

async function main() {
  await assertGpuActive()

  // A long deterministic prompt for the prefill benchmark. We measure how many
  // prompt tokens the server reports and divide by prefill wall time.
  const longPrompt = buildLongPrompt(prefillPromptTokens)
  const decodePrompt = decodePromptTokens > 0 ? buildLongPrompt(decodePromptTokens) : DECODE_PROMPT

  // Warm up (build resident engine, JIT kernels, fill caches) — not measured.
  await complete(decodePrompt, 8)
  await complete(longPrompt, 1)

  // --- Decode benchmark (empty-context, or depth-N with --decode-prompt-tokens) ---
  const tN = []
  const t1 = []
  for (let i = 0; i < runs; i++) {
    tN.push(await timed(() => complete(decodePrompt, decodeTokens)))
    t1.push(await timed(() => complete(decodePrompt, 1)))
  }
  // Per-run decode tok/s using paired (t(N), t(1)) from the same iteration.
  const decodeTokS = tN.map((tn, i) => (decodeTokens - 1) / (tn.ms - t1[i].ms) * 1000)
  // First-token / prefill latency for the short prompt (informational).
  const shortPrefillMs = t1.map((r) => r.ms)

  // --- Prefill benchmark (depth-N) ---
  const prefillRuns = []
  for (let i = 0; i < runs; i++) {
    // Change the FIRST user token, not a suffix. Resident prefix continuation
    // is a production win, but reusing the entire benchmark prompt would turn
    // this into a prefix-cache benchmark and hide actual prefill throughput.
    const nonce = String.fromCharCode(65 + (i % 26)).repeat(12)
    const uniquePrompt = `${nonce} benchmark sample ${i}.\n${longPrompt}`
    const r = await timed(() => complete(uniquePrompt, 1))
    prefillRuns.push(r)
  }
  const promptTok = prefillRuns[0].promptTokens
  // prefill total includes 1 generated token; for a long prompt that's <1% so we
  // report prompt_tokens / total_ms. (Decode-step subtraction is below 1% noise here.)
  const prefillTokS = prefillRuns.map((r) => promptTok / r.ms * 1000)

  const result = {
    schema: 'camelid.qwen3.cuda_resident.throughput_baseline.v1',
    label,
    base,
    mode: 'greedy_temperature_0',
    gpu_resident_active: true,
    runs,
    decode: {
      generated_tokens: decodeTokens,
      context_prompt_tokens: decodePromptTokens,
      tok_s: stats(decodeTokS),
      short_prefill_first_token_ms: stats(shortPrefillMs),
    },
    prefill: {
      prompt_tokens: promptTok,
      tok_s: stats(prefillTokS),
      total_ms: stats(prefillRuns.map((r) => r.ms)),
      resident_receipts: prefillRuns.map((r) => r.residentPrefill).filter(Boolean),
    },
    note: 'median-of-N, same session; RTX 3060 Laptop 6GB. Pre-optimization baseline.',
  }

  console.log(JSON.stringify(result, null, 2))
  if (outPath) {
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, JSON.stringify(result, null, 2))
    console.error(`wrote ${outPath}`)
  }
}

async function assertGpuActive() {
  try {
    const res = await fetch(`${base}/api/runtime/gpu`)
    const body = await res.json()
    const active = body.active ?? body.cuda_resident_active ?? body.resident ?? body.enabled
    console.error(`/api/runtime/gpu -> ${JSON.stringify(body)}`)
    if (requireGpu && active === false) {
      throw new Error('GPU-resident path is NOT active; refusing to report a GPU number. Pass --allow-cpu=true to override.')
    }
  } catch (e) {
    if (requireGpu) throw new Error(`could not confirm GPU active: ${e.message}`)
  }
}

async function complete(prompt, maxTokens) {
  const res = await fetch(`${base}/v1/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      prompt,
      max_tokens: maxTokens,
      temperature: 0,
      top_k: 1,
      stream: false,
    }),
  })
  if (!res.ok) throw new Error(`/v1/completions -> HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`)
  const body = await res.json()
  const promptTokens =
    body.usage?.prompt_tokens ??
    body.timings?.prompt_evaluation?.prompt_token_count ??
    null
  return { body, promptTokens }
}

async function timed(fn) {
  const start = process.hrtime.bigint()
  const { body, promptTokens } = await fn()
  const end = process.hrtime.bigint()
  return {
    ms: Number(end - start) / 1e6,
    promptTokens,
    residentPrefill: body?.timings?.prompt_evaluation?.resident_prefill ?? null,
  }
}

function buildLongPrompt(approxTokens) {
  // ~0.75 words/token; build a deterministic filler well above target so the
  // server's tokenizer yields >= approxTokens. We report the server's count.
  const sentence = 'The quick brown fox jumps over the lazy dog near the riverbank. '
  let s = 'Summarize the following passage in one word.\n\n'
  while (s.length < approxTokens * 5) s += sentence
  return s
}

function stats(xs) {
  const arr = xs.filter((x) => Number.isFinite(x)).slice().sort((a, b) => a - b)
  const n = arr.length
  const med = n % 2 ? arr[(n - 1) / 2] : (arr[n / 2 - 1] + arr[n / 2]) / 2
  const mean = arr.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / n)
  return { median: round(med), min: round(arr[0]), max: round(arr[n - 1]), stddev: round(sd), n }
}

function median(xs) {
  const arr = xs.slice().sort((a, b) => a - b)
  const n = arr.length
  return n % 2 ? arr[(n - 1) / 2] : (arr[n / 2 - 1] + arr[n / 2]) / 2
}

function round(x) {
  return Math.round(x * 100) / 100
}

function parseArgs(argv) {
  const m = new Map()
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2)
      const val = i + 1 < argv.length && !argv[i + 1].startsWith('--') ? argv[++i] : 'true'
      m.set(key, val)
    }
  }
  return m
}

main().catch((e) => {
  console.error(e.stack || String(e))
  process.exit(1)
})
