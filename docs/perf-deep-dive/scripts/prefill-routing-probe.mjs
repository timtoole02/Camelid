#!/usr/bin/env node
// CPU prefill ROUTING sweep for Camelid (loop-ordering only; math unchanged => must stay bit-identical).
// Defeats prompt-prefix cache with a unique system nonce. CUDA must be hidden by the caller.
import { spawn } from 'node:child_process'
import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const CAMELID_BIN = process.env.CAMELID_BIN
const MODEL = process.env.MODEL_GGUF
const MODEL_ID = process.env.MODEL_ID || 'probe-model'
const OUT = process.env.OUT_JSON
const FACTS = parseInt(process.env.FACTS || '16', 10)
const PARITY_TOKENS = parseInt(process.env.PARITY_TOKENS || '24', 10)
if (!CAMELID_BIN || !MODEL) { console.error('need CAMELID_BIN, MODEL_GGUF'); process.exit(2) }

const longBlock = Array.from({ length: FACTS }, (_, i) =>
  `Fact ${i + 1}: the quick brown fox jumps over the lazy dog near the riverbank at dawn.`).join(' ')
let nonce = 0
const mkPrefill = (n) => ([
  { role: 'system', content: `You are a concise assistant. [run ${n}]` },
  { role: 'user', content: `Read the following text and answer with the single word OK.\n\n${longBlock}` },
])
const parityMessages = [
  { role: 'system', content: 'You are a concise assistant.' },
  { role: 'user', content: 'Count from 1 to 40, separated by commas. Output only the numbers.' },
]
const round = (x) => Number.isFinite(x) ? Math.round(x * 100) / 100 : null
const sleep = (ms) => new Promise(r => setTimeout(r, ms))
async function waitHealth(url, ms = 600000) {
  const s = Date.now()
  for (;;) { try { if ((await fetch(url)).ok) return } catch {} ; if (Date.now() - s > ms) throw new Error('health timeout'); await sleep(400) }
}
async function chat(base, messages, maxTokens) {
  const t0 = performance.now()
  const r = await fetch(`${base}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL_ID, messages, max_tokens: maxTokens, temperature: 0, stream: false }) })
  const t1 = performance.now()
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0,200)}`)
  const j = await r.json()
  return { ms: t1 - t0, text: j?.choices?.[0]?.message?.content ?? '', usage: j?.usage ?? null }
}
function startCamelid(env) {
  const c = spawn(CAMELID_BIN, ['serve', '--addr', '127.0.0.1:8181'], { stdio: ['ignore','pipe','pipe'], env: { ...process.env, ...env } })
  c.stdout.on('data', d => process.stderr.write(`[c] ${d}`)); c.stderr.on('data', d => process.stderr.write(`[c] ${d}`))
  return c
}
async function probe(base) {
  await chat(base, mkPrefill('warm'), 1)               // warm weights
  const reps = []
  for (let i = 0; i < 2; i++) reps.push(await chat(base, mkPrefill(`m-${++nonce}-${Date.now()}`), 1))
  const best = reps.reduce((a, b) => a.ms < b.ms ? a : b)
  const pt = best.usage?.prompt_tokens ?? null
  const par = await chat(base, parityMessages, PARITY_TOKENS)
  return { prompt_tokens: pt, prefill_ms_runs: reps.map(r => round(r.ms)), prefill_ms_best: round(best.ms),
           prefill_tok_s: pt ? round(pt / (best.ms / 1000)) : null, parity_text: par.text }
}

const base = 'http://127.0.0.1:8181'
const configs = {
  A_default:        { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0' },
  layer_major_on:   { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0', CAMELID_PREFILL_LAYER_MAJOR: '1' },
  layer_major_off:  { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0', CAMELID_PREFILL_LAYER_MAJOR: '0' },
  chunk_all:        { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0', CAMELID_PREFILL_CHUNK_TOKENS: 'all' },
  chunk_64:         { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0', CAMELID_PREFILL_CHUNK_TOKENS: '64' },
  lm_on_chunk_all:  { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0', CAMELID_PREFILL_LAYER_MAJOR: '1', CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS: 'all' },
}
const report = { generated_utc: new Date().toISOString(), model: MODEL, facts: FACTS, runs: {} }
for (const [name, env] of Object.entries(configs)) {
  let c
  try {
    c = startCamelid(env)
    await waitHealth(`${base}/v1/health`)
    await (await fetch(`${base}/api/models/load`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path: MODEL, id: MODEL_ID }) })).json()
    const res = await probe(base)
    res.config_env = env
    report.runs[name] = res
    console.error(`=== ${name}: prefill ${res.prefill_tok_s} tok/s (best ${res.prefill_ms_best}ms) ===`)
  } catch (e) { report.runs[name] = { error: String(e) }; console.error(`=== ${name} ERROR: ${e} ===`) }
  finally { try { c?.kill('SIGTERM') } catch {}; await sleep(1500) }
}
// parity: every config's greedy parity_text must match A_default
const ref = report.runs.A_default?.parity_text ?? null
report.parity = Object.fromEntries(Object.entries(report.runs).map(([k, v]) => [k, v?.parity_text === ref]))
report.all_parity_identical = Object.values(report.parity).every(Boolean)
console.log(JSON.stringify(report, null, 2))
if (OUT) { await mkdir(dirname(OUT), { recursive: true }); await writeFile(OUT, JSON.stringify(report, null, 2) + '\n'); console.error(`wrote ${OUT}`) }
