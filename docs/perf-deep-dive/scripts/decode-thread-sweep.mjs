#!/usr/bin/env node
// Camelid CPU decode/prefill thread-sensitivity sweep (vary RAYON_NUM_THREADS).
// Bit-identical across thread counts (independent dots) — so output must match.
// CUDA must be hidden by caller. Diagnoses whether decode is under-parallelized.
import { spawn } from 'node:child_process'
import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const CAMELID_BIN = process.env.CAMELID_BIN
const MODEL = process.env.MODEL_GGUF
const MODEL_ID = process.env.MODEL_ID || 'probe-model'
const OUT = process.env.OUT_JSON
const FACTS = parseInt(process.env.FACTS || '16', 10)
const DECODE_TOKENS = parseInt(process.env.DECODE_TOKENS || '48', 10)
if (!CAMELID_BIN || !MODEL) { console.error('need CAMELID_BIN, MODEL_GGUF'); process.exit(2) }

const longBlock = Array.from({ length: FACTS }, (_, i) =>
  `Fact ${i + 1}: the quick brown fox jumps over the lazy dog near the riverbank at dawn.`).join(' ')
let nonce = 0
const mkPrefill = (n) => ([{ role: 'system', content: `You are a concise assistant. [run ${n}]` },
  { role: 'user', content: `Read the following text and answer with the single word OK.\n\n${longBlock}` }])
const decodeMessages = [{ role: 'system', content: 'You are a concise assistant.' },
  { role: 'user', content: 'Count from 1 to 60, separated by commas. Output only the numbers.' }]
const round = (x) => Number.isFinite(x) ? Math.round(x * 100) / 100 : null
const sleep = (ms) => new Promise(r => setTimeout(r, ms))
async function waitHealth(url, ms = 600000) { const s = Date.now(); for (;;) { try { if ((await fetch(url)).ok) return } catch {}; if (Date.now() - s > ms) throw new Error('health timeout'); await sleep(400) } }
async function chat(base, messages, maxTokens) {
  const t0 = performance.now()
  const r = await fetch(`${base}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL_ID, messages, max_tokens: maxTokens, temperature: 0, stream: false }) })
  const t1 = performance.now(); if (!r.ok) throw new Error(`HTTP ${r.status}`); const j = await r.json()
  return { ms: t1 - t0, text: j?.choices?.[0]?.message?.content ?? '', usage: j?.usage ?? null }
}
async function probe(base) {
  // decode: t_N - t_1 isolates decode from prefill
  await chat(base, decodeMessages, 4)
  const d1 = await chat(base, decodeMessages, 1)
  const dN = await chat(base, decodeMessages, DECODE_TOKENS)
  const nOut = dN.usage?.completion_tokens ?? null
  const win = dN.ms - d1.ms
  // prefill: 1 unique cold pass
  await chat(base, mkPrefill('warm'), 1)
  const p = await chat(base, mkPrefill(`m-${++nonce}-${Date.now()}`), 1)
  const pt = p.usage?.prompt_tokens ?? null
  return {
    decode_tok_s: (nOut && nOut > 1 && win > 0) ? round((nOut - 1) / (win / 1000)) : null,
    decode_completion_tokens: nOut, decode_text: dN.text,
    prefill_tok_s: pt ? round(pt / (p.ms / 1000)) : null, prefill_prompt_tokens: pt,
  }
}
const base = 'http://127.0.0.1:8181'
const threadCounts = (process.env.THREADS || 'default,1,2,4,6,8,16').split(',')
const report = { generated_utc: new Date().toISOString(), model: MODEL, runs: {} }
for (const tc of threadCounts) {
  const env = { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0' }
  if (tc !== 'default') env.RAYON_NUM_THREADS = tc
  let c
  try {
    c = spawn(CAMELID_BIN, ['serve', '--addr', '127.0.0.1:8181'], { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, ...env } })
    c.stdout.on('data', d => process.stderr.write(`[c] ${d}`)); c.stderr.on('data', d => process.stderr.write(`[c] ${d}`))
    await waitHealth(`${base}/v1/health`)
    await (await fetch(`${base}/api/models/load`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path: MODEL, id: MODEL_ID }) })).json()
    const res = await probe(base); res.rayon_threads = tc; report.runs[`threads_${tc}`] = res
    console.error(`=== threads=${tc}: decode ${res.decode_tok_s} tok/s | prefill ${res.prefill_tok_s} tok/s ===`)
  } catch (e) { report.runs[`threads_${tc}`] = { error: String(e) }; console.error(`=== threads=${tc} ERROR: ${e} ===`) }
  finally { try { c?.kill('SIGTERM') } catch {}; await sleep(1500) }
}
const ref = report.runs.threads_default?.decode_text ?? null
report.parity = Object.fromEntries(Object.entries(report.runs).map(([k, v]) => [k, v?.decode_text === ref]))
report.all_parity_identical = Object.values(report.parity).every(Boolean)
console.log(JSON.stringify(report, null, 2))
if (OUT) { await mkdir(dirname(OUT), { recursive: true }); await writeFile(OUT, JSON.stringify(report, null, 2) + '\n'); console.error(`wrote ${OUT}`) }
