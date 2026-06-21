#!/usr/bin/env node
// CPU prefill+decode matrix: llama.cpp (CPU) vs Camelid (CPU) in two configs:
//   A = default (shipping)            B = gated x86 packed-rows4 SIMD prefill kernels ON
// Greedy/temp=0. Node-managed child processes (reliable kill on Windows).
//
// Env required:
//   CAMELID_BIN, LLAMA_SERVER_BIN, MODEL_GGUF
import { spawn } from 'node:child_process'
import { writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const CAMELID_BIN = process.env.CAMELID_BIN
const LLAMA_SERVER_BIN = process.env.LLAMA_SERVER_BIN
const MODEL = process.env.MODEL_GGUF
const MODEL_ID = process.env.MODEL_ID || 'probe-model'
const OUT = process.env.OUT_JSON
const FACTS = parseInt(process.env.FACTS || '24', 10)
const DECODE_TOKENS = parseInt(process.env.DECODE_TOKENS || '64', 10)
const LLAMA_CTX = process.env.LLAMA_CTX || '1536'
if (!CAMELID_BIN || !LLAMA_SERVER_BIN || !MODEL) { console.error('need CAMELID_BIN, LLAMA_SERVER_BIN, MODEL_GGUF'); process.exit(2) }

const longBlock = Array.from({ length: FACTS }, (_, i) =>
  `Fact ${i + 1}: the quick brown fox jumps over the lazy dog near the riverbank at dawn.`).join(' ')
// nonce in the SYSTEM message => first token differs => defeats prompt-prefix caching
// (Camelid caches by default; llama also gets cache_prompt:false). Real cold prefill each call.
let nonceCtr = 0
const mkPrefill = (nonce) => ([
  { role: 'system', content: `You are a concise assistant. [run ${nonce}]` },
  { role: 'user', content: `Read the following text and answer with the single word OK.\n\n${longBlock}` },
])
const decodeMessages = [
  { role: 'system', content: 'You are a concise assistant.' },
  { role: 'user', content: 'Count from 1 to 80, separated by commas. Output only the numbers.' },
]
const round = (x) => Number.isFinite(x) ? Math.round(x * 100) / 100 : null
const sleep = (ms) => new Promise(r => setTimeout(r, ms))

async function waitHealth(url, ms = 600000) {
  const start = Date.now()
  for (;;) {
    try { const r = await fetch(url); if (r.ok) return } catch {}
    if (Date.now() - start > ms) throw new Error(`health timeout ${url}`)
    await sleep(400)
  }
}
async function chat(base, messages, maxTokens) {
  const isLlama = base.includes('8183')
  const body = { model: MODEL_ID, messages, max_tokens: maxTokens, temperature: 0, stream: false }
  if (isLlama) body.cache_prompt = false // defeat llama prompt cache so prefill is re-measured; Camelid rejects this param
  const t0 = performance.now()
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  const t1 = performance.now()
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`)
  const j = await r.json()
  return { ms: t1 - t0, text: j?.choices?.[0]?.message?.content ?? '', usage: j?.usage ?? null, timings: j?.timings ?? null }
}
async function probe(base) {
  await chat(base, mkPrefill('warm-weights'), 1) // warmup weights/page cache only
  const p = await chat(base, mkPrefill(`measure-${++nonceCtr}-${Date.now()}`), 1) // unique prefix => real cold prefill
  const promptTokens = p.usage?.prompt_tokens ?? p.timings?.prompt_n ?? null
  await chat(base, decodeMessages, 4) // warmup
  const d1 = await chat(base, decodeMessages, 1)
  const dN = await chat(base, decodeMessages, DECODE_TOKENS)
  const nOut = dN.usage?.completion_tokens ?? null
  const win = dN.ms - d1.ms
  return {
    prefill: {
      prompt_tokens: promptTokens, wall_ms: round(p.ms),
      prefill_tok_s: promptTokens ? round(promptTokens / (p.ms / 1000)) : null,
      llama_prompt_per_s: p.timings?.prompt_per_second ? round(p.timings.prompt_per_second) : null,
    },
    decode: {
      completion_tokens: nOut, t1_ms: round(d1.ms), tN_ms: round(dN.ms), window_ms: round(win),
      decode_tok_s: (nOut && nOut > 1 && win > 0) ? round((nOut - 1) / (win / 1000)) : null,
      llama_predicted_per_s: dN.timings?.predicted_per_second ? round(dN.timings.predicted_per_second) : null,
      output_text: dN.text,
    },
  }
}
function startProc(bin, argv, env) {
  const child = spawn(bin, argv, { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, ...env } })
  const tag = bin.includes('camelid') ? 'camelid' : 'llama'
  child.stdout.on('data', c => process.stderr.write(`[${tag}] ${c}`))
  child.stderr.on('data', c => process.stderr.write(`[${tag}] ${c}`))
  return child
}

const report = { generated_utc: new Date().toISOString(), model: MODEL, model_id: MODEL_ID, facts: FACTS, decode_tokens: DECODE_TOKENS, runs: {} }
let llama, camelid
try {
  // llama-server CPU once
  llama = startProc(LLAMA_SERVER_BIN, ['--host', '127.0.0.1', '--port', '8183', '-m', MODEL, '-ngl', '0', '-c', LLAMA_CTX, '-t', '8', '--no-warmup'], {})
  await waitHealth('http://127.0.0.1:8183/health')
  report.runs.llama_cpp_cpu = await probe('http://127.0.0.1:8183')
  console.error('=== llama.cpp done ===')

  const configs = {
    camelid_cpu_A_default: { CAMELID_CUDA_RESIDENT_DECODE: '0', CAMELID_CUDA_RESIDENT_PREFILL: '0' },
    camelid_cpu_B_packed_simd: {
      CAMELID_CUDA_RESIDENT_DECODE: '0',
      CAMELID_CUDA_RESIDENT_PREFILL: '0',
      CAMELID_X86_Q8_REPACK: 'on',
      CAMELID_X86_Q8_KERNEL: 'avx2',
      CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL: 'on',
      CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2: 'on',
      CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED: 'on',
      CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL: 'on',
      CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL: 'on',
      CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL: 'on',
      CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL: 'on',
      CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL: 'on',
    },
  }
  for (const [name, cfgEnv] of Object.entries(configs)) {
    camelid = startProc(CAMELID_BIN, ['serve', '--addr', '127.0.0.1:8181'], cfgEnv)
    await waitHealth('http://127.0.0.1:8181/v1/health')
    await (await fetch('http://127.0.0.1:8181/api/models/load', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path: MODEL, id: MODEL_ID }) })).json()
    report.runs[name] = await probe('http://127.0.0.1:8181')
    report.runs[name].config_env = cfgEnv
    console.error(`=== ${name} done ===`)
    camelid.kill('SIGTERM'); await sleep(1500); camelid = null
  }

  // parity diff: A vs B output text
  const a = report.runs.camelid_cpu_A_default?.decode?.output_text ?? ''
  const b = report.runs.camelid_cpu_B_packed_simd?.decode?.output_text ?? ''
  report.parity_A_vs_B_decode_identical = a === b
  report.parity_note = a === b ? 'A and B produced byte-identical greedy decode output' : 'DIVERGENCE between A and B'
} finally {
  try { camelid?.kill('SIGTERM') } catch {}
  try { llama?.kill('SIGTERM') } catch {}
}
console.log(JSON.stringify(report, null, 2))
if (OUT) { await mkdir(dirname(OUT), { recursive: true }); await writeFile(OUT, JSON.stringify(report, null, 2) + '\n'); console.error(`wrote ${OUT}`) }
