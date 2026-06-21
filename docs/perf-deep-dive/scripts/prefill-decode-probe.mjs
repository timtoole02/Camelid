#!/usr/bin/env node
// Focused CPU prefill+decode probe for Camelid vs llama.cpp via OpenAI-compatible
// /v1/chat/completions (both servers must already be running). Greedy (temp=0).
//
// Prefill tok/s : long fixed prompt, max_tokens=1, wall-clock; tokens from server usage.
// Decode tok/s  : short prompt; (t_N - t_1) isolates decode from prefill; no streaming needed.
// Output text   : captured from the decode run for cross-config parity diff.
//
// Usage: node prefill-decode-probe.mjs --base http://127.0.0.1:8181 --model <id> --label <s>
//        [--decode-tokens 64] [--out file.json]

const args = new Map()
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i]
  if (a.startsWith('--')) { const k = a.slice(2); const v = process.argv[i+1] && !process.argv[i+1].startsWith('--') ? process.argv[++i] : 'true'; args.set(k, v) }
}
const base = (args.get('base') || 'http://127.0.0.1:8181').replace(/\/$/, '')
const model = args.get('model') || 'm'
const label = args.get('label') || 'run'
const decodeTokens = parseInt(args.get('decode-tokens') || '64', 10)
const out = args.get('out')

// A long, fixed, deterministic prompt to exercise prefill. ~repeated factual sentence.
const longBlock = Array.from({ length: 60 }, (_, i) =>
  `Fact ${i + 1}: the quick brown fox jumps over the lazy dog near the riverbank at dawn.`).join(' ')
const prefillMessages = [
  { role: 'system', content: 'You are a concise assistant.' },
  { role: 'user', content: `Read the following text and answer with the single word OK.\n\n${longBlock}` },
]
// Short, deterministic decode prompt that elicits a long greedy continuation.
const decodeMessages = [
  { role: 'system', content: 'You are a concise assistant.' },
  { role: 'user', content: 'Count from 1 to 80, separated by commas. Output only the numbers.' },
]

async function chat(messages, maxTokens) {
  const t0 = performance.now()
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0, stream: false, cache_prompt: false }),
  })
  const t1 = performance.now()
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`)
  const j = await r.json()
  return {
    ms: t1 - t0,
    text: j?.choices?.[0]?.message?.content ?? '',
    usage: j?.usage ?? null,
    timings: j?.timings ?? null, // llama-server ground truth
  }
}

function round(x) { return Number.isFinite(x) ? Math.round(x * 100) / 100 : null }

const result = { label, base }

// ---- Prefill ----
// warmup once (weights/page cache), then measure.
await chat(prefillMessages, 1)
const p = await chat(prefillMessages, 1)
const promptTokens = p.usage?.prompt_tokens ?? p.timings?.prompt_n ?? null
result.prefill = {
  prompt_tokens: promptTokens,
  wall_ms: round(p.ms),
  prefill_tok_s: promptTokens ? round(promptTokens / (p.ms / 1000)) : null,
  llama_prompt_per_s: p.timings?.prompt_per_second ? round(p.timings.prompt_per_second) : null,
}

// ---- Decode (t_N - t_1) ----
await chat(decodeMessages, 4) // warmup short prefill
const d1 = await chat(decodeMessages, 1)
const dN = await chat(decodeMessages, decodeTokens)
const nOut = dN.usage?.completion_tokens ?? null
const decodeWindowMs = dN.ms - d1.ms
const decodeTokSEstimate = (nOut && nOut > 1 && decodeWindowMs > 0) ? round((nOut - 1) / (decodeWindowMs / 1000)) : null
result.decode = {
  completion_tokens: nOut,
  t1_ms: round(d1.ms),
  tN_ms: round(dN.ms),
  decode_window_ms: round(decodeWindowMs),
  decode_tok_s_subtraction: decodeTokSEstimate,
  llama_predicted_per_s: dN.timings?.predicted_per_second ? round(dN.timings.predicted_per_second) : null,
  output_text: dN.text,
}

console.log(JSON.stringify(result, null, 2))
if (out) {
  const { writeFile } = await import('node:fs/promises')
  await writeFile(out, JSON.stringify(result, null, 2) + '\n')
  console.error(`wrote ${out}`)
}
