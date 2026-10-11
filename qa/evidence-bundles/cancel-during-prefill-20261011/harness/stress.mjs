// Brutal cancel test: many aborts at random points, then check nothing leaked.
// node stress.mjs <port> <rounds> <fillerSentences> <maxAbortMs> <seed>
// Each round's long prompt starts with "Request <seed>-<i>:" so no two rounds share it
// and the prompt cache cannot skip a prompt read.
// 1. the tiny reply, for reference
// 2. <rounds> times: send the long prompt (stream or not, alternating), abort at a
//    pseudo-random time in [300, maxAbortMs] ms, then time a tiny request
// 3. round 0's prompt (cancelled in step 2) again, uncancelled. Its text is compared
//    with the same prompt run fully on another server (rounds = 0 prints just that),
//    so a half-read prompt reused through the prompt cache would show.
// Prints one JSON line.
import { postJson } from "./http-json.mjs"
const [port, roundsArg, fillerArg, maxAbortArg, seedArg] = process.argv.slice(2)
const base = `http://127.0.0.1:${port}`
const rounds = Number(roundsArg)
const maxAbortMs = Number(maxAbortArg)
let seed = Number(seedArg) >>> 0
const rand = () => {
  seed = (seed * 1664525 + 1013904223) >>> 0
  return seed / 2 ** 32
}

const health = await (await fetch(`${base}/v1/health`)).json()
const model = health.active_model_id
const post = (body, signal) =>
  fetch(`${base}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal })

const subjects = ["The river", "A lantern", "The old bridge", "Each harvest", "The north road", "A quiet bell", "The mill", "Every winter"]
const verbs = ["carried", "lit", "connected", "brought", "led to", "rang for", "ground", "covered"]
const objects = ["the valley towns", "the market square", "two farming villages", "grain and cider", "the coastal fort", "the evening prayers", "barley for bread", "the hills in snow"]
let filler = ""
for (let i = 0; i < Number(fillerArg); i++) filler += `${subjects[i % 8]} ${verbs[(i * 3) % 8]} ${objects[(i * 5) % 8]} in year ${1500 + i}. `
const longBody = (stream, n) => ({ model, messages: [{ role: "user", content: `Request ${n}: ${filler}\n\nIn one sentence, what is this text about?` }], max_tokens: 24, temperature: 0, stream })
const tinyBody = { model, messages: [{ role: "user", content: "Say hi." }], max_tokens: 4, temperature: 0, stream: false }

const runFull = async (n) => {
  const t = performance.now()
  const r = await postJson(base, "/v1/chat/completions", longBody(false, n))
  return { ms: Math.round(performance.now() - t), status: r.status, text: r.body?.choices?.[0]?.message?.content ?? null, usage: r.body?.usage ?? null }
}
const runTiny = async () => {
  const t = performance.now()
  const r = await postJson(base, "/v1/chat/completions", tinyBody)
  return { ms: Math.round(performance.now() - t), status: r.status, text: r.body?.choices?.[0]?.message?.content ?? null }
}

if (rounds === 0) {
  console.log(JSON.stringify({ model, reference: await runFull(`${seedArg}-0`) }))
  process.exit(0)
}
const tinyReference = await runTiny()
const trials = []
for (let i = 0; i < rounds; i++) {
  const stream = i % 2 === 1
  const abortAfterMs = Math.round(300 + rand() * (maxAbortMs - 300))
  const controller = new AbortController()
  const started = performance.now()
  const body = longBody(stream, `${seedArg}-${i}`)
  const pending = (stream ? post(body, controller.signal) : postJson(base, "/v1/chat/completions", body, controller.signal))
    .then(async (r) => {
      if (!stream) return { status: r.status }
      const reader = r.body.getReader()
      for (;;) if ((await reader.read()).done) return { status: r.status }
    })
    .catch((e) => ({ error: e.name }))
  await new Promise((r) => setTimeout(r, abortAfterMs))
  controller.abort()
  const outcome = await pending
  const tiny = await runTiny()
  trials.push({ i, stream, abortAfterMs, outcome: outcome.error ?? outcome.status, tinyMs: tiny.ms, tinyStatus: tiny.status, tinySame: tiny.text === tinyReference.text, totalMs: Math.round(performance.now() - started) })
}
const final = await runFull(`${seedArg}-0`)
const after = await (await fetch(`${base}/v1/health`)).json()
console.log(JSON.stringify({
  model, rounds, fillerSentences: Number(fillerArg), maxAbortMs, seed: Number(seedArg),
  tinyReference, trials, final,
  allTinySame: trials.every((t) => t.tinySame && t.tinyStatus === 200),
  maxTinyMs: Math.max(...trials.map((t) => t.tinyMs)),
  healthAfter: { status: after.status, generation_ready: after.generation_ready, active_model_id: after.active_model_id },
}))
