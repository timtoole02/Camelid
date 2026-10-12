// Measures how long an abandoned request keeps the engine busy.
// node cancel-bench.mjs <port> <stream|nonstream> <abortAfterMs> <fillerSentences> [maxTokens]
// Each run gets a unique lead so the prompt cache cannot skip the prompt read.
// abortAfterMs = 0 runs the long request to completion (no cancel) as a reference.
// Prints one JSON line.
import { postJson } from "./http-json.mjs"
const [port, mode, abortAfterArg, fillerArg, maxTokensArg] = process.argv.slice(2)
const base = `http://127.0.0.1:${port}`
const stream = mode === "stream"
const abortAfterMs = Number(abortAfterArg)
const fillerSentences = Number(fillerArg)
const maxTokens = Number(maxTokensArg ?? 64)

const model = (await (await fetch(`${base}/v1/health`)).json()).active_model_id

const post = (body, signal) =>
  fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })

const tinyBody = { model, messages: [{ role: "user", content: "Say hi." }], max_tokens: 4, temperature: 0, stream: false }
const timeTiny = async () => {
  const t = performance.now()
  const r = await postJson(base, "/v1/chat/completions", tinyBody)
  return { ms: Math.round(performance.now() - t), status: r.status, text: r.body?.choices?.[0]?.message?.content ?? null }
}

const subjects = ["The river", "A lantern", "The old bridge", "Each harvest", "The north road", "A quiet bell", "The mill", "Every winter"]
const verbs = ["carried", "lit", "connected", "brought", "led to", "rang for", "ground", "covered"]
const objects = ["the valley towns", "the market square", "two farming villages", "grain and cider", "the coastal fort", "the evening prayers", "barley for bread", "the hills in snow"]
let filler = ""
for (let i = 0; i < fillerSentences; i++) {
  filler += `${subjects[i % 8]} ${verbs[(i * 3) % 8]} ${objects[(i * 5) % 8]} in year ${1500 + i}. `
}
const longBody = {
  model,
  messages: [{ role: "user", content: `Request ${Date.now()}: ${filler}\n\nIn one sentence, what is this text about?` }],
  max_tokens: maxTokens,
  temperature: 0,
  stream,
}

const idle = await timeTiny()

const controller = new AbortController()
const started = performance.now()
let firstTokenMs = null
const pending = (stream ? post(longBody, controller.signal) : postJson(base, "/v1/chat/completions", longBody, controller.signal))
  .then(async (r) => {
    if (!stream) return r
    const reader = r.body.getReader()
    const decoder = new TextDecoder()
    for (;;) {
      const { done, value } = await reader.read()
      if (done) return { status: r.status, done: true }
      if (firstTokenMs === null && decoder.decode(value).includes("\"content\"")) {
        firstTokenMs = Math.round(performance.now() - started)
      }
    }
  })
  .catch((e) => ({ error: e.name }))

let abortedAtMs = null
if (abortAfterMs > 0) {
  await new Promise((r) => setTimeout(r, abortAfterMs))
  controller.abort()
  abortedAtMs = Math.round(performance.now() - started)
}
const longResult = await pending
const longEndedMs = Math.round(performance.now() - started)
const after = await timeTiny()

console.log(JSON.stringify({
  model, mode, fillerSentences, maxTokens, abortAfterMs, idle, abortedAtMs, firstTokenMs, longEndedMs,
  longResult: longResult.error ?? longResult.status,
  longUsage: longResult.body?.usage ?? null,
  longText: longResult.body?.choices?.[0]?.message?.content ?? null,
  after,
}))
