// A request cancelled while it waits in the engine queue must never run.
// node queue.mjs <port> <fillerSentences>
// A: long prompt, not cancelled. B: the same long prompt with one word changed (so
// the prompt cache cannot serve it), sent while A runs and cancelled while still
// queued. C: a tiny request sent right after B is cancelled.
// If B is skipped, C finishes about when A does; if B ran, C waits for B's prompt read too.
// Prints one JSON line.
import { postJson } from "./http-json.mjs"
const [port, fillerArg] = process.argv.slice(2)
const base = `http://127.0.0.1:${port}`
const model = (await (await fetch(`${base}/v1/health`)).json()).active_model_id
const post = (body, signal) => postJson(base, "/v1/chat/completions", body, signal)
const subjects = ["The river", "A lantern", "The old bridge", "Each harvest", "The north road", "A quiet bell", "The mill", "Every winter"]
const verbs = ["carried", "lit", "connected", "brought", "led to", "rang for", "ground", "covered"]
const objects = ["the valley towns", "the market square", "two farming villages", "grain and cider", "the coastal fort", "the evening prayers", "barley for bread", "the hills in snow"]
let filler = ""
for (let i = 0; i < Number(fillerArg); i++) filler += `${subjects[i % 8]} ${verbs[(i * 3) % 8]} ${objects[(i * 5) % 8]} in year ${1500 + i}. `
const long = (lead) => ({ model, messages: [{ role: "user", content: `${lead} ${filler}\n\nIn one sentence, what is this text about?` }], max_tokens: 8, temperature: 0, stream: false })

const t0 = performance.now()
const at = () => Math.round(performance.now() - t0)
const a = post(long("First:")).then((r) => ({ status: r.status, endMs: at(), usage: r.body?.usage }))
await new Promise((r) => setTimeout(r, 1500))
const controller = new AbortController()
const b = post(long("Second:"), controller.signal).then((r) => ({ status: r.status })).catch((e) => ({ error: e.name, endMs: at() }))
await new Promise((r) => setTimeout(r, 1500))
controller.abort()
const bOut = await b
const cStart = at()
const c = await post({ model, messages: [{ role: "user", content: "Say hi." }], max_tokens: 4, temperature: 0, stream: false })
const cEnd = at()
const aOut = await a
console.log(JSON.stringify({ model, fillerSentences: Number(fillerArg), a: aOut, b: bOut, c: { status: c.status, startMs: cStart, endMs: cEnd, waitedMs: cEnd - cStart }, cEndMinusAEndMs: cEnd - aOut.endMs }))
