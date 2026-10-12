import { pathToFileURL } from "node:url"
const [port, frontendDir] = process.argv.slice(2)
const memory = await import(pathToFileURL(frontendDir + "/src/lib/memory.js").href)
const base = `http://127.0.0.1:${port}`
const model = (await (await fetch(`${base}/v1/health`)).json()).active_model_id
const tiny = () => ({ model, messages: [{ role: "user", content: "Say hi." }], max_tokens: 4, temperature: 0, stream: false })
const timeTiny = async () => { const t = performance.now(); const r = await fetch(`${base}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(tiny()) }); await r.json(); return Math.round(performance.now() - t) }
const idle = await timeTiny()
console.log("tiny chat, idle engine:", idle, "ms")
const controller = new AbortController()
const started = performance.now()
const pending = fetch(`${base}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
  body: JSON.stringify(memory.suggestionRequest({ model, userText: "Hi, I am Priya. I work as a nurse in Leeds.", constrained: false })) }).then(r => r.json()).catch(e => e.name)
await new Promise(r => setTimeout(r, 10000))
controller.abort()
console.log("suggestion request cancelled after", Math.round(performance.now() - started), "ms:", await pending)
const after = await timeTiny()
console.log("tiny chat right after the cancel:", after, "ms")
