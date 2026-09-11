#!/usr/bin/env node
/* Browser-level acceptance for math rendering.
 *
 * Requires `npm run build` first. Serves the compiled app from an ephemeral
 * loopback server and aborts every cross-origin request, so a green run
 * proves the lazy chunk and its fonts resolve from the LOCAL server -- which
 * is the only place they can come from once the app is embedded in the
 * engine binary.
 *
 * KaTeX does its work in an effect, so static rendering can only see the
 * fallback. This is the gate that proves the real thing:
 *   - KaTeX actually typesets, and the lazy chunk actually arrives
 *   - currency in a real rendered reply is still currency
 *   - a formula that re-renders on every streamed frame does not unmount the app
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { extname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { launchBrowser } from './lib/launch-browser.mjs'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const distDir = resolve(scriptDir, '../dist')
const ledgerPath = resolve(scriptDir, '../../ledger/camelid-ledger.json')
const MODEL_FILENAME = 'Qwen3-0.6B-Q8_0.gguf'

const MATH_PROMPT = 'Show me Euler'
const PRICE_PROMPT = 'What do the tiers cost'
const STREAMED_MATH_PROMPT = 'Stream me a formula'

const MATH_ANSWER = 'Inline $e^{i\\pi} + 1 = 0$ and a block:\n\n$$\\int_0^1 x^2\\,dx = \\frac{1}{3}$$'
const PRICE_ANSWER = 'The 8B costs $40 and the 27B costs $90 per month.'

const MIME = {
  '.css': 'text/css',
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = {
  ...ledger.capabilities,
  model_compatibility: ledger.model_rows.map((row) => row.contract),
}

const pageErrors = []
const externalRequests = []
const servedAssets = []

function sendJson(res, status, body) {
  const payload = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) })
  res.end(payload)
}

function sendFile(res, path) {
  const body = readFileSync(path)
  res.writeHead(200, { 'Content-Type': MIME[extname(path)] || 'application/octet-stream' })
  res.end(body)
}

async function readJsonBody(req) {
  const chunks = []
  for await (const chunk of req) chunks.push(chunk)
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : null
}

function sendChatCompletion(res, content) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
  res.socket?.setNoDelay(true)
  res.flushHeaders()
  const frame = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)
  frame({ choices: [{ delta: { role: 'assistant' } }] })
  frame({ choices: [{ delta: { content } }] })
  frame({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 20, completion_tokens: 30, total_tokens: 50 } })
  res.write('data: [DONE]\n\n')
  res.end()
}

/* Emits the answer one character at a time. A formula's TeX therefore changes
   on almost every frame, which is the case that caught the original defect:
   writing KaTeX output into a node React also renders children into makes the
   next reconcile remove children that are already gone, throwing
   NotFoundError and unmounting the whole app. A single-delta reply never
   re-renders a formula and would not have caught it. */
function sendChatCompletionStreamed(res, content) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
  res.socket?.setNoDelay(true)
  res.flushHeaders()
  const frame = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)
  frame({ choices: [{ delta: { role: 'assistant' } }] })
  let index = 0
  const tick = () => {
    if (res.writableEnded) return
    if (index >= content.length) {
      frame({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 20, completion_tokens: 30, total_tokens: 50 } })
      res.write('data: [DONE]\n\n')
      res.end()
      return
    }
    frame({ choices: [{ delta: { content: content[index] } }] })
    index += 1
    setTimeout(tick, 4)
  }
  tick()
}

function isFile(path) {
  try { return statSync(path).isFile() } catch { return false }
}

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true,
        engine: 'camelid',
        api_surface: 'full',
        version: 'math-browser-smoke',
        build: 'math-browser-smoke',
        backend: 'llama',
        model_family: 'qwen3',
        loaded_now: true,
        generation_ready: true,
        active_model_id: MODEL_FILENAME,
        active_context_length: 4096,
        max_prompt_tokens: 4096,
        max_generation_tokens: 8192,
      })
    }
    if (path === '/v1/models') {
      return sendJson(res, 200, {
        object: 'list',
        data: [{ id: MODEL_FILENAME, object: 'model', created: 0, owned_by: 'camelid', meta: { n_ctx_train: 32768, n_params: 600000000, size: 639446688 } }],
      })
    }
    if (path === '/api/capabilities') return sendJson(res, 200, capabilities)
    if (path === '/api/models/catalog/downloads') return sendJson(res, 200, [])
    if (path === '/api/models/local') {
      return sendJson(res, 200, {
        models_dir: 'models',
        models: [{
          filename: MODEL_FILENAME,
          size_bytes: 639446688,
          architecture: 'qwen3',
          quantization: 'Q8_0',
          admitted: true,
          oracle_qualified: true,
          chat_capable: true,
          generation_capable: true,
          context_length: 32768,
          lane_class: 'supported',
        }],
      })
    }
    if (path === '/api/models/current') {
      return sendJson(res, 200, {
        id: MODEL_FILENAME,
        path: `models/${MODEL_FILENAME}`,
        gguf: { metadata: { general: { architecture: 'qwen3', file_type: 7 } } },
        tokenizer: { status: 'available' },
      })
    }
    if (path === '/api/web/research' && req.method === 'POST') {
      return sendJson(res, 200, { status: 'skipped', triggered: false, reason: 'not_needed', sources: [], warnings: [] })
    }
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      const body = await readJsonBody(req)
      const lastUser = [...(body?.messages || [])].reverse().find((m) => m?.role === 'user')
      const text = typeof lastUser?.content === 'string' ? lastUser.content : ''
      if (text === PRICE_PROMPT) return sendChatCompletion(res, PRICE_ANSWER)
      if (text === STREAMED_MATH_PROMPT) return sendChatCompletionStreamed(res, MATH_ANSWER)
      return sendChatCompletion(res, MATH_ANSWER)
    }

    const filePath = resolve(distDir, `.${path}`)
    if (path !== '/' && isFile(filePath)) {
      servedAssets.push(path)
      return sendFile(res, filePath)
    }
    return sendFile(res, resolve(distDir, 'index.html'))
  } catch (error) {
    if (!res.writableEnded) sendJson(res, 500, { error: String(error) })
  }
})

await new Promise((done) => server.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${server.address().port}`

const browser = await launchBrowser({ purpose: 'the math browser smoke', headless: 'new' })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
page.on('pageerror', (error) => pageErrors.push(String(error)))
await page.setRequestInterception(true)
page.on('request', (request) => {
  const url = request.url()
  if (url.startsWith('data:') || url.startsWith('blob:')) return request.continue()
  try {
    if (new URL(url).origin === origin) return request.continue()
  } catch {
    // fall through and abort
  }
  externalRequests.push(url)
  return request.abort()
})
await page.evaluateOnNewDocument(() => {
  if (window.sessionStorage.getItem('camelid.mathSmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.mathSmokeInitialized', 'true')
})

async function sendPrompt(prompt) {
  await page.$eval('textarea[aria-label="Message Camelid"]:not([disabled])', (textarea, value) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, value)
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
  }, prompt)
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true'
  ), { timeout: 30000 })
  await page.click('button[aria-label="Send message"]')
  await page.waitForFunction(() => (
    !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming')
  ), { timeout: 30000 })
}

async function newChat() {
  await page.click('button.rail__new-chat')
  await page.waitForFunction(() => (
    document.querySelectorAll('main[data-view="chat"] .cxturn--assistant').length === 0
  ), { timeout: 30000 })
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') !== null
  ), { timeout: 30000 })

  /* ---- 1. KaTeX actually typesets -------------------------------------- */
  await sendPrompt(MATH_PROMPT)
  await page.waitForSelector('.cx-math .katex', { timeout: 30000 })
  const mathState = await page.evaluate(() => {
    const spans = [...document.querySelectorAll('.cx-math')]
    return {
      count: spans.length,
      typeset: spans.filter((node) => node.querySelector('.katex')).length,
      displayBlocks: spans.filter((node) => node.classList.contains('cx-math--display')).length,
      // KaTeX emits MathML alongside the visual HTML; its absence means the
      // output is decorative and unreadable to a screen reader.
      mathml: spans.filter((node) => node.querySelector('math')).length,
      /* KaTeX keeps the original TeX in a MathML <annotation>: that is what
         makes a formula copyable and legible to a screen reader, so it SHOULD
         be there. Only the visual layer must be free of control sequences. */
      rawTexVisible: spans.filter((node) => {
        const visual = node.querySelector('.katex-html')
        return visual ? /\\(int|pi|frac)/.test(visual.textContent) : false
      }).length,
      annotated: spans.filter((node) => node.querySelector('annotation')).length,
    }
  })
  assert.equal(mathState.count, 2, 'the reply has one inline and one display formula')
  assert.equal(mathState.typeset, 2, 'both formulas are typeset by KaTeX, not left as source')
  assert.equal(mathState.displayBlocks, 1, 'the block formula renders as display math')
  assert.equal(mathState.mathml, 2, 'KaTeX emits MathML so the formulas are not screen-reader-opaque')
  assert.equal(mathState.rawTexVisible, 0, 'no raw TeX control sequences remain in the visible layer')
  assert.equal(mathState.annotated, 2, 'the original TeX is kept as a MathML annotation, so the formula stays copyable and screen-reader-legible')

  assert.ok(
    servedAssets.some((path) => /katex/i.test(path)),
    'the KaTeX chunk was fetched from the local server, not bundled into the main entry',
  )
  assert.ok(
    servedAssets.some((path) => /KaTeX_.*\.woff2$/.test(path)),
    'KaTeX fonts resolve locally — a formula must not depend on a CDN',
  )

  /* ---- 2. currency in a real rendered reply is still currency ----------- */
  await newChat()
  await sendPrompt(PRICE_PROMPT)
  await page.waitForFunction((answer) => (
    [...document.querySelectorAll('.cxturn--assistant .cxturn__body')].some((n) => n.textContent.includes(answer))
  ), { timeout: 30000 }, PRICE_ANSWER)
  const priceState = await page.evaluate(() => ({
    mathSpans: document.querySelectorAll('main[data-view="chat"] .cx-math').length,
    text: [...document.querySelectorAll('.cxturn--assistant .cxturn__body')].map((n) => n.textContent).join(' '),
  }))
  assert.equal(priceState.mathSpans, 0, 'two prices in a sentence must not be typeset as a formula')
  assert.match(priceState.text, /costs \$40 and the 27B costs \$90/, 'the sentence reads exactly as written')

  /* ---- 3. a formula whose TeX changes on every frame -------------------- */
  /* This is the regression guard for the defect the first run of this smoke
     found: KaTeX writing into a React-owned node. It only reproduces when a
     formula RE-renders, which is every frame of a streamed reply. The
     pageErrors assertion at the end is what actually fails if it returns --
     the app unmounts and every later assertion would fail for the wrong
     reason, so keep this before them. */
  await newChat()
  await sendPrompt(STREAMED_MATH_PROMPT)
  await page.waitForSelector('.cx-math .katex', { timeout: 30000 })
  const streamedState = await page.evaluate(() => {
    const spans = [...document.querySelectorAll('.cx-math')]
    return {
      count: spans.length,
      typeset: spans.filter((node) => node.querySelector('.katex')).length,
      // A stale source line left beside typeset output means the placeholder
      // and the host both rendered at once.
      doubled: spans.filter((node) => node.querySelector('.katex') && node.querySelector('.cx-math__source')).length,
      appAlive: Boolean(document.querySelector('main[data-view="chat"]')),
    }
  })
  assert.equal(streamedState.appAlive, true, 'the app must survive a formula that re-renders on every streamed frame')
  assert.equal(streamedState.count, 2, 'the streamed reply settles on the same two formulas')
  assert.equal(streamedState.typeset, 2, 'both are typeset once the stream completes')
  assert.equal(streamedState.doubled, 0, 'typeset output replaces the source placeholder rather than sitting beside it')

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'nothing may be fetched off-origin — no CDN dependency')

  console.log('math browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
