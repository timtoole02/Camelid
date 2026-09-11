#!/usr/bin/env node
/* Browser-level acceptance for Mermaid diagram rendering.
 *
 * Requires `npm run build` first. Serves the compiled app from an ephemeral
 * loopback server and aborts every cross-origin request, so a green run
 * proves the lazy Mermaid chunks resolve from the LOCAL server -- which is
 * the only place they can come from once the app is embedded in the engine
 * binary.
 *
 * Mermaid draws in an effect, so static rendering can only see the fallback.
 * This is the gate that proves the real thing:
 *   - Mermaid actually draws an SVG, from this diagram's own source
 *   - the SVG it generates carries no script
 *   - a theme switch redraws the diagram without unmounting the app -- the
 *     redraw writes into the host node a second time, which is exactly the
 *     React-owned-node hazard the math smoke guards for KaTeX
 *   - a malformed diagram falls back to the code card instead of a blank box
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

const DIAGRAM_PROMPT = 'Draw the pipeline'
const BROKEN_DIAGRAM_PROMPT = 'Draw something broken'

const DIAGRAM_ANSWER = '```mermaid\ngraph TD;\n  Prompt-->Prefill;\n  Prefill-->Decode;\n```'
const BROKEN_DIAGRAM_ANSWER = '```mermaid\nthis is definitely not a diagram {{{\n```'

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
        version: 'diagram-browser-smoke',
        build: 'diagram-browser-smoke',
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
      if (text === BROKEN_DIAGRAM_PROMPT) return sendChatCompletion(res, BROKEN_DIAGRAM_ANSWER)
      return sendChatCompletion(res, DIAGRAM_ANSWER)
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

const browser = await launchBrowser({ purpose: 'the diagram browser smoke', headless: 'new' })
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
  if (window.sessionStorage.getItem('camelid.diagramSmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.diagramSmokeInitialized', 'true')
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

  /* ---- 1. Mermaid actually draws --------------------------------------- */
  const scriptsBeforeDiagram = servedAssets.filter((path) => path.endsWith('.js')).length
  await sendPrompt(DIAGRAM_PROMPT)
  await page.waitForFunction(() => (
    document.querySelector('.cx-mermaid[data-mermaid-state="ready"] svg') !== null
  ), { timeout: 30000 })
  const diagramState = await page.evaluate(() => {
    const figure = document.querySelector('.cx-mermaid')
    const svg = figure?.querySelector('svg')
    return {
      state: figure?.getAttribute('data-mermaid-state'),
      hasSvg: Boolean(svg),
      // Proves it drew THIS diagram rather than an empty canvas.
      labels: svg ? svg.textContent : '',
      hasScript: Boolean(svg?.querySelector('script')),
      sourceHidden: !figure?.querySelector('.cx-mermaid__source'),
    }
  })
  assert.equal(diagramState.state, 'ready', 'the diagram reached the ready state')
  assert.equal(diagramState.hasSvg, true, 'Mermaid produced an SVG')
  assert.match(diagramState.labels, /Prefill/, 'the SVG contains this diagram’s own node labels')
  assert.equal(diagramState.hasScript, false, 'the generated SVG carries no script element')
  assert.equal(diagramState.sourceHidden, true, 'the source panel starts collapsed')
  assert.ok(
    servedAssets.filter((path) => path.endsWith('.js')).length > scriptsBeforeDiagram,
    'drawing fetched Mermaid’s lazy chunks from the local server rather than shipping them in the main entry',
  )

  await page.click('.cx-mermaid__bar button')
  await page.waitForSelector('.cx-mermaid__source', { timeout: 10000 })
  const revealed = await page.$eval('.cx-mermaid__source', (node) => node.textContent)
  assert.match(revealed, /graph TD/, 'the diagram source can be revealed')

  /* ---- 2. a theme switch redraws without unmounting the app ------------- */
  /* The SVG bakes its colours in, so the component redraws on a theme change
     -- which writes into the host node a second time. If that node were also
     React-owned, this is where the app would unmount. Waiting for a NEW svg
     id proves the redraw actually happened rather than being skipped. */
  const firstSvgId = await page.$eval('.cx-mermaid__svg svg', (node) => node.id)
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'))
  await page.waitForFunction((previousId) => {
    const svg = document.querySelector('.cx-mermaid[data-mermaid-state="ready"] .cx-mermaid__svg svg')
    return Boolean(svg) && svg.id !== previousId
  }, { timeout: 30000 }, firstSvgId)
  const afterTheme = await page.evaluate(() => ({
    svgs: document.querySelectorAll('.cx-mermaid__svg svg').length,
    appAlive: Boolean(document.querySelector('main[data-view="chat"]')),
    sourceStillOpen: Boolean(document.querySelector('.cx-mermaid__source')),
  }))
  assert.equal(afterTheme.appAlive, true, 'the app survives a diagram redraw')
  assert.equal(afterTheme.svgs, 1, 'the redraw replaces the diagram rather than stacking a second one')
  assert.equal(afterTheme.sourceStillOpen, true, 'and keeps the reader’s open source panel')

  /* ---- 3. a malformed diagram degrades to the code card ----------------- */
  await newChat()
  await sendPrompt(BROKEN_DIAGRAM_PROMPT)
  await page.waitForFunction(() => (
    document.querySelector('main[data-view="chat"] .message-code-card') !== null
  ), { timeout: 30000 })
  const brokenState = await page.evaluate(() => ({
    figures: document.querySelectorAll('.cx-mermaid').length,
    codeCards: document.querySelectorAll('main[data-view="chat"] .message-code-card').length,
    text: document.querySelector('main[data-view="chat"] .message-code-card')?.textContent || '',
    // Mermaid appends an error node to <body> on a parse failure.
    strayErrorNodes: document.querySelectorAll('body > svg[id^="dcx-mermaid"], body > div[id^="dcx-mermaid"]').length,
  }))
  assert.equal(brokenState.figures, 0, 'an unparseable diagram leaves no empty diagram box')
  assert.equal(brokenState.codeCards, 1, 'it falls back to the ordinary code card')
  assert.match(brokenState.text, /not a diagram/, 'and the reader still gets the text of it')
  assert.equal(brokenState.strayErrorNodes, 0, 'Mermaid’s failure node is cleaned up, not left on the page')

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'nothing may be fetched off-origin — no CDN dependency')

  console.log('diagram browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
