#!/usr/bin/env node
/* Browser-level acceptance for regenerated-reply siblings.
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app and deterministic fixtures; every cross-origin request is
 * aborted.
 *
 * The pure smoke covers the data shape. What only a real render can prove:
 *
 *   - regenerating adds a version instead of replacing the reply
 *   - the regeneration request does NOT duplicate the question and does NOT
 *     show the model the answer it is meant to replace
 *   - switching version moves the token counts with the text
 *   - the NEXT turn sends whichever version is selected, which is the entire
 *     point of keeping them
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

const PROMPT = 'Name one animal.'
const FOLLOW_UP = 'Why that one?'
const ANSWERS = ['ANSWER-ALPHA is a camel.', 'ANSWER-BRAVO is a llama.', 'ANSWER-CHARLIE is a vicuna.']
const FOLLOW_UP_ANSWER = 'Because it suits the question.'

const MIME = {
  '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map((row) => row.contract) }

const chatRequests = []
const pageErrors = []
const externalRequests = []
let answerIndex = 0

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
/* Each answer reports a DIFFERENT completion_tokens so a footer showing the
   wrong version's telemetry is detectable, not merely suspected. */
function sendChatCompletion(res, content, completionTokens) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
  res.socket?.setNoDelay(true)
  res.flushHeaders()
  const frame = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)
  frame({ choices: [{ delta: { role: 'assistant' } }] })
  frame({ choices: [{ delta: { content } }] })
  frame({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 11, completion_tokens: completionTokens, total_tokens: 11 + completionTokens } })
  res.write('data: [DONE]\n\n')
  res.end()
}
function isFile(path) { try { return statSync(path).isFile() } catch { return false } }

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: 'full',
        version: 'variants-browser-smoke', build: 'variants-browser-smoke',
        backend: 'llama', model_family: 'qwen3', loaded_now: true, generation_ready: true,
        active_model_id: MODEL_FILENAME, active_context_length: 4096,
        max_prompt_tokens: 4096, max_generation_tokens: 8192,
      })
    }
    if (path === '/v1/models') {
      return sendJson(res, 200, { object: 'list', data: [{ id: MODEL_FILENAME, object: 'model', created: 0, owned_by: 'camelid', meta: { n_ctx_train: 32768, n_params: 600000000, size: 639446688 } }] })
    }
    if (path === '/api/capabilities') return sendJson(res, 200, capabilities)
    if (path === '/api/models/catalog/downloads') return sendJson(res, 200, [])
    if (path === '/api/models/local') {
      return sendJson(res, 200, {
        models_dir: 'models',
        models: [{ filename: MODEL_FILENAME, size_bytes: 639446688, architecture: 'qwen3', quantization: 'Q8_0', admitted: true, oracle_qualified: true, chat_capable: true, generation_capable: true, context_length: 32768, lane_class: 'supported' }],
      })
    }
    if (path === '/api/models/current') {
      return sendJson(res, 200, { id: MODEL_FILENAME, path: `models/${MODEL_FILENAME}`, gguf: { metadata: { general: { architecture: 'qwen3', file_type: 7 } } }, tokenizer: { status: 'available' } })
    }
    if (path === '/api/web/research' && req.method === 'POST') {
      return sendJson(res, 200, { status: 'skipped', triggered: false, reason: 'not_needed', sources: [], warnings: [] })
    }
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      const body = await readJsonBody(req)
      chatRequests.push(body)
      const lastUser = [...(body?.messages || [])].reverse().find((m) => m?.role === 'user')
      const text = typeof lastUser?.content === 'string' ? lastUser.content : ''
      if (text === FOLLOW_UP) return sendChatCompletion(res, FOLLOW_UP_ANSWER, 7)
      const content = ANSWERS[Math.min(answerIndex, ANSWERS.length - 1)]
      answerIndex += 1
      return sendChatCompletion(res, content, 100 + answerIndex)
    }
    const filePath = resolve(distDir, `.${path}`)
    if (path !== '/' && isFile(filePath)) return sendFile(res, filePath)
    return sendFile(res, resolve(distDir, 'index.html'))
  } catch (error) {
    if (!res.writableEnded) sendJson(res, 500, { error: String(error) })
  }
})

await new Promise((done) => server.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${server.address().port}`

const browser = await launchBrowser({ purpose: 'the reply-variants browser smoke', headless: 'new' })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
page.on('pageerror', (error) => pageErrors.push(String(error)))
await page.setRequestInterception(true)
page.on('request', (request) => {
  const url = request.url()
  if (url.startsWith('data:') || url.startsWith('blob:')) return request.continue()
  try { if (new URL(url).origin === origin) return request.continue() } catch { /* abort below */ }
  externalRequests.push(url)
  return request.abort()
})
await page.evaluateOnNewDocument(() => {
  if (window.sessionStorage.getItem('camelid.variantsSmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.variantsSmokeInitialized', 'true')
})

const idle = () => page.waitForFunction(() => (
  !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming')
), { timeout: 30000 })

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
  await idle()
}

const view = () => page.evaluate(() => {
  const turns = [...document.querySelectorAll('main[data-view="chat"] .cxturn--assistant')]
  const last = turns[turns.length - 1]
  return {
    assistantTurns: turns.length,
    text: last?.querySelector('.cxturn__body')?.textContent || '',
    nav: last?.querySelector('.cxturn__variant-count')?.textContent || null,
    footer: last?.querySelector('.cxturn__meta')?.textContent || '',
  }
})

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') !== null
  ), { timeout: 30000 })

  /* ---- 1. an ordinary reply shows no navigation ------------------------- */
  await sendPrompt(PROMPT)
  const first = await view()
  assert.match(first.text, /ANSWER-ALPHA/, 'the first answer renders')
  assert.equal(first.nav, null, 'a reply with one version shows no version control')

  /* ---- 2. regenerate ADDS a version ------------------------------------ */
  await page.click('button[aria-label^="Regenerate response"]')
  await idle()
  const second = await view()
  assert.equal(second.assistantTurns, 1, 'regenerating must not open a second reply bubble')
  assert.match(second.text, /ANSWER-BRAVO/, 'the new answer is shown')
  assert.doesNotMatch(second.text, /ANSWER-ALPHA/, 'and only one version is shown at a time')
  assert.equal(second.nav, '2/2', 'the first answer is kept as a sibling')
  assert.match(second.footer, /out 102/, 'the footer reports the NEW version’s output tokens')

  /* ---- 3. the regeneration request was well formed --------------------- */
  assert.equal(chatRequests.length, 2, 'regenerating issues exactly one more request')
  const rerollRoles = chatRequests[1].messages.map((m) => m.role)
  assert.equal(rerollRoles.at(-1), 'user', 'the re-roll ends on the question, so the model answers it again')
  assert.equal(
    chatRequests[1].messages.filter((m) => m.role === 'user' && m.content === PROMPT).length,
    1,
    'the question must appear ONCE — a re-roll adds no user turn of its own',
  )
  assert.equal(
    chatRequests[1].messages.filter((m) => typeof m.content === 'string' && m.content.includes('ANSWER-ALPHA')).length,
    0,
    'the model must not be shown the answer it is being asked to replace',
  )

  /* ---- 4. switching moves the telemetry with the text ------------------ */
  await page.click('button[aria-label="Previous version of this reply"]')
  await page.waitForFunction(() => (
    document.querySelector('.cxturn--assistant .cxturn__body')?.textContent.includes('ANSWER-ALPHA')
  ), { timeout: 10000 })
  const switched = await view()
  assert.equal(switched.nav, '1/2', 'the position updates')
  assert.match(switched.footer, /out 101/, 'and the footer reports THIS version’s tokens, not the other one’s')

  /* ---- 5. the next turn sends the SELECTED version --------------------- */
  await sendPrompt(FOLLOW_UP)
  const followUp = chatRequests[2]
  const assistantSent = followUp.messages.filter((m) => m.role === 'assistant')
  assert.equal(assistantSent.length, 1, 'one reply is sent forward, not both versions')
  assert.match(assistantSent[0].content, /ANSWER-ALPHA/, 'the SELECTED version is what the model continues from — the whole point of keeping them')
  assert.doesNotMatch(assistantSent[0].content, /ANSWER-BRAVO/, 'the unselected version stays out of the prompt')

  /* ---- 6. it survives a reload ------------------------------------------ */
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('main[data-view="chat"] .cxturn--assistant', { timeout: 30000 })
  const reloaded = await page.evaluate(() => {
    const turn = document.querySelector('main[data-view="chat"] .cxturn--assistant')
    return {
      nav: turn?.querySelector('.cxturn__variant-count')?.textContent || null,
      text: turn?.querySelector('.cxturn__body')?.textContent || '',
    }
  })
  assert.equal(reloaded.nav, '1/2', 'versions and the selection survive a reload')
  assert.match(reloaded.text, /ANSWER-ALPHA/, 'showing the version that was selected')

  /* ---- 7. discarding ---------------------------------------------------- */
  await page.click('button[aria-label="Discard this version"]')
  await page.waitForFunction(() => (
    document.querySelector('main[data-view="chat"] .cxturn--assistant .cxturn__variant-count') === null
  ), { timeout: 10000 })
  const afterDiscard = await page.evaluate(() => (
    document.querySelector('main[data-view="chat"] .cxturn--assistant .cxturn__body')?.textContent || ''
  ))
  assert.match(afterDiscard, /ANSWER-BRAVO/, 'discarding the shown version selects the neighbour')
  assert.equal(
    await page.$('main[data-view="chat"] .cxturn--assistant .cxturn__variant-count'),
    null,
    'and with one version left the control disappears again',
  )

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')

  console.log('message variants browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
