#!/usr/bin/env node
/* Browser-level acceptance for the in-chat document viewer and files tray.
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app and deterministic fixtures; every cross-origin request is
 * aborted.
 *
 *   - an attached document opens from its composer chip in a scrollable viewer
 *     showing exactly the text the server verified, rendered as text
 *   - a corrupted, deleted or unreachable document shows no text at all
 *   - a slow answer for a closed document cannot replace the one opened after it
 *   - a sent message keeps chips for the documents it was sent with, with the
 *     number of passages each contributed, and they survive a reload
 *   - generated files are listed in a tray above the composer that opens the
 *     files panel on the chosen file
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

const HOSTILE = '<img src=x onerror="window.__viewerXss=1"><script>window.__viewerXss=2</script>'
const LONG_TOKEN = 'unbroken'.repeat(60)
const END = 'END-OF-DOCUMENT-SENTINEL'
const STALE = 'STALE-TEXT-SENTINEL'
const SLOW_MS = 1500
const TEXT = [
  'Camelid Support Policy',
  ...Array.from({ length: 300 }, (_, i) => `Clause ${i + 1}. Enterprise refunds are handled within 60 days.`),
  `Markup stays text: ${HOSTILE}`,
  LONG_TOKEN,
  END,
].join('\n')

const DOCS = [
  { doc_id: 'doc-ok', filename: `${HOSTILE}.md`, chunk_count: 42, byte_size: 20000 },
  { doc_id: 'doc-corrupted', filename: 'corrupted.txt', chunk_count: 3, byte_size: 900 },
  { doc_id: 'doc-gone', filename: 'deleted.txt', chunk_count: 2, byte_size: 400 },
  { doc_id: 'doc-drop', filename: 'dropped.txt', chunk_count: 1, byte_size: 100 },
  { doc_id: 'doc-slow', filename: 'slow.txt', chunk_count: 1, byte_size: 100 },
]
const bind = (n) => ({ chunk_sha256: String(n).repeat(64).slice(0, 64), doc_sha256: 'd'.repeat(64) })
const RESULTS = [
  { doc_id: 'doc-ok', filename: DOCS[0].filename, chunk_index: 0, excerpt: 'Clause 1.', byte_start: 23, byte_end: 32, ...bind(1) },
  { doc_id: 'doc-ok', filename: DOCS[0].filename, chunk_index: 5, excerpt: 'Clause 9.', byte_start: 500, byte_end: 509, ...bind(2) },
]
const CODE = "print('refund window: 60 days')"
const JSON_OUT = '{"refund_days": 60}'
const ANSWER = `Refunds are handled within 60 days [1] [2].\n\n\`\`\`python\n${CODE}\n\`\`\`\n\n\`\`\`json\n${JSON_OUT}\n\`\`\`\n`

const MIME = {
  '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map((row) => row.contract) }

const sourceRequests = []
const chatRequests = []
const pageErrors = []
const externalRequests = []

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
  frame({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 40, completion_tokens: 30, total_tokens: 70 } })
  res.write('data: [DONE]\n\n')
  res.end()
}
function isFile(path) { try { return statSync(path).isFile() } catch { return false } }

async function answerSource(req, res, docId) {
  sourceRequests.push(docId)
  switch (docId) {
    case 'doc-ok': return sendJson(res, 200, { doc_id: docId, filename: DOCS[0].filename, doc_sha256: 'd'.repeat(64), text: TEXT })
    case 'doc-corrupted':
      // A refusal carrying text must still show none of it.
      return sendJson(res, 409, { text: `${STALE} ${TEXT}`, error: { code: 'document_source_corrupted', message: 'The stored text no longer matches its recorded hash, so it is not shown.' } })
    case 'doc-drop': return req.socket.destroy()
    case 'doc-slow':
      await new Promise((done) => setTimeout(done, SLOW_MS))
      return sendJson(res, 200, { doc_id: docId, filename: 'slow.txt', doc_sha256: 'e'.repeat(64), text: 'SLOW-RESPONSE-TEXT' })
    default: return sendJson(res, 404, { error: { code: 'document_not_found', message: 'This document is no longer in the library.' } })
  }
}

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: 'full',
        version: 'document-viewer-browser-smoke', build: 'document-viewer-browser-smoke',
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
    if (path === '/api/documents' && req.method === 'GET') {
      return sendJson(res, 200, DOCS.map((doc) => ({ id: doc.doc_id, filename: doc.filename, file_type: 'txt', byte_size: doc.byte_size, chunk_count: doc.chunk_count, created_at: 1 })))
    }
    const source = path.match(/^\/api\/documents\/([^/]+)\/source$/)
    if (source && req.method === 'GET') return answerSource(req, res, decodeURIComponent(source[1]))
    if (path === '/api/documents/search' && req.method === 'POST') {
      await readJsonBody(req)
      return sendJson(res, 200, { results: RESULTS.map((r) => ({ score: 0.5, retrieval: 'keyword', ...r })) })
    }
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      chatRequests.push(await readJsonBody(req))
      return sendChatCompletion(res, ANSWER)
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

const browser = await launchBrowser({ purpose: 'the document viewer browser smoke', headless: 'new' })
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
await page.evaluateOnNewDocument((docs) => {
  if (window.sessionStorage.getItem('camelid.documentViewerSmokeInitialized')) return
  window.localStorage.clear()
  window.localStorage.setItem('camelid.attachedDocuments', JSON.stringify(docs))
  window.sessionStorage.setItem('camelid.documentViewerSmokeInitialized', 'true')
}, DOCS)

const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const idle = () => page.waitForFunction(() => (
  !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming')
), { timeout: 30000 })

async function sendPrompt(prompt) {
  const before = chatRequests.length
  await page.$eval('textarea[aria-label="Message Camelid"]:not([disabled])', (textarea, value) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, value)
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
  }, prompt)
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true'
  ), { timeout: 30000 })
  await page.click('button[aria-label="Send message"]')
  const deadline = Date.now() + 30000
  while (chatRequests.length === before && Date.now() < deadline) await sleep(50)
  assert.ok(chatRequests.length > before, 'the prompt reached the model endpoint')
  await page.waitForSelector('main[data-view="chat"] button.citation-pill', { timeout: 30000 })
  await idle()
}

const composerChip = (filename) => page.evaluateHandle((name) => (
  [...document.querySelectorAll('.cxcomposer__doc-open')].find((node) => node.getAttribute('title') === `Open ${name}`)
), filename)
const messageChip = (filename) => page.evaluateHandle((name) => (
  [...document.querySelectorAll('.cxturn__user-doc')].find((node) => node.getAttribute('title') === `Open ${name}`)
), filename)

async function openFrom(handle) {
  const element = handle.asElement()
  assert.ok(element, 'the chip exists')
  await element.click()
  await page.waitForSelector('.document-viewer', { timeout: 10000 })
}
async function settled() {
  await page.waitForFunction(() => {
    const badge = document.querySelector('.document-viewer .citation-modal__badge')
    return badge && !badge.textContent.includes('Checking')
  }, { timeout: 10000 })
}
async function closeWithEscape() {
  await page.keyboard.press('Escape')
  await page.waitForFunction(() => !document.querySelector('.document-viewer'), { timeout: 10000 })
}
const viewer = () => page.evaluate(() => {
  const root = document.querySelector('.document-viewer')
  if (!root) return null
  const text = root.querySelector('.document-viewer__text')
  const rect = root.getBoundingClientRect()
  return {
    role: root.getAttribute('role'),
    title: root.querySelector('.cx-modal__title')?.textContent ?? null,
    badge: root.querySelector('.citation-modal__badge')?.textContent || '',
    badgeClass: root.querySelector('.citation-modal__badge')?.className || '',
    text: text?.textContent ?? null,
    refusalCode: root.querySelector('.citation-modal__refusal-code')?.textContent ?? null,
    all: root.textContent,
    markup: root.querySelectorAll('img, script').length,
    scrollable: text ? text.scrollHeight - text.clientHeight : 0,
    overflowX: text ? text.scrollWidth - text.clientWidth : 0,
    fits: rect.top >= 0 && rect.bottom <= window.innerHeight,
  }
})

function assertRefused(view, code, label) {
  assert.ok(view, `${label}: the viewer opens`)
  assert.match(view.badge, /Refused/, `${label}: the badge says Refused`)
  assert.match(view.badgeClass, /--refused/, `${label}: with the refused style`)
  assert.equal(view.refusalCode, code, `${label}: the refusal names the typed reason`)
  assert.equal(view.text, null, `${label}: no document text is shown`)
  assert.doesNotMatch(view.all, /Clause 1\.|SLOW-RESPONSE-TEXT|STALE-TEXT-SENTINEL/, `${label}: not even another document's text`)
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await page.waitForSelector('.cxcomposer__doc-open', { timeout: 30000 })
  assert.equal(await page.$('.files-tray'), null, 'no files tray before anything was generated')

  /* ---- 1. a composer chip opens the verified text, scrollable ----------- */
  await openFrom(await composerChip(DOCS[0].filename))
  await settled()
  const ok = await viewer()
  assert.equal(ok.role, 'dialog', 'the viewer is a dialog')
  assert.equal(ok.title, DOCS[0].filename, 'titled with the literal filename')
  assert.match(ok.badge, /Verified/, 'the badge says Verified')
  assert.equal(ok.text, TEXT, 'the viewer shows exactly the text the server verified')
  assert.equal(ok.markup, 0, 'hostile filenames and text render as text, not markup')
  assert.equal(await page.evaluate(() => window.__viewerXss), undefined, 'and none of it executed')
  assert.ok(ok.scrollable > 1000, `a long document scrolls inside the viewer (${ok.scrollable}px)`)
  assert.ok(ok.overflowX <= 1, `a long unbroken token wraps (overflow ${ok.overflowX}px)`)
  assert.ok(ok.fits, 'the viewer fits in the window')
  const scrolled = await page.$eval('.document-viewer__text', (node) => {
    node.scrollTop = node.scrollHeight
    return node.scrollTop
  })
  assert.ok(scrolled > 0, 'the text can be scrolled to its end')
  await closeWithEscape()
  assert.equal(
    await page.evaluate(() => document.activeElement?.getAttribute('title')),
    `Open ${DOCS[0].filename}`,
    'closing returns focus to the chip that opened it',
  )

  /* ---- 2. a document that cannot be verified shows no text -------------- */
  await openFrom(await composerChip('corrupted.txt'))
  await settled()
  assertRefused(await viewer(), 'document_source_corrupted', 'a corrupted document')
  await page.click('.document-viewer button[aria-label="Close"]')
  await page.waitForFunction(() => !document.querySelector('.document-viewer'), { timeout: 10000 })

  await openFrom(await composerChip('deleted.txt'))
  await settled()
  assertRefused(await viewer(), 'document_not_found', 'a deleted document')
  await closeWithEscape()

  await openFrom(await composerChip('dropped.txt'))
  await settled()
  assertRefused(await viewer(), 'document_source_unreachable', 'a dropped connection')
  await closeWithEscape()

  /* ---- 3. a slow answer cannot replace the document opened after it ----- */
  await openFrom(await composerChip('slow.txt'))
  assert.match((await viewer()).badge, /Checking/, 'loading is visible')
  assert.equal((await viewer()).text, null, 'and shows no text yet')
  await closeWithEscape()
  await openFrom(await composerChip(DOCS[0].filename))
  await settled()
  await sleep(SLOW_MS + 500)
  const afterRace = await viewer()
  assert.equal(afterRace.title, DOCS[0].filename, 'the viewer still describes the document opened last')
  assert.equal(afterRace.text, TEXT, 'with its own text')
  assert.doesNotMatch(afterRace.all, /SLOW-RESPONSE-TEXT/, 'the late answer for the closed document is discarded')
  await closeWithEscape()

  /* ---- 4. a sent message keeps its documents ---------------------------- */
  await sendPrompt('What is the refund window?')
  const chips = await page.$$eval('.cxturn__user-doc', (nodes) => nodes.map((node) => ({
    name: node.querySelector('.cxturn__user-doc-name')?.textContent,
    meta: node.querySelector('.cxturn__user-doc-meta')?.textContent,
  })))
  assert.deepEqual(chips, [
    { name: DOCS[0].filename, meta: '2 passages used' },
    { name: 'corrupted.txt', meta: 'no passages used' },
    { name: 'deleted.txt', meta: 'no passages used' },
    { name: 'dropped.txt', meta: 'no passages used' },
    { name: 'slow.txt', meta: 'no passages used' },
  ], 'the message lists every document it was sent with and what each contributed')
  const beforeOpen = sourceRequests.length
  await openFrom(await messageChip(DOCS[0].filename))
  await settled()
  assert.equal((await viewer()).text, TEXT, 'a message chip opens the same verified text')
  assert.equal(sourceRequests.length, beforeOpen + 1, 'opening asks the server; nothing is cached')
  await closeWithEscape()

  /* ---- 5. generated files are pinned above the composer ----------------- */
  await page.waitForSelector('.cxchat__dock .files-tray', { timeout: 10000 })
  const tray = await page.$eval('.files-tray', (root) => ({
    head: root.querySelector('.files-tray__head')?.textContent,
    expanded: root.querySelector('.files-tray__head')?.getAttribute('aria-expanded'),
    names: [...root.querySelectorAll('.files-tray__name')].map((node) => node.textContent),
  }))
  assert.equal(tray.head, '2 files from this conversation', 'the tray counts the generated files')
  assert.equal(tray.expanded, 'true', 'and starts expanded')
  assert.equal(tray.names.length, 2, 'listing each')
  await page.click('.files-tray__file')
  await page.waitForSelector('aside.conversation-files', { timeout: 10000 })
  assert.equal(await page.$eval('input[aria-label="Output filename"]', (node) => node.value), tray.names[0], 'the files panel opens on the chosen file, not the latest')
  const panelText = await page.$eval('aside.conversation-files', (node) => node.textContent)
  assert.ok(panelText.includes(CODE), 'showing its content')
  assert.ok(!panelText.includes(JSON_OUT), 'and not the other file')
  await page.click('button[aria-label="Close conversation files"]')
  await page.click('.files-tray__head')
  assert.equal(await page.$eval('.files-tray__head', (node) => node.getAttribute('aria-expanded')), 'false', 'the tray collapses')
  assert.equal(await page.$('.files-tray__list'), null, 'hiding the list')

  /* ---- 6. chips survive a reload and still verify ----------------------- */
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('.cxturn__user-doc', { timeout: 30000 })
  assert.equal(await page.$$eval('.cxturn__user-doc', (nodes) => nodes.length), DOCS.length, 'every message chip is restored')
  await openFrom(await messageChip('corrupted.txt'))
  await settled()
  assertRefused(await viewer(), 'document_source_corrupted', 'a restored chip for a corrupted document')
  await closeWithEscape()

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')

  console.log('document viewer browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
