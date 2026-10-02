#!/usr/bin/env node
/* Browser-level acceptance for semantic document search in chat.
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app and deterministic fixtures; every cross-origin request is
 * aborted.
 *
 *   - an attached document shows its background indexing progress, and the
 *     page stops polling once indexing is done
 *   - a document waiting for the indexer says so and is still polled
 *   - a stopped indexer is shown as stopped, with its error, and is not
 *     polled; a retry running after a failure shows progress again
 *   - a missing or foreign encoder says keyword search only, and offers the
 *     Models page when the encoder is simply not installed
 *   - a deliberately disabled encoder or a failing status endpoint shows
 *     nothing and breaks nothing
 *   - the chat still searches in the default mode
 *   - the citation viewer says how each passage was found
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
const POLL_MS = 1500

const DOC = { doc_id: 'doc-ok', filename: 'policy.md', chunk_count: 48, byte_size: 20000 }
const status = (semantic, indexed, skipped = 0) => ({
  semantic: { encoder: 'nomic-embed-text-v1.5.Q8_0.gguf', ...semantic },
  documents: [{ id: DOC.doc_id, indexable_chunks: 48, indexed_chunks: indexed, skipped_chunks: skipped }],
})
const READY = { available: true, indexing: false }
const INDEXING = { available: true, indexing: true }
const FAILURE = 'chunk embedding failed: the encoder ran out of memory'
const STOPPED = { available: true, indexing: false, error: FAILURE }
const RETRYING = { available: true, indexing: true, error: FAILURE }
const NOT_INSTALLED = {
  available: false, indexing: false, reason: 'encoder_not_installed',
  message: 'Semantic document search needs nomic-embed-text-v1.5.Q8_0.gguf in the models directory.',
}
const DISABLED = {
  available: false, indexing: false, reason: 'semantic_disabled',
  message: 'Semantic document search is turned off by CAMELID_DOCUMENT_SEMANTIC.',
}

const bind = (n) => ({ chunk_sha256: String(n).repeat(64).slice(0, 64), doc_sha256: 'd'.repeat(64) })
const RESULTS = [
  { chunk_index: 3, retrieval: 'semantic', excerpt: 'Refunds within 60 days.', byte_start: 100, byte_end: 123, ...bind(1) },
  { chunk_index: 4, retrieval: 'hybrid', excerpt: 'Refunds go to the card.', byte_start: 130, byte_end: 153, ...bind(2) },
  { chunk_index: 5, retrieval: 'keyword', excerpt: 'Trials are not refunded.', byte_start: 160, byte_end: 184, ...bind(3) },
  { chunk_index: 0, retrieval: 'attached', excerpt: 'Camelid Support Policy', byte_start: 0, byte_end: 22, ...bind(4) },
].map((result) => ({ doc_id: DOC.doc_id, filename: DOC.filename, score: 0.03, ...result }))
const ANSWER = 'Refunds take 60 days [1] [2] [3] [4].'

const MIME = {
  '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map((row) => row.contract) }

let statusScript = []
let statusFailing = false
let statusCalls = 0
const searchRequests = []
const chatRequests = []
const pageErrors = []
const externalRequests = []

function sendJson(res, statusCode, body) {
  const payload = JSON.stringify(body)
  res.writeHead(statusCode, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) })
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

function answerStatus(res) {
  statusCalls += 1
  if (statusFailing) {
    res.writeHead(500, { 'Content-Type': 'text/plain' })
    return res.end('internal failure, not json')
  }
  const next = statusScript.length > 1 ? statusScript.shift() : statusScript[0]
  return sendJson(res, 200, next)
}

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: 'full',
        version: 'semantic-index-browser-smoke', build: 'semantic-index-browser-smoke',
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
      return sendJson(res, 200, [{ id: DOC.doc_id, filename: DOC.filename, file_type: 'md', byte_size: DOC.byte_size, chunk_count: DOC.chunk_count, created_at: 1 }])
    }
    if (path === '/api/documents/index-status' && req.method === 'GET') return answerStatus(res)
    if (path === '/api/documents/search' && req.method === 'POST') {
      searchRequests.push(await readJsonBody(req))
      return sendJson(res, 200, { results: RESULTS, retrieval: { mode: 'hybrid', semantic: { available: true, indexable_chunks: 48, indexed_chunks: 48, skipped_chunks: 0 } } })
    }
    if (path === '/api/documents/citation/resolve' && req.method === 'POST') {
      const body = await readJsonBody(req)
      const hit = RESULTS.find((result) => result.chunk_index === body?.chunk_index)
      return sendJson(res, 200, { ...hit, before: '', span: hit.excerpt, after: '' })
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

const browser = await launchBrowser({ purpose: 'the semantic index browser smoke', headless: 'new' })
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
await page.evaluateOnNewDocument((doc) => {
  if (window.sessionStorage.getItem('camelid.semanticSmokeInitialized')) return
  window.localStorage.clear()
  window.localStorage.setItem('camelid.attachedDocuments', JSON.stringify([doc]))
  window.sessionStorage.setItem('camelid.semanticSmokeInitialized', 'true')
}, DOC)

const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const chipText = () => page.$eval('.cxcomposer__doc-open .cxcomposer__doc-chunks', (node) => node.textContent)
const note = () => page.$eval('.cxcomposer__semantic-note', (node) => node.textContent).catch(() => null)

async function load(script, { failing = false } = {}) {
  statusScript = script
  statusFailing = failing
  const before = statusCalls
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await page.waitForSelector('.cxcomposer__doc-open', { timeout: 30000 })
  const deadline = Date.now() + 10000
  while (statusCalls === before && Date.now() < deadline) await sleep(50)
  assert.ok(statusCalls > before, 'an attached document asks for its index status')
  await sleep(300)
}

const idle = () => page.waitForFunction(() => (
  !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming')
), { timeout: 30000 })

try {
  /* ---- 1. indexing progress, then the poll stops ------------------------ */
  await load([status(INDEXING, 0), status(INDEXING, 24), status(READY, 48)])
  await page.waitForFunction(() => document.querySelector('.cxcomposer__doc-chunks--indexing')?.textContent.includes('24/48'), { timeout: POLL_MS * 4 })
  assert.match(await page.$eval('.cxcomposer__doc-chunks--indexing', (node) => node.getAttribute('title')), /Keyword search works meanwhile/, 'the progress explains itself')
  await page.waitForFunction(() => document.querySelector('.cxcomposer__doc-open .cxcomposer__doc-chunks')?.textContent === '48 chunks', { timeout: POLL_MS * 4 })
  const callsWhenDone = statusCalls
  await sleep(POLL_MS * 3)
  assert.equal(statusCalls, callsWhenDone, 'polling stops once the document is indexed')
  assert.equal(await note(), null, 'an available encoder needs no note')

  /* ---- 2. a skipped chunk still completes the progress ------------------ */
  await load([status(READY, 47, 1)])
  assert.equal(await chipText(), '48 chunks', 'a chunk skipped for failing its hash does not leave indexing stuck')

  /* ---- 3. waiting for the indexer: say so, keep polling ----------------- */
  await load([status(READY, 0), status(INDEXING, 12), status(READY, 48)])
  assert.equal(await chipText(), 'waiting to index 0/48', 'incomplete coverage with no indexer running is not shown as indexing')
  await page.waitForFunction(() => document.querySelector('.cxcomposer__doc-chunks--indexing')?.textContent === 'indexing 12/48', { timeout: POLL_MS * 4 })
  await page.waitForFunction(() => document.querySelector('.cxcomposer__doc-open .cxcomposer__doc-chunks')?.textContent === '48 chunks', { timeout: POLL_MS * 4 })

  /* ---- 4. a stopped indexer: show it and its error, stop polling -------- */
  await load([status(STOPPED, 0)])
  assert.equal(await chipText(), 'indexing stopped 0/48')
  assert.equal(await page.$('.cxcomposer__doc-chunks--indexing'), null, 'a stopped indexer is not shown as progress')
  assert.match(await page.$eval('.cxcomposer__doc-chunks--stopped', (node) => node.getAttribute('title')), /Keyword search still works/)
  assert.match(await note(), /Indexing for search by meaning stopped:/)
  assert.equal(await page.$eval('.cxcomposer__semantic-error', (node) => node.textContent), FAILURE, 'the reported error is shown as reported')
  assert.match(await note(), /when a document is added or Camelid restarts/)
  const callsWhenStopped = statusCalls
  await sleep(POLL_MS * 3)
  assert.equal(statusCalls, callsWhenStopped, 'a stopped indexer is not polled')

  await load([status(RETRYING, 10), status(READY, 48)])
  assert.equal(await chipText(), 'indexing 10/48', 'a retry after a failure shows progress')
  assert.equal(await note(), null, 'the previous failure is not reported while the retry runs')
  await page.waitForFunction(() => document.querySelector('.cxcomposer__doc-open .cxcomposer__doc-chunks')?.textContent === '48 chunks', { timeout: POLL_MS * 4 })

  /* ---- 5. no encoder installed: say so, offer Models, do not poll ------- */
  await load([status(NOT_INSTALLED, 0)])
  await page.waitForSelector('.cxcomposer__semantic-note', { timeout: 5000 })
  assert.match(await note(), /Keyword search only\. Semantic document search needs nomic-embed-text-v1\.5\.Q8_0\.gguf/)
  assert.equal(await chipText(), '48 chunks', 'no progress is shown when nothing will be indexed')
  const callsWhenUnavailable = statusCalls
  await sleep(POLL_MS * 2)
  assert.equal(statusCalls, callsWhenUnavailable, 'an unavailable encoder is not polled')
  await page.click('.cxcomposer__semantic-note button')
  await page.waitForSelector('main[data-view="library"]', { timeout: 10000 })
  await page.evaluate(() => [...document.querySelectorAll('nav[aria-label="Primary"] button')].find((button) => button.textContent.trim() === 'Chat').click())
  await page.waitForSelector('main[data-view="chat"]', { timeout: 10000 })

  /* ---- 6. disabled on purpose, or a failing endpoint: nothing shown ----- */
  await load([status(DISABLED, 0)])
  assert.equal(await note(), null, 'an operator who turned it off is not nagged')
  await load([], { failing: true })
  assert.equal(await note(), null)
  assert.equal(await chipText(), '48 chunks', 'a failing status endpoint breaks nothing')

  /* ---- 7. the chat searches in the default mode ------------------------- */
  await load([status(READY, 48)])
  await page.$eval('textarea[aria-label="Message Camelid"]:not([disabled])', (textarea) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, 'How long do big customers have to get their money back?')
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 30000 })
  await page.click('button[aria-label="Send message"]')
  await page.waitForSelector('main[data-view="chat"] button.citation-pill', { timeout: 30000 })
  await idle()
  assert.equal(searchRequests.length, 1)
  assert.equal(searchRequests[0].mode, undefined, 'the chat leaves the mode to the server')
  assert.deepEqual(searchRequests[0].doc_ids, [DOC.doc_id])

  /* ---- 8. the citation viewer says how each passage was found ----------- */
  const expected = ['Found by meaning', 'Found by keyword and by meaning', 'Found by keyword', 'Included because nothing else matched']
  for (const [index, text] of expected.entries()) {
    await page.click(`main[data-view="chat"] button.citation-pill[title="View source citation [${index + 1}]"]`)
    await page.waitForFunction(() => document.querySelector('.citation-modal .citation-modal__badge')?.textContent.includes('Verified'), { timeout: 10000 })
    assert.equal(await page.$eval('.citation-modal__found', (node) => node.textContent), text, `citation ${index + 1}`)
    await page.click('.citation-modal__footer button')
    await page.waitForFunction(() => !document.querySelector('.citation-modal'), { timeout: 10000 })
  }

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')

  console.log('semantic index browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
