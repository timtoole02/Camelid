#!/usr/bin/env node
/* Browser-level acceptance for whole-library search in chat.
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app and a documents API that answers the way the server does;
 * every cross-origin request is aborted.
 *
 *   - Attach -> Whole library turns it on for this chat, and the composer says so
 *   - stopped indexing shows on the library chip with the error, and is not
 *     polled; waiting for the indexer says that instead
 *   - a message searches with `library: true` and nothing else pinned, and the
 *     sent message says how many passages the library supplied
 *   - a message nothing in the library is close to is sent unchanged
 *   - with an attached document too, both are searched and counted apart
 *   - the choice is saved with the chat, and a new chat starts without it
 *   - without the encoder the chat says so, sends without the library, and
 *     still searches what is attached
 *   - the LAN chat surface offers no library search and never searches
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { launchBrowser } from './lib/launch-browser.mjs'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const distDir = resolve(scriptDir, '../dist')
const ledgerPath = resolve(scriptDir, '../../ledger/camelid-ledger.json')
const MODEL_FILENAME = 'Qwen3-0.6B-Q8_0.gguf'

const MIME = {
  '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map((row) => row.contract) }

const documents = [
  { id: 'doc-leave', filename: 'leave-policy.md', file_type: 'md', byte_size: 900, chunk_count: 3, created_at: 3 },
  { id: 'doc-expenses', filename: 'expenses.md', file_type: 'md', byte_size: 700, chunk_count: 2, created_at: 2 },
  { id: 'doc-roadmap', filename: 'roadmap.md', file_type: 'md', byte_size: 500, chunk_count: 1, created_at: 1 },
]
let encoderAvailable = true
let lanOnly = false
// Overrides the fully indexed default: every chunk unindexed, with these semantic fields.
let unindexed = null
let statusCalls = 0
const searchRequests = []
const chatRequests = []
const pageErrors = []
const externalRequests = []

const bind = (n) => ({ chunk_sha256: String(n).repeat(64).slice(0, 64), doc_sha256: 'd'.repeat(64) })
const PASSAGES = [
  { doc_id: 'doc-leave', filename: 'leave-policy.md', chunk_index: 1, retrieval: 'hybrid', similarity: 0.81, excerpt: 'Book leave two weeks ahead.', byte_start: 40, byte_end: 67, ...bind(1) },
  { doc_id: 'doc-expenses', filename: 'expenses.md', chunk_index: 0, retrieval: 'semantic', similarity: 0.74, excerpt: 'Claims need a receipt.', byte_start: 0, byte_end: 22, ...bind(2) },
  { doc_id: 'doc-roadmap', filename: 'roadmap.md', chunk_index: 0, retrieval: 'semantic', similarity: 0.72, excerpt: 'The mobile release ships in March.', byte_start: 0, byte_end: 34, ...bind(3) },
].map((passage) => ({ score: 0.03, ...passage }))
const ANSWER = 'Book two weeks ahead [1] and keep receipts [2].'
const OFF_TOPIC = 'Tell me a joke about penguins.'

function sendJson(res, statusCode, body) {
  const payload = JSON.stringify(body)
  res.writeHead(statusCode, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) })
  res.end(payload)
}
function sendFile(res, path) {
  res.writeHead(200, { 'Content-Type': MIME[extname(path)] || 'application/octet-stream' })
  res.end(readFileSync(path))
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

const semanticStatus = () => (encoderAvailable
  ? { encoder: 'nomic-embed-text-v1.5.Q8_0.gguf', available: true, indexing: false }
  : { available: false, indexing: false, reason: 'encoder_not_installed', message: 'The embedding model nomic-embed-text-v1.5.Q8_0.gguf is not in the models directory.' })

/* The server's rules: `library` searches everything and keeps passages from
   outside doc_ids only when close enough; it needs the encoder. */
function search(body) {
  if (body.library && !encoderAvailable) {
    return [409, { error: { code: 'encoder_not_installed', message: 'The embedding model nomic-embed-text-v1.5.Q8_0.gguf is not in the models directory.', param: 'library' } }]
  }
  const pinned = new Set(body.doc_ids || [])
  const offTopic = body.query === OFF_TOPIC
  const results = PASSAGES.filter((passage) => pinned.has(passage.doc_id) || (body.library && !offTopic))
    .slice(0, body.top_k || 4)
    .map((passage) => (body.library ? passage : { ...passage, similarity: undefined }))
  return [200, { results, retrieval: { mode: 'hybrid', semantic: { available: encoderAvailable }, ...(body.library ? { relevance_floor: 0.69 } : {}) } }]
}

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: lanOnly ? 'lan_chat_only' : 'full',
        version: 'library-search-browser-smoke', build: 'library-search-browser-smoke',
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
    if (lanOnly && path.startsWith('/api/documents')) {
      return sendJson(res, 403, { error: { code: 'lan_chat_only', message: 'this Camelid listener serves authenticated Chat and local model switching only' } })
    }
    if (path === '/api/collections' && req.method === 'GET') return sendJson(res, 200, [])
    if (path === '/api/documents' && req.method === 'GET') return sendJson(res, 200, documents)
    if (path === '/api/documents/index-status' && req.method === 'GET') {
      statusCalls += 1
      return sendJson(res, 200, {
        semantic: { ...semanticStatus(), ...unindexed },
        documents: documents.map((doc) => ({ id: doc.id, indexable_chunks: doc.chunk_count, indexed_chunks: encoderAvailable && !unindexed ? doc.chunk_count : 0, skipped_chunks: 0 })),
      })
    }
    if (path === '/api/documents/ingest' && req.method === 'POST') {
      const body = await readJsonBody(req)
      const doc = documents.find((item) => item.filename === body.filename)
      return sendJson(res, 200, { doc_id: doc.id, filename: doc.filename, chunk_count: doc.chunk_count, byte_size: doc.byte_size })
    }
    if (path === '/api/documents/search' && req.method === 'POST') {
      const body = await readJsonBody(req)
      searchRequests.push(body)
      const [status, answer] = search(body)
      return sendJson(res, status, answer)
    }
    if (path === '/api/documents/citation/resolve' && req.method === 'POST') {
      const body = await readJsonBody(req)
      const hit = PASSAGES.find((passage) => passage.doc_id === body?.doc_id && passage.chunk_index === body?.chunk_index)
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

const browser = await launchBrowser({ purpose: 'the library search browser smoke', headless: 'new' })
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
  if (window.sessionStorage.getItem('camelid.librarySmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.librarySmokeInitialized', 'true')
})

const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const composerReady = 'textarea[aria-label="Message Camelid"]:not([disabled])'
const libraryChip = '.cxcomposer__doc-pill--library'

async function load() {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector(composerReady, { timeout: 30000 })
}
async function setValue(selector, value) {
  await page.$eval(selector, (node, next) => {
    const prototype = node.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(node, next)
    node.dispatchEvent(new Event('input', { bubbles: true }))
  }, value)
}
const texts = (selector) => page.$$eval(selector, (nodes) => nodes.map((node) => node.textContent.trim()))
async function openAttach() {
  await page.click('button[aria-label="Attach"]')
  await page.waitForSelector('button[aria-label="Attach documents for RAG"]', { timeout: 5000 })
}
async function toggleLibrary() {
  await openAttach()
  await page.click('button[aria-label="Search the whole library"]')
}
async function send(text) {
  await setValue(composerReady, text)
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 30000 })
  const before = chatRequests.length
  const turns = await page.$$eval('.cxturn--user', (nodes) => nodes.length)
  await page.click('button[aria-label="Send message"]')
  await page.waitForFunction((n) => document.querySelectorAll('.cxturn--user').length > n && !document.querySelector('.cxcomposer__stop'), { timeout: 30000 }, turns)
  const deadline = Date.now() + 10000
  while (chatRequests.length === before && Date.now() < deadline) await sleep(50)
  assert.equal(chatRequests.length, before + 1, 'the message reached the model')
  return chatRequests.at(-1)
}
const lastUserText = (request) => {
  const message = [...request.messages].reverse().find((item) => item.role === 'user')
  return typeof message.content === 'string' ? message.content : message.content.map((part) => part.text || '').join('')
}
const lastTurnChips = (kind) => page.$$eval('.cxturn--user', (turns, cls) => [...turns.at(-1).querySelectorAll(cls)].map((node) => node.textContent.trim()), kind)

try {
  /* ---- 1. turn it on ---------------------------------------------------- */
  await load()
  assert.equal(await page.$(libraryChip), null, 'a new chat does not search the whole library')
  await openAttach()
  assert.equal(await page.$eval('button[aria-label="Search the whole library"]', (button) => button.getAttribute('aria-pressed')), 'false')
  await page.click('button[aria-label="Search the whole library"]')
  await page.waitForSelector(libraryChip, { timeout: 5000 })
  await page.waitForFunction((s) => /docs$/.test(document.querySelector(`${s} .cxcomposer__doc-chunks`)?.textContent || ''), { timeout: 5000 }, libraryChip)
  assert.deepEqual(await texts(`${libraryChip} .cxcomposer__doc-open`), ['Whole library3 docs'], 'the composer says the whole library will be searched')

  /* ---- 2. stopped or waiting indexing on the library chip -------------- */
  const failure = 'chunk embedding failed: the encoder ran out of memory'
  unindexed = { error: failure }
  await load()
  await page.waitForFunction((s) => document.querySelector(`${s} .cxcomposer__doc-chunks--stopped`)?.textContent === 'indexing stopped 0/6', { timeout: 10000 }, libraryChip)
  assert.match(await page.$eval(`${libraryChip} .cxcomposer__doc-chunks--stopped`, (node) => node.title), /not yet indexed are not found/)
  assert.equal(await page.$eval('.cxcomposer__semantic-note--stopped .cxcomposer__semantic-error', (node) => node.textContent), failure, 'the reported error is shown')
  assert.match(await page.$eval('.cxcomposer__semantic-note--stopped', (node) => node.textContent), /not at all elsewhere in the library/)
  const callsWhenStopped = statusCalls
  await sleep(4500)
  assert.equal(statusCalls, callsWhenStopped, 'a stopped indexer is not polled')
  unindexed = {}
  await load()
  await page.waitForFunction((s) => document.querySelector(`${s} .cxcomposer__doc-chunks--indexing`)?.textContent === 'waiting to index 0/6', { timeout: 10000 }, libraryChip)
  assert.equal(await page.$('.cxcomposer__semantic-note--stopped'), null, 'waiting is not a failure')
  unindexed = null
  await page.waitForFunction((s) => document.querySelector(`${s} .cxcomposer__doc-chunks`)?.textContent === '3 docs', { timeout: 10000 }, libraryChip)

  /* ---- 3. a question the library answers -------------------------------- */
  const request = await send('How early do I book leave, and what do claims need?')
  assert.equal(searchRequests.length, 1, 'one document search per message')
  assert.deepEqual(searchRequests[0], { query: 'How early do I book leave, and what do claims need?', top_k: 4, library: true }, 'only the library flag, nothing pinned')
  assert.match(lastUserText(request), /Book leave two weeks ahead\.[\s\S]*Claims need a receipt\./, 'the passages reach the model')
  assert.deepEqual(await lastTurnChips('.cxturn__user-doc--library'), ['Whole library3 passages used'])
  await page.click('main[data-view="chat"] button.citation-pill[title="View source citation [2]"]')
  await page.waitForFunction(() => document.querySelector('.citation-modal .citation-modal__badge')?.textContent.includes('Verified'), { timeout: 10000 })
  await page.click('.citation-modal__footer button')

  /* ---- 4. nothing close enough ------------------------------------------ */
  const quiet = await send(OFF_TOPIC)
  assert.equal(searchRequests.at(-1).library, true)
  assert.equal(lastUserText(quiet), OFF_TOPIC, 'no document context is invented when nothing clears the floor')
  assert.deepEqual(await lastTurnChips('.cxturn__user-doc--library'), ['Whole libraryno passages used'])

  /* ---- 5. an attached document as well ---------------------------------- */
  const dir = mkdtempSync(join(tmpdir(), 'camelid-library-'))
  const file = join(dir, 'leave-policy.md')
  writeFileSync(file, 'Book leave two weeks ahead.')
  await (await page.$('input.sr-only[type="file"][accept*=".md"]')).uploadFile(file)
  await page.waitForSelector('.cxcomposer__doc-open[title="Open leave-policy.md"]', { timeout: 10000 })
  await send('What should I know about leave and claims?')
  assert.deepEqual(searchRequests.at(-1).doc_ids, ['doc-leave'])
  assert.equal(searchRequests.at(-1).library, true)
  assert.deepEqual(await lastTurnChips('.cxturn__user-doc--library'), ['Whole library2 passages used'], 'the library counts only what it added')
  assert.deepEqual(await lastTurnChips('button.cxturn__user-doc'), ['leave-policy.md1 passage used'])
  await page.click('button[aria-label="Remove leave-policy.md"]')

  /* ---- 6. saved with the chat ------------------------------------------- */
  await load()
  await page.waitForSelector(libraryChip, { timeout: 5000 })
  await page.click('button[aria-label="Stop searching the whole library"]')
  await page.waitForFunction((s) => !document.querySelector(s), { timeout: 5000 }, libraryChip)
  await load()
  await sleep(500)
  assert.equal(await page.$(libraryChip), null, 'turning it off is saved too')
  const searchesBefore = searchRequests.length
  const plain = await send('Anything new?')
  assert.equal(searchRequests.length, searchesBefore, 'with nothing attached and the library off, nothing is searched')
  assert.equal(lastUserText(plain), 'Anything new?')
  await toggleLibrary()
  await page.waitForSelector(libraryChip, { timeout: 5000 })
  const newChat = await page.evaluate(() => {
    const button = document.querySelector('button.rail__new-chat, button[aria-label="New chat"]')
    button?.click()
    return Boolean(button)
  })
  assert.ok(newChat, 'the sidebar offers New chat')
  await page.waitForFunction((s) => !document.querySelector(s), { timeout: 5000 }, libraryChip)

  /* ---- 7. without the encoder ------------------------------------------- */
  encoderAvailable = false
  await load()
  await toggleLibrary()
  await page.waitForSelector('.cxcomposer__semantic-note--library', { timeout: 10000 })
  assert.match(await page.$eval('.cxcomposer__semantic-note--library', (node) => node.textContent), /^Whole-library search needs search by meaning\. The embedding model/)
  await (await page.$('input.sr-only[type="file"][accept*=".md"]')).uploadFile(file)
  await page.waitForSelector('.cxcomposer__doc-open[title="Open leave-policy.md"]', { timeout: 10000 })
  const searchesBeforeNoEncoder = searchRequests.length
  const degraded = await send('How early do I book leave?')
  assert.equal(searchRequests.length, searchesBeforeNoEncoder + 2, 'tried the library, then searched what is attached')
  assert.equal(searchRequests.at(-2).library, true)
  assert.equal(searchRequests.at(-1).library, undefined)
  assert.deepEqual(searchRequests.at(-1).doc_ids, ['doc-leave'])
  assert.match(lastUserText(degraded), /Book leave two weeks ahead\./, 'the attached document still answers')
  await page.waitForFunction(() => [...document.querySelectorAll('p[role="alert"]')].some((node) => node.textContent.includes('This message was sent without searching the whole library.')), { timeout: 5000 })
  assert.deepEqual(await lastTurnChips('.cxturn__user-doc--library'), [], 'a library search that did not run is not claimed')
  await page.click('button[aria-label="Remove leave-policy.md"]')
  encoderAvailable = true

  /* ---- 8. the LAN chat surface ------------------------------------------ */
  lanOnly = true
  await load()
  await sleep(1000)
  assert.equal(await page.$(libraryChip), null, 'a saved library search is not offered on the LAN chat surface')
  await openAttach()
  assert.equal(await page.$('button[aria-label="Search the whole library"]'), null, 'no library search on the LAN chat surface')
  await page.keyboard.press('Escape')
  const searchesBeforeLan = searchRequests.length
  await send('What is the release month?')
  assert.equal(searchRequests.length, searchesBeforeLan, 'the LAN chat surface never searches the library')

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')
  console.log('library search browser smoke passed')
} finally {
  await browser.close()
  server.close()
}
