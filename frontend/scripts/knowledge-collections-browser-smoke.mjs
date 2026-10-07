#!/usr/bin/env node
/* Browser-level acceptance for knowledge collections.
 *
 * Requires `npm run build` first. Separate loopback UI and authenticated API
 * fixtures verify the saved backend setting; every other origin is aborted.
 *
 *   - the knowledge library creates, renames, fills, empties and deletes a
 *     collection, and reports a clashing name
 *   - a collection searched in a chat shows in the composer, is sent as
 *     collection_ids, and is named on the sent message with its passages
 *   - a collection whose members stopped indexing says so, with the error, and
 *     is not polled; one waiting for the indexer says that instead
 *   - the chat stops searching it from the chip, and starts again from the
 *     conversation context
 *   - a project's collection reaches its new chats and can be turned off per chat
 *   - a deleted collection shows as unavailable and is left out of the search
 *   - uploading into a collection ingests with that collection
 *   - the LAN chat surface offers and searches no collections, and never asks for them
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
let collections = []
let nextId = 1
let lanOnly = false
// Overrides the fully indexed default: every chunk unindexed, with these semantic fields.
let unindexed = null
let statusCalls = 0
const collectionsCalls = []
const searchRequests = []
const ingestRequests = []
const chatRequests = []
const pageErrors = []
const externalRequests = []
const wrongBackendRequests = []
let uiOrigin = ''
const API_KEY = 'collections-fixture-key'

const bind = (n) => ({ chunk_sha256: String(n).repeat(64).slice(0, 64), doc_sha256: 'd'.repeat(64) })
const RESULTS = [
  { doc_id: 'doc-leave', filename: 'leave-policy.md', chunk_index: 1, retrieval: 'hybrid', excerpt: 'Book leave two weeks ahead.', byte_start: 40, byte_end: 67, ...bind(1) },
  { doc_id: 'doc-expenses', filename: 'expenses.md', chunk_index: 0, retrieval: 'semantic', excerpt: 'Claims need a receipt.', byte_start: 0, byte_end: 22, ...bind(2) },
  { doc_id: 'doc-leave', filename: 'leave-policy.md', chunk_index: 2, retrieval: 'keyword', excerpt: 'Unused leave carries over.', byte_start: 70, byte_end: 96, ...bind(3) },
].map((result) => ({ score: 0.03, ...result }))
const ANSWER = 'Book two weeks ahead [1], keep receipts [2], and leave carries over [3].'

function sendJson(res, statusCode, body) {
  const payload = JSON.stringify(body)
  res.writeHead(statusCode, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) })
  res.end(payload)
}
const apiError = (res, statusCode, code, message) => sendJson(res, statusCode, { error: { code, message, type: 'invalid_request_error' } })
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

/* The server's rules, in memory: names trimmed, unique regardless of case,
   at most 80 characters; unknown collections and documents are 404s. */
function collectionsApi(req, res, path, body) {
  const nameOf = (raw) => String(raw ?? '').trim()
  const clash = (name, except) => collections.some((c) => c.id !== except && c.name.toLowerCase() === name.toLowerCase())
  const invalid = (name) => !name ? 'Give the collection a name.' : name.length > 80 ? 'Collection names are at most 80 characters.' : null
  if (path === '/api/collections' && req.method === 'GET') return sendJson(res, 200, collections)
  if (path === '/api/collections' && req.method === 'POST') {
    const name = nameOf(body?.name)
    if (invalid(name)) return apiError(res, 422, 'invalid_collection_name', invalid(name))
    if (clash(name)) return apiError(res, 409, 'collection_name_taken', `A collection named ${JSON.stringify(name)} already exists.`)
    const collection = { id: `col-${nextId++}`, name, created_at: 10, doc_ids: [] }
    collections.push(collection)
    return sendJson(res, 201, collection)
  }
  const match = path.match(/^\/api\/collections\/([^/]+)(\/documents(?:\/([^/]+))?)?$/)
  const collection = match && collections.find((c) => c.id === decodeURIComponent(match[1]))
  if (!collection) return apiError(res, 404, 'collection_not_found', 'No such collection.')
  if (!match[2] && req.method === 'PATCH') {
    const name = nameOf(body?.name)
    if (invalid(name)) return apiError(res, 422, 'invalid_collection_name', invalid(name))
    if (clash(name, collection.id)) return apiError(res, 409, 'collection_name_taken', `A collection named ${JSON.stringify(name)} already exists.`)
    collection.name = name
    return sendJson(res, 200, collection)
  }
  if (!match[2] && req.method === 'DELETE') {
    collections = collections.filter((c) => c !== collection)
    return res.writeHead(204).end()
  }
  if (match[2] && !match[3] && req.method === 'POST') {
    const missing = (body?.doc_ids || []).find((id) => !documents.some((doc) => doc.id === id))
    if (missing) return apiError(res, 404, 'document_not_found', `No document has the id ${JSON.stringify(missing)}.`)
    for (const id of body.doc_ids) if (!collection.doc_ids.includes(id)) collection.doc_ids.push(id)
    return sendJson(res, 200, collection)
  }
  if (match[3] && req.method === 'DELETE') {
    collection.doc_ids = collection.doc_ids.filter((id) => id !== decodeURIComponent(match[3]))
    return res.writeHead(204).end()
  }
  return apiError(res, 405, 'method_not_allowed', 'Not allowed.')
}

const server = createServer(async (req, res) => {
  try {
    res.setHeader('Access-Control-Allow-Origin', uiOrigin)
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PATCH,DELETE,OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type,X-API-Key,Authorization')
    if (req.method === 'OPTIONS') return res.writeHead(204).end()
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if ((path.startsWith('/api/') || path.startsWith('/v1/')) && req.headers['x-api-key'] !== API_KEY) {
      return apiError(res, 401, 'unauthorized', 'The fixture requires its API key.')
    }
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: lanOnly ? 'lan_chat_only' : 'full',
        version: 'knowledge-collections-browser-smoke', build: 'knowledge-collections-browser-smoke',
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
    if (path.startsWith('/api/collections')) {
      collectionsCalls.push(`${req.method} ${path}`)
      if (lanOnly) return apiError(res, 403, 'lan_chat_only', 'this Camelid listener serves authenticated Chat and local model switching only')
      return collectionsApi(req, res, path, ['POST', 'PATCH'].includes(req.method) ? await readJsonBody(req) : null)
    }
    if (path === '/api/documents' && req.method === 'GET') return sendJson(res, 200, documents)
    if (path === '/api/documents/index-status' && req.method === 'GET') {
      statusCalls += 1
      return sendJson(res, 200, {
        semantic: { encoder: 'nomic-embed-text-v1.5.Q8_0.gguf', available: true, indexing: false, ...unindexed },
        documents: documents.map((doc) => ({ id: doc.id, indexable_chunks: doc.chunk_count, indexed_chunks: unindexed ? 0 : doc.chunk_count, skipped_chunks: 0 })),
      })
    }
    if (path === '/api/documents/ingest' && req.method === 'POST') {
      const body = await readJsonBody(req)
      ingestRequests.push(body)
      const missing = (body.collection_ids || []).find((id) => !collections.some((c) => c.id === id))
      if (missing) return apiError(res, 404, 'collection_not_found', 'No such collection.')
      const doc = { id: `doc-upload-${ingestRequests.length}`, filename: body.filename, file_type: 'txt', byte_size: body.content.length, chunk_count: 1, created_at: 20 }
      documents.push(doc)
      for (const id of body.collection_ids || []) collections.find((c) => c.id === id).doc_ids.push(doc.id)
      return sendJson(res, 200, { doc_id: doc.id, filename: doc.filename, chunk_count: 1, byte_size: doc.byte_size })
    }
    if (path === '/api/documents/search' && req.method === 'POST') {
      const body = await readJsonBody(req)
      searchRequests.push(body)
      const unknown = (body.collection_ids || []).find((id) => !collections.some((c) => c.id === id))
      if (unknown) return apiError(res, 404, 'collection_not_found', 'No such collection.')
      const scope = new Set([...(body.doc_ids || []), ...collections.filter((c) => (body.collection_ids || []).includes(c.id)).flatMap((c) => c.doc_ids)])
      return sendJson(res, 200, { results: RESULTS.filter((result) => scope.has(result.doc_id)), retrieval: { mode: 'hybrid', semantic: { available: true, indexable_chunks: 6, indexed_chunks: 6, skipped_chunks: 0 } } })
    }
    if (path === '/api/documents/citation/resolve' && req.method === 'POST') {
      const body = await readJsonBody(req)
      const hit = RESULTS.find((result) => result.doc_id === body?.doc_id && result.chunk_index === body?.chunk_index)
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
const apiOrigin = `http://127.0.0.1:${server.address().port}`
const uiServer = createServer((req, res) => {
  const path = new URL(req.url, 'http://127.0.0.1').pathname
  if (path.startsWith('/api/documents') || path.startsWith('/api/collections')) {
    wrongBackendRequests.push(path)
    return apiError(res, 404, 'wrong_backend', 'The library belongs to the configured API server.')
  }
  const filePath = resolve(distDir, `.${path}`)
  return sendFile(res, path !== '/' && isFile(filePath) ? filePath : resolve(distDir, 'index.html'))
})
await new Promise((done) => uiServer.listen(0, '127.0.0.1', done))
const origin = uiOrigin = `http://127.0.0.1:${uiServer.address().port}`

const browser = await launchBrowser({ purpose: 'the knowledge collections browser smoke', headless: 'new' })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
page.on('pageerror', (error) => pageErrors.push(String(error)))
await page.setRequestInterception(true)
page.on('request', (request) => {
  const url = request.url()
  if (url.startsWith('data:') || url.startsWith('blob:')) return request.continue()
  try { if ([origin, apiOrigin].includes(new URL(url).origin)) return request.continue() } catch { /* abort below */ }
  externalRequests.push(url)
  return request.abort()
})
await page.evaluateOnNewDocument((apiBase, key) => {
  if (window.sessionStorage.getItem('camelid.collectionsSmokeInitialized')) return
  window.localStorage.clear()
  window.localStorage.setItem('camelid.apiBase', apiBase)
  window.localStorage.setItem('camelid.apiKey', key)
  window.sessionStorage.setItem('camelid.collectionsSmokeInitialized', 'true')
}, apiOrigin, API_KEY)

const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const composerReady = 'textarea[aria-label="Message Camelid"]:not([disabled])'
const library = '.knowledge-modal'

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
async function clickText(selector, text) {
  const clicked = await page.$$eval(selector, (nodes, wanted) => {
    const node = nodes.find((item) => item.textContent.trim() === wanted)
    if (node) node.click()
    return Boolean(node)
  }, text)
  assert.ok(clicked, `no ${selector} reads ${JSON.stringify(text)}`)
}
const waitText = (selector, text, timeout = 10000) => page.waitForFunction((s, t) => [...document.querySelectorAll(s)].some((node) => node.textContent.includes(t)), { timeout }, selector, text)
const texts = (selector) => page.$$eval(selector, (nodes) => nodes.map((node) => node.textContent.trim()))
const collectionChips = () => texts('.cxcomposer__doc-pill--collection .cxcomposer__doc-open')

async function openLibrary() {
  await page.click('button[aria-label="Attach"]')
  await page.waitForSelector('button[aria-label="Knowledge collections"]', { timeout: 5000 })
  await page.click('button[aria-label="Knowledge collections"]')
  await page.waitForSelector(library, { timeout: 5000 })
}
async function closeLibrary() {
  await page.click(`${library} .cx-modal__footer button`)
  await page.waitForFunction((s) => !document.querySelector(s), { timeout: 5000 }, library)
}
async function send(text) {
  await setValue(composerReady, text)
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 30000 })
  const before = chatRequests.length
  await page.click('button[aria-label="Send message"]')
  await page.waitForFunction((n) => document.querySelectorAll('.cxturn--assistant').length > 0 && !document.querySelector('.cxcomposer__stop'), { timeout: 30000 }, before)
  const deadline = Date.now() + 10000
  while (chatRequests.length === before && Date.now() < deadline) await sleep(50)
  assert.equal(chatRequests.length, before + 1, 'the message reached the model')
  return chatRequests.at(-1)
}
const lastUserText = (request) => {
  const message = [...request.messages].reverse().find((item) => item.role === 'user')
  return typeof message.content === 'string' ? message.content : message.content.map((part) => part.text || '').join('')
}

try {
  /* ---- 1. the library: create, clash, fill, empty, rename ---------------- */
  await load()
  await openLibrary()
  await waitText(library, 'No collections yet.')
  await setValue('input[aria-label="New collection name"]', '  HR policies ')
  await page.click('button[aria-label="Create collection"]')
  await waitText('.knowledge-collection', 'HR policies')
  assert.equal(collections[0].name, 'HR policies', 'the name was sent trimmed by the server rules')
  await setValue('input[aria-label="New collection name"]', 'hr POLICIES')
  await page.click('button[aria-label="Create collection"]')
  await waitText(`${library} [role="alert"]`, 'A collection named "hr POLICIES" already exists.')
  assert.equal(collections.length, 1, 'a clashing name creates nothing')

  await page.click(`${library} .knowledge-add summary`).catch(() => {})
  await page.waitForSelector('.knowledge-pick input[type="checkbox"]', { visible: true, timeout: 5000 })
  const pickLabels = await texts('.knowledge-pick label')
  assert.deepEqual(pickLabels, ['expenses.md2 chunks', 'leave-policy.md3 chunks', 'roadmap.md1 chunk'], 'library documents are offered by name')
  for (const name of ['leave-policy.md', 'expenses.md', 'roadmap.md']) {
    await page.$$eval('.knowledge-pick label', (labels, wanted) => labels.find((label) => label.textContent.startsWith(wanted)).querySelector('input').click(), name)
  }
  await clickText(`${library} button`, 'Add 3 documents')
  await page.waitForFunction(() => document.querySelectorAll('.knowledge-docs li').length === 3, { timeout: 5000 })
  assert.deepEqual(collections[0].doc_ids, ['doc-leave', 'doc-expenses', 'doc-roadmap'])
  await page.click('button[aria-label="Remove roadmap.md from HR policies"]')
  await page.waitForFunction(() => document.querySelectorAll('.knowledge-docs li').length === 2, { timeout: 5000 })
  assert.deepEqual(collections[0].doc_ids, ['doc-leave', 'doc-expenses'], 'removing a member keeps the others')
  assert.ok(documents.some((doc) => doc.id === 'doc-roadmap'), 'removing a member keeps the document in the library')
  await page.click('button[aria-label="Rename HR policies"]')
  await setValue('input[aria-label="Collection name"]', 'HR')
  await clickText(`${library} button`, 'Save name')
  await waitText('.knowledge-detail h3', 'HR')
  assert.equal(collections[0].name, 'HR')

  /* ---- 2. search it in this chat ---------------------------------------- */
  const hrId = collections[0].id
  await page.$eval('.knowledge-search-toggle input', (input) => input.click())
  await waitText('.knowledge-collection', 'In this chat')
  await closeLibrary()
  assert.deepEqual(await collectionChips(), ['HR2 docs'], 'the composer shows the collection it will search')
  const request = await send('How early do I book leave, and what do expense claims need?')
  assert.equal(searchRequests.length, 1, 'one document search per message')
  assert.deepEqual(searchRequests[0].collection_ids, [hrId], 'the chat sends the collection, not its members')
  assert.equal(searchRequests[0].doc_ids, undefined, 'nothing was attached one by one')
  assert.equal(searchRequests[0].mode, undefined, 'the chat leaves the mode to the server')
  assert.match(lastUserText(request), /Book leave two weeks ahead\.[\s\S]*Claims need a receipt\./, 'the passages reach the model')
  assert.deepEqual(await texts('.cxturn__user-doc--collection'), ['HR3 passages used'], 'the sent message names the collection and what it supplied')
  await page.click('main[data-view="chat"] button.citation-pill[title="View source citation [2]"]')
  await page.waitForFunction(() => document.querySelector('.citation-modal .citation-modal__badge')?.textContent.includes('Verified'), { timeout: 10000 })
  await page.click('.citation-modal__footer button')

  /* ---- 3. stopped or waiting indexing shows on the collection ----------- */
  const failure = 'chunk embedding failed: the encoder ran out of memory'
  unindexed = { error: failure }
  await load()
  await waitText('.cxcomposer__doc-pill--collection .cxcomposer__doc-chunks--stopped', 'indexing stopped 0/5')
  assert.equal(await page.$eval('.cxcomposer__semantic-note--stopped .cxcomposer__semantic-error', (node) => node.textContent), failure, 'the reported error is shown')
  const callsWhenStopped = statusCalls
  await sleep(4500)
  assert.equal(statusCalls, callsWhenStopped, 'a stopped indexer is not polled')
  unindexed = {}
  await load()
  await waitText('.cxcomposer__doc-pill--collection .cxcomposer__doc-chunks--indexing', 'waiting to index 0/5')
  assert.equal(await page.$('.cxcomposer__semantic-note--stopped'), null, 'waiting is not a failure')
  unindexed = null
  await waitText('.cxcomposer__doc-pill--collection .cxcomposer__doc-open', 'HR2 docs')

  /* ---- 4. stop from the chip, start again from the context --------------- */
  await page.click('button[aria-label="Stop searching HR"]')
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__doc-pill--collection'), { timeout: 5000 })
  await load()
  assert.deepEqual(await collectionChips(), [], 'the choice is saved with the conversation')
  await page.click('button[aria-label="Edit conversation context"]')
  await page.waitForSelector('.context-collections', { timeout: 5000 })
  assert.deepEqual(await texts('.context-collections label'), ['HR2 docs'])
  await page.$eval('.context-collections label input', (input) => input.click())
  await clickText('.context-modal .cx-modal__footer button', 'Save context')
  await page.waitForSelector('.cxcomposer__doc-pill--collection', { timeout: 5000 })
  assert.deepEqual(await collectionChips(), ['HR2 docs'])

  /* ---- 5. a project's collections reach its chats ------------------------ */
  await page.evaluate(() => [...document.querySelectorAll('nav[aria-label="Primary"] button')].find((button) => button.textContent.trim() === 'Projects').click())
  await page.waitForSelector('.projects-view', { timeout: 10000 })
  await clickText('.projects-view button', 'New project')
  await page.waitForSelector('input[aria-label="Project name"]', { timeout: 5000 })
  await setValue('input[aria-label="Project name"]', 'People team')
  await page.waitForSelector('.context-collections label input', { timeout: 5000 })
  await page.$eval('.context-collections label input', (input) => input.click())
  await clickText('.context-modal .cx-modal__footer button', 'Save project')
  await waitText('.project-card', '1 collection')
  await page.click('button[aria-label="New chat in People team"]')
  await page.waitForSelector(composerReady, { timeout: 10000 })
  await page.waitForSelector('.cxcomposer__doc-pill--collection', { timeout: 5000 })
  assert.deepEqual(await collectionChips(), ['HR2 docs · project'], 'a new project chat searches the project collection')
  await send('What do expense claims need?')
  assert.deepEqual(searchRequests.at(-1).collection_ids, [hrId])
  await page.click('button[aria-label="Stop searching HR"]')
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__doc-pill--collection'), { timeout: 5000 })
  await page.click('button[aria-label="Edit conversation context"]')
  await page.waitForSelector('.context-collections .context-inheritance', { timeout: 5000 })
  assert.equal(await page.$eval('.context-collections .context-inheritance input', (input) => input.checked), false, 'this chat turned the project collection off')
  await page.$eval('.context-collections .context-inheritance input', (input) => input.click())
  await clickText('.context-modal .cx-modal__footer button', 'Save context')
  await page.waitForSelector('.cxcomposer__doc-pill--collection', { timeout: 5000 })
  assert.deepEqual(await collectionChips(), ['HR2 docs · project'], 'the dialog turns the project collection back on')
  await page.click('button[aria-label="Edit conversation context"]')
  await page.waitForSelector('.context-collections .context-inheritance input', { timeout: 5000 })
  await page.$eval('.context-collections .context-inheritance input', (input) => input.click())
  await clickText('.context-modal .cx-modal__footer button', 'Save context')
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__doc-pill--collection'), { timeout: 5000 })

  /* ---- 6. a deleted collection is unavailable, and not searched ---------- */
  await page.evaluate(() => [...document.querySelectorAll('nav[aria-label="Primary"] button')].find((button) => button.textContent.trim() === 'Chat').click())
  await page.waitForSelector(composerReady, { timeout: 10000 })
  await page.evaluate(() => document.querySelector('button[aria-label="New chat"]')?.click())
  await openLibrary()
  await page.$eval('.knowledge-search-toggle input', (input) => input.click())
  await page.click('button[aria-label="Delete HR"]')
  await clickText(`${library} .knowledge-confirm button`, 'Delete collection')
  await waitText(library, 'No collections yet.')
  await closeLibrary()
  assert.deepEqual(await collectionChips(), ['Collection unavailable'], 'a deleted collection stays visible as unavailable')
  const searchesBefore = searchRequests.length
  const blind = await send('Anything about leave?')
  assert.equal(searchRequests.length, searchesBefore, 'a deleted collection is not searched')
  assert.equal(lastUserText(blind), 'Anything about leave?', 'no document context is invented')
  await page.click('button[aria-label="Stop searching Collection unavailable"]')
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__doc-pill--collection'), { timeout: 5000 })

  /* ---- 7. upload straight into a collection ------------------------------ */
  await openLibrary()
  await setValue('input[aria-label="New collection name"]', 'Finance')
  await page.click('button[aria-label="Create collection"]')
  await waitText('.knowledge-detail h3', 'Finance')
  const dir = mkdtempSync(join(tmpdir(), 'camelid-collections-'))
  const file = join(dir, 'budget.txt')
  writeFileSync(file, 'The travel budget is reviewed every quarter.')
  const upload = await page.$('input[aria-label="Upload files into Finance"]')
  await upload.uploadFile(file)
  await page.waitForFunction(() => document.querySelectorAll('.knowledge-docs li').length === 1, { timeout: 10000 })
  assert.deepEqual(ingestRequests.at(-1).collection_ids, [collections.at(-1).id], 'the upload names its collection')
  assert.equal(ingestRequests.at(-1).filename, 'budget.txt')
  await page.$eval('.knowledge-search-toggle input', (input) => input.click())
  await closeLibrary()
  assert.deepEqual(await collectionChips(), ['Finance1 doc'])

  /* ---- 8. the LAN chat surface: no collections, no requests -------------- */
  lanOnly = true
  const callsBefore = collectionsCalls.length
  await load()
  await sleep(1000)
  await page.click('button[aria-label="Attach"]')
  await page.waitForSelector('button[aria-label="Attach documents for RAG"]', { timeout: 5000 })
  assert.equal(await page.$('button[aria-label="Knowledge collections"]'), null, 'no collections on the LAN chat surface')
  await page.keyboard.press('Escape')
  assert.deepEqual(await collectionChips(), [], 'a chat\'s collections are not offered on the LAN chat surface')
  const searchesBeforeLan = searchRequests.length
  const lanRequest = await send('What is reviewed every quarter?')
  assert.equal(searchRequests.length, searchesBeforeLan, 'the LAN chat surface searches no collection')
  assert.equal(lastUserText(lanRequest), 'What is reviewed every quarter?')
  assert.equal(collectionsCalls.length, callsBefore, 'the LAN chat surface never asks for collections')

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(wrongBackendRequests, [], 'every library request reaches the configured backend')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything outside its two fixtures')
  console.log('knowledge collections browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => uiServer.close(done))
  await new Promise((done) => server.close(done))
}
