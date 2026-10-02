#!/usr/bin/env node
/* Browser-level acceptance for watched folders and dropped files.
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app and an in-memory library with the server's folder rules; every
 * cross-origin request is aborted.
 *
 *   - a collection watches a folder: the path is sent with the collection, the
 *     check's progress shows, and the collection's documents appear when it ends
 *   - the server's refusal of a path is shown as it gave it
 *   - Browse lists the server's folders, and choosing one fills the path
 *   - skipped files are listed with their reasons
 *   - Check now asks the server for a scan
 *   - a check the server's timer ran between two polls still refreshes the
 *     collection's documents
 *   - Stop watching asks first, names what leaves the library, and refreshes it
 *   - dropping files ingests the readable ones into the collection, by name,
 *     and says how many were left out
 *   - deleting a collection says its folders stop being watched
 *   - the LAN chat surface never asks for folders
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
const MIME = {
  '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map((row) => row.contract) }

/* A folder tree for Browse, and what the policies folder holds. */
const TREE = { '/': ['/srv'], '/srv': ['/srv/archive', '/srv/policies'], '/srv/archive': [], '/srv/policies': [] }
const FOLDER_FILES = ['refunds.md', 'incidents.md', 'team/notes.md']
const FOLDER_SKIPS = [{ path: 'broken.pdf', reason: 'extract_failed' }, { path: 'empty.txt', reason: 'no_text' }]

let documents = []
let collections = []
let folders = []
let nextId = 1
let lanOnly = false
const folderCalls = []
const ingestRequests = []
const pageErrors = []
const externalRequests = []

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
function isFile(path) { try { return statSync(path).isFile() } catch { return false } }

function addDocument(filename, collectionId) {
  const doc = { id: `doc-${nextId++}`, filename, file_type: filename.split('.').pop(), byte_size: 400, chunk_count: 2, created_at: nextId }
  documents.push(doc)
  collections.find((c) => c.id === collectionId)?.doc_ids.push(doc.id)
  return doc
}

/* A scan advances one step per listing: queued, then halfway, then done. */
function advance(folder) {
  if (folder.step === 0) {
    Object.assign(folder, { queued: false, scanning: true, progress: { done: 1, total: FOLDER_FILES.length + FOLDER_SKIPS.length } })
  } else if (folder.step === 1) {
    const fresh = !folder.doc_ids.length
    if (fresh) folder.doc_ids = FOLDER_FILES.map((name) => addDocument(name, folder.collection_id).id)
    Object.assign(folder, {
      scanning: false, progress: undefined, last_scan_at: Math.floor(Date.now() / 1000), documents: folder.doc_ids.length,
      skipped_count: FOLDER_SKIPS.length, skipped: FOLDER_SKIPS,
      last_changes: { added: fresh ? FOLDER_FILES.length : 0, updated: fresh ? 0 : 1, removed: 0, unchanged: fresh ? 0 : 2, skipped: FOLDER_SKIPS.length },
    })
  }
  folder.step += 1
}
const view = ({ step, doc_ids, ...folder }) => folder

/* A check the server's timer runs starts and ends between two polls. */
function backgroundScan(folder) {
  folder.doc_ids.push(addDocument('handbook.md', folder.collection_id).id)
  Object.assign(folder, {
    last_scan_at: Math.floor(Date.now() / 1000), documents: folder.doc_ids.length,
    last_changes: { added: 1, updated: 0, removed: 0, unchanged: folder.doc_ids.length - 1, skipped: FOLDER_SKIPS.length },
  })
}

function foldersApi(req, res, path, body) {
  if (path === '/api/folders' && req.method === 'GET') {
    folders.forEach(advance)
    return sendJson(res, 200, folders.map(view))
  }
  if (path === '/api/folders' && req.method === 'POST') {
    const requested = String(body?.path ?? '').trim()
    if (!requested.startsWith('/')) return apiError(res, 422, 'invalid_folder', "Give the folder's full path.")
    if (!(requested in TREE) || requested === '/') return apiError(res, 422, 'invalid_folder', 'That folder does not exist or cannot be opened.')
    if (!collections.some((c) => c.id === body?.collection_id)) return apiError(res, 404, 'collection_not_found', 'No such collection.')
    if (folders.some((f) => f.path === requested || f.path.startsWith(`${requested}/`) || requested.startsWith(`${f.path}/`))) {
      return apiError(res, 409, 'folder_overlaps', `${requested} is already watched, or is inside or around a watched folder.`)
    }
    const folder = {
      id: `folder-${nextId++}`, path: requested, collection_id: body.collection_id, created_at: 1, last_scan_at: null,
      last_error: null, last_changes: null, documents: 0, skipped_count: 0, skipped: [], scanning: false, queued: true, step: 0, doc_ids: [],
    }
    folders.push(folder)
    return sendJson(res, 201, view(folder))
  }
  const match = path.match(/^\/api\/folders\/([^/]+)(\/scan)?$/)
  const folder = match && folders.find((f) => f.id === decodeURIComponent(match[1]))
  if (!folder) return apiError(res, 404, 'folder_not_found', 'No such folder.')
  if (match[2] && req.method === 'POST') {
    Object.assign(folder, { queued: true, step: 0 })
    return sendJson(res, 202, view(folder))
  }
  if (!match[2] && req.method === 'DELETE') {
    folders = folders.filter((f) => f !== folder)
    documents = documents.filter((doc) => !folder.doc_ids.includes(doc.id))
    for (const collection of collections) collection.doc_ids = collection.doc_ids.filter((id) => !folder.doc_ids.includes(id))
    return res.writeHead(204).end()
  }
  return apiError(res, 405, 'method_not_allowed', 'Not allowed.')
}

function collectionsApi(req, res, path, body) {
  if (path === '/api/collections' && req.method === 'GET') return sendJson(res, 200, collections)
  if (path === '/api/collections' && req.method === 'POST') {
    const collection = { id: `col-${nextId++}`, name: String(body?.name ?? '').trim(), created_at: 10, doc_ids: [] }
    collections.push(collection)
    return sendJson(res, 201, collection)
  }
  const match = path.match(/^\/api\/collections\/([^/]+)$/)
  const collection = match && collections.find((c) => c.id === decodeURIComponent(match[1]))
  if (!collection) return apiError(res, 404, 'collection_not_found', 'No such collection.')
  if (req.method === 'DELETE') {
    collections = collections.filter((c) => c !== collection)
    folders = folders.filter((f) => f.collection_id !== collection.id)
    return res.writeHead(204).end()
  }
  return apiError(res, 405, 'method_not_allowed', 'Not allowed.')
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://127.0.0.1')
    const path = url.pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: lanOnly ? 'lan_chat_only' : 'full',
        version: 'watched-folders-browser-smoke', build: 'watched-folders-browser-smoke',
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
    if (path.startsWith('/api/folders')) {
      folderCalls.push(`${req.method} ${path}`)
      if (lanOnly) return apiError(res, 403, 'lan_chat_only', 'this Camelid listener serves authenticated Chat and local model switching only')
      return foldersApi(req, res, path, req.method === 'POST' ? await readJsonBody(req) : null)
    }
    if (path === '/api/agent/workspace/browse') {
      const requested = url.searchParams.get('path') || '/'
      if (!(requested in TREE)) return apiError(res, 400, 'workspace_browse_path_invalid', 'that folder is not accessible')
      const parent = requested === '/' ? null : requested.slice(0, requested.lastIndexOf('/')) || '/'
      return sendJson(res, 200, {
        path: requested, parent, has_roots: false, separator: '/', truncated: false,
        entries: TREE[requested].map((child) => ({ name: child.slice(child.lastIndexOf('/') + 1), path: child })),
      })
    }
    if (path.startsWith('/api/collections')) {
      if (lanOnly) return apiError(res, 403, 'lan_chat_only', 'this Camelid listener serves authenticated Chat and local model switching only')
      return collectionsApi(req, res, path, req.method === 'POST' ? await readJsonBody(req) : null)
    }
    if (path === '/api/documents' && req.method === 'GET') return sendJson(res, 200, documents)
    if (path === '/api/documents/index-status' && req.method === 'GET') {
      return sendJson(res, 200, {
        semantic: { encoder: 'nomic-embed-text-v1.5.Q8_0.gguf', available: true, indexing: false },
        documents: documents.map((doc) => ({ id: doc.id, indexable_chunks: doc.chunk_count, indexed_chunks: doc.chunk_count, skipped_chunks: 0 })),
      })
    }
    if (path === '/api/documents/ingest' && req.method === 'POST') {
      const body = await readJsonBody(req)
      ingestRequests.push(body)
      const doc = addDocument(body.filename, body.collection_ids?.[0])
      return sendJson(res, 200, { doc_id: doc.id, filename: doc.filename, chunk_count: doc.chunk_count, byte_size: doc.byte_size })
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

const browser = await launchBrowser({ purpose: 'the watched folders browser smoke', headless: 'new' })
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
  if (window.sessionStorage.getItem('camelid.foldersSmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.foldersSmokeInitialized', 'true')
})

const composerReady = 'textarea[aria-label="Message Camelid"]:not([disabled])'
const library = '.knowledge-modal'
const folderItem = `${library} .knowledge-folder`

async function load() {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector(composerReady, { timeout: 30000 })
}
async function setValue(selector, value) {
  await page.$eval(selector, (node, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, next)
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
const text = (selector) => page.$eval(selector, (node) => node.textContent.trim())
const texts = (selector) => page.$$eval(selector, (nodes) => nodes.map((node) => node.textContent.trim()))
const inputValue = (selector) => page.$eval(selector, (node) => node.value)

async function openLibrary() {
  await page.click('button[aria-label="Attach"]')
  await page.waitForSelector('button[aria-label="Knowledge collections"]', { timeout: 5000 })
  await page.click('button[aria-label="Knowledge collections"]')
  await page.waitForSelector(library, { timeout: 5000 })
}

try {
  /* ---- 1. a collection watches a folder --------------------------------- */
  await load()
  await openLibrary()
  await setValue('input[aria-label="New collection name"]', 'Policies')
  await page.click('button[aria-label="Create collection"]')
  await waitText('.knowledge-collection', 'Policies')
  const collectionId = collections[0].id
  await waitText(`${library} .knowledge-folders`, 'Watch a folder')
  await clickText(`${library} .knowledge-folder-add`, 'Watch a folder')
  await page.waitForSelector('#knowledge-folder-path')

  await setValue('#knowledge-folder-path', 'relative/policies')
  await clickText(`${library} .knowledge-folder-form button`, 'Watch into Policies')
  await waitText(`${library} [role="alert"]`, "Give the folder's full path.")
  assert.equal(folders.length, 0, 'a refused path watches nothing')

  /* ---- 2. Browse fills the path ------------------------------------------ */
  await setValue('#knowledge-folder-path', '/srv')
  await clickText(`${library} .knowledge-folder-form button`, 'Browse')
  await waitText(`${library} .knowledge-browser`, 'policies')
  assert.deepEqual(await texts(`${library} .knowledge-browser__entry`), ['archive', 'policies'], 'Browse lists the folders the server reports')
  await clickText(`${library} .knowledge-browser__entry`, 'policies')
  await page.waitForFunction(() => document.querySelector('#knowledge-folder-path')?.value === '/srv/policies', { timeout: 5000 })
  assert.equal(await text(`${library} .knowledge-browser__bar code`), '/srv/policies')

  /* ---- 3. watching: progress, then the collection's documents ------------ */
  await clickText(`${library} .knowledge-folder-form button`, 'Watch into Policies')
  await page.waitForSelector(folderItem, { timeout: 10000 })
  assert.deepEqual(folderCalls.filter((call) => call.startsWith('POST')), ['POST /api/folders', 'POST /api/folders'], 'one refused and one accepted request')
  assert.equal(folders[0].path, '/srv/policies')
  assert.equal(folders[0].collection_id, collectionId, 'the folder was sent with its collection')
  await waitText(`${folderItem} .knowledge-folder__status`, 'Checking 1 of 5 files')
  await waitText(`${folderItem} .knowledge-folder__status`, '3 documents. Checked just now. Last check: 3 added.')
  await waitText(`${library} .knowledge-docs`, 'team/notes.md')
  assert.deepEqual((await texts(`${library} .knowledge-docs .knowledge-doc__name`)).sort(), ['incidents.md', 'refunds.md', 'team/notes.md'], 'the collection lists the folder\'s documents once the check ends')

  /* ---- 4. skipped files and their reasons -------------------------------- */
  assert.equal(await text(`${folderItem} .knowledge-folder__skipped summary`), '2 files skipped')
  assert.deepEqual(await texts(`${folderItem} .knowledge-folder__skipped li`), ['broken.pdfcould not be parsed', 'empty.txtno readable text'])

  /* ---- 5. Check now ------------------------------------------------------- */
  const scansBefore = folderCalls.filter((call) => call.endsWith('/scan')).length
  await page.click('button[aria-label="Check /srv/policies now"]')
  await waitText(`${folderItem} .knowledge-folder__status`, 'Last check: 1 updated.')
  assert.equal(folderCalls.filter((call) => call.endsWith('/scan')).length, scansBefore + 1, 'Check now asked for one scan')

  /* ---- 6. an overlapping folder is refused as the server says ------------ */
  await clickText(`${library} .knowledge-folder-add`, 'Watch a folder')
  await setValue('#knowledge-folder-path', '/srv')
  await clickText(`${library} .knowledge-folder-form button`, 'Watch into Policies')
  await waitText(`${library} [role="alert"]`, '/srv is already watched, or is inside or around a watched folder.')
  await clickText(`${library} .knowledge-folder-form button`, 'Cancel')

  /* ---- 7. dropping files --------------------------------------------------- */
  await page.evaluate(() => {
    const target = document.querySelector('.knowledge-modal .knowledge-detail')
    const transfer = new DataTransfer()
    transfer.items.add(new File(['# Leave\n\nBook leave two weeks ahead.'], 'leave.md', { type: 'text/markdown' }))
    transfer.items.add(new File(['binary'], 'diagram.png', { type: 'image/png' }))
    target.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: transfer }))
    target.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: transfer }))
  })
  await waitText(`${library} [role="alert"]`, 'Skipped 1 file of a type the library does not read.')
  assert.equal(ingestRequests.length, 1, 'only the readable file was ingested')
  assert.equal(ingestRequests[0].filename, 'leave.md')
  assert.deepEqual(ingestRequests[0].collection_ids, [collectionId], 'the drop ingests into the open collection')
  await waitText(`${library} .knowledge-docs`, 'leave.md')

  /* ---- 8. a check the server's timer ran between two polls ---------------- */
  backgroundScan(folders[0])
  await waitText(`${folderItem} .knowledge-folder__status`, '4 documents. Checked just now. Last check: 1 added.', 15000)
  await waitText(`${library} .knowledge-docs`, 'handbook.md', 15000)

  /* ---- 9. Stop watching ---------------------------------------------------- */
  await page.click('button[aria-label="Stop watching /srv/policies"]')
  await waitText(`${library} .knowledge-confirm`, 'Stop watching /srv/policies? Its 4 documents leave the library. The files on disk are not touched.')
  await clickText(`${library} .knowledge-confirm button`, 'Stop watching')
  await page.waitForFunction((s) => !document.querySelector(s), { timeout: 5000 }, folderItem)
  assert.ok(folderCalls.some((call) => call.startsWith('DELETE /api/folders/folder-')), 'the watch was stopped on the server')
  await page.waitForFunction(() => [...document.querySelectorAll('.knowledge-modal .knowledge-docs .knowledge-doc__name')].map((n) => n.textContent.trim()).join(',') === 'leave.md', { timeout: 5000 })

  /* ---- 10. deleting the collection names its folders ---------------------- */
  await page.click('button[aria-label="Delete Policies"]')
  await waitText(`${library} .knowledge-confirm`, 'the folders it watches stop being watched')

  /* ---- 11. the LAN chat surface never asks for folders -------------------- */
  lanOnly = true
  const callsBefore = folderCalls.length
  await load()
  await new Promise((done) => setTimeout(done, 1500))
  assert.equal(folderCalls.length, callsBefore, 'the LAN chat surface did not request folders')

  assert.deepEqual(pageErrors, [], 'no page errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')
  console.log('watched folders browser smoke: 11 checks passed')
} finally {
  await browser.close()
  server.close()
}
