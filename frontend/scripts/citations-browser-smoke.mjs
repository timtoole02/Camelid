#!/usr/bin/env node
/* Browser-level acceptance for verifiable citations (F2a).
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app and deterministic fixtures; every cross-origin request is
 * aborted.
 *
 * The server-side contract is covered by Rust tests. What only a real render
 * can prove is that the viewer never shows a passage the server did not just
 * re-derive from the source:
 *
 *   - a verified citation highlights exactly the re-derived span, in context
 *   - a refused, legacy, failing or unreachable citation shows no source text,
 *     and in particular never the stale excerpt the answer was built from
 *   - the viewer sends back the exact bindings search returned
 *   - a slow verification cannot overwrite the citation opened after it
 *   - source text is rendered as text, never as markup
 *   - citations survive a reload and are re-verified, not cached
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

const PROMPT = 'What is the refund window?'
const STALE = 'STALE-EXCERPT-SENTINEL'
const HOSTILE = '<img src=x onerror="window.__citationXss=1"><script>window.__citationXss=2</script>'
const LONG_TOKEN = 'unbroken'.repeat(60)
const SLOW_MS = 1500

const bind = (n) => ({ chunk_sha256: String(n).repeat(64).slice(0, 64), doc_sha256: `${n}`.padEnd(64, 'd') })
const RESULTS = [
  { doc_id: 'doc-ok', filename: `${HOSTILE}.md`, chunk_index: 3, excerpt: `Refunds within 60 days. ${STALE}-ok`, byte_start: 120, byte_end: 180, ...bind(1) },
  { doc_id: 'doc-corrupted', filename: 'corrupted.txt', chunk_index: 0, excerpt: `Refunds within 30 days. ${STALE}-corrupted`, byte_start: 0, byte_end: 40, ...bind(2) },
  { doc_id: 'doc-500', filename: 'server-error.txt', chunk_index: 1, excerpt: `${STALE}-500`, byte_start: 10, byte_end: 20, ...bind(3) },
  { doc_id: 'doc-drop', filename: 'dropped.txt', chunk_index: 2, excerpt: `${STALE}-drop`, byte_start: 10, byte_end: 20, ...bind(4) },
  { doc_id: 'doc-slow', filename: 'slow.txt', chunk_index: 4, excerpt: `${STALE}-slow`, byte_start: 5, byte_end: 9, ...bind(5) },
  { doc_id: 'doc-legacy', filename: 'legacy.txt', chunk_index: 0, excerpt: `${STALE}-legacy`, byte_start: null, byte_end: null, chunk_sha256: null, doc_sha256: null },
  { filename: 'unbound.txt', chunk_index: 0, excerpt: `${STALE}-unbound` },
]
const ANSWER = 'Refunds take 60 days [1]. Stale [2]. Error [3]. Dropped [4]. Slow [5]. Legacy [6]. Unbound [7]. Missing [9].'
const VERIFIED = {
  doc_id: 'doc-ok', filename: `${HOSTILE}.md`, chunk_index: 3, byte_start: 120, byte_end: 180,
  chunk_sha256: bind(1).chunk_sha256, doc_sha256: bind(1).doc_sha256,
  before: 'Section 4. ', span: `Refunds within 60 days. ${HOSTILE} ${LONG_TOKEN}`, after: ' End of section.',
}

const MIME = {
  '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map((row) => row.contract) }

const searchRequests = []
const resolveRequests = []
const chatRequests = []
const pageErrors = []
const externalRequests = []
let slowAnswered = false

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
function refusal(res, code, message) {
  return sendJson(res, 409, { error: { message, type: 'invalid_request', code, param: null } })
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

async function answerResolve(req, res) {
  const body = await readJsonBody(req)
  resolveRequests.push(body)
  switch (body?.doc_id) {
    case 'doc-ok': return sendJson(res, 200, VERIFIED)
    case 'doc-corrupted': return refusal(res, 'citation_source_corrupted', 'The stored source text no longer matches its recorded hash.')
    case 'doc-legacy': return refusal(res, 'citation_unverifiable', 'This document predates verifiable citations. Re-ingest it to cite from it.')
    case 'doc-500':
      res.writeHead(500, { 'Content-Type': 'text/plain' })
      return res.end('internal failure, not json')
    case 'doc-drop': return req.socket.destroy()
    case 'doc-slow':
      await new Promise((done) => setTimeout(done, SLOW_MS))
      slowAnswered = true
      return sendJson(res, 200, { ...VERIFIED, doc_id: 'doc-slow', filename: 'slow.txt', span: 'SLOW-RESPONSE-SPAN' })
    default: return refusal(res, 'citation_unknown_document', 'The cited document is no longer in the library.')
  }
}

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: 'full',
        version: 'citations-browser-smoke', build: 'citations-browser-smoke',
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
    if (path === '/api/documents/search' && req.method === 'POST') {
      searchRequests.push(await readJsonBody(req))
      return sendJson(res, 200, { results: RESULTS.map((r) => ({ score: 0.5, retrieval: 'keyword', ...r })) })
    }
    if (path === '/api/documents/citation/resolve' && req.method === 'POST') return answerResolve(req, res)
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

const browser = await launchBrowser({ purpose: 'the citations browser smoke', headless: 'new' })
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
  if (window.sessionStorage.getItem('camelid.citationsSmokeInitialized')) return
  window.localStorage.clear()
  window.localStorage.setItem('camelid.attachedDocuments', JSON.stringify([
    { doc_id: 'doc-ok', filename: 'policy.md', chunk_count: 8, byte_size: 4096 },
  ]))
  window.sessionStorage.setItem('camelid.citationsSmokeInitialized', 'true')
})

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
  // Retrieval runs before generation starts, so wait for the request itself, not just idleness.
  const deadline = Date.now() + 30000
  while (chatRequests.length === before && Date.now() < deadline) await new Promise((done) => setTimeout(done, 50))
  assert.ok(chatRequests.length > before, 'the prompt reached the model endpoint')
  await page.waitForSelector('main[data-view="chat"] button.citation-pill', { timeout: 30000 })
  await idle()
}

const pill = (n) => `main[data-view="chat"] button.citation-pill[title="View source citation [${n}]"]`

async function openCitation(n) {
  await page.click(pill(n))
  await page.waitForSelector('.citation-modal', { timeout: 10000 })
}
async function settled() {
  await page.waitForFunction(() => {
    const badge = document.querySelector('.citation-modal .citation-modal__badge')
    return badge && !badge.textContent.includes('Verifying')
  }, { timeout: 10000 })
}
async function closeCitation() {
  await page.click('.citation-modal__footer button')
  await page.waitForFunction(() => !document.querySelector('.citation-modal'), { timeout: 10000 })
}
const modal = () => page.evaluate(() => {
  const root = document.querySelector('.citation-modal')
  if (!root) return null
  const body = root.querySelector('.citation-modal__body')
  return {
    title: root.querySelector('.citation-modal__title')?.textContent || '',
    badge: root.querySelector('.citation-modal__badge')?.textContent || '',
    badgeClass: root.querySelector('.citation-modal__badge')?.className || '',
    span: root.querySelector('.citation-modal__span')?.textContent ?? null,
    context: [...root.querySelectorAll('.citation-modal__context')].map((node) => node.textContent),
    refusalCode: root.querySelector('.citation-modal__refusal-code')?.textContent ?? null,
    refusalMessage: root.querySelector('.citation-modal__refusal-message')?.textContent ?? null,
    text: root.textContent,
    markup: root.querySelectorAll('img, script').length,
    overflow: body ? body.scrollWidth - body.clientWidth : 0,
  }
})

function assertRefused(view, code, label) {
  assert.ok(view, `${label}: the viewer opens`)
  assert.match(view.badge, /Refused/, `${label}: the badge says Refused`)
  assert.match(view.badgeClass, /--refused/, `${label}: with the refused style`)
  assert.equal(view.refusalCode, code, `${label}: the refusal names the typed reason`)
  assert.equal(view.span, null, `${label}: no passage is highlighted`)
  assert.doesNotMatch(view.text, new RegExp(STALE), `${label}: the stale excerpt the answer was built from is never shown`)
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') !== null
  ), { timeout: 30000 })

  /* ---- 1. retrieval feeds the answer and the pills ---------------------- */
  await sendPrompt(PROMPT)
  assert.equal(searchRequests.length, 1, 'sending with an attached document searches once')
  assert.deepEqual(searchRequests[0].doc_ids, ['doc-ok'], 'the search is scoped to the attached document')
  const sent = chatRequests[0].messages.filter((m) => m.role === 'user').at(-1).content
  assert.match(sent, /--- DOCUMENT CONTEXT ---/, 'the retrieved excerpts reach the model')
  const pills = await page.$$eval('main[data-view="chat"] button.citation-pill', (nodes) => nodes.map((n) => n.textContent))
  assert.deepEqual(pills, ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[9]'], 'every inline marker renders as a pill')

  /* ---- 2. a verified citation shows the re-derived span, in context ----- */
  await openCitation(1)
  await settled()
  const ok = await modal()
  assert.match(ok.badge, /Verified/, 'the badge says Verified')
  assert.match(ok.badge, /120.*180/, 'and names the byte range that was checked')
  assert.equal(ok.span, VERIFIED.span, 'the highlight is exactly the span the server re-derived')
  assert.deepEqual(ok.context, [VERIFIED.before, VERIFIED.after], 'surrounding source text frames it')
  assert.doesNotMatch(ok.text, new RegExp(STALE), 'the stored excerpt is not what the viewer shows')
  assert.equal(ok.markup, 0, 'hostile source text and filenames render as text, not markup')
  assert.ok(ok.text.includes('<img src=x'), 'the hostile text is shown literally')
  assert.equal(await page.evaluate(() => window.__citationXss), undefined, 'and none of it executed')
  assert.ok(ok.overflow <= 1, `a long unbroken token wraps instead of overflowing (overflow ${ok.overflow}px)`)
  const okRequest = resolveRequests.at(-1)
  assert.deepEqual(
    { doc_id: okRequest.doc_id, chunk_index: okRequest.chunk_index, chunk_sha256: okRequest.chunk_sha256, doc_sha256: okRequest.doc_sha256 },
    { doc_id: 'doc-ok', chunk_index: 3, chunk_sha256: bind(1).chunk_sha256, doc_sha256: bind(1).doc_sha256 },
    'the viewer sends back exactly the bindings search returned',
  )
  await closeCitation()

  /* ---- 3. every way verification can fail shows no source text ---------- */
  await openCitation(2)
  await settled()
  assertRefused(await modal(), 'citation_source_corrupted', 'a corrupted source')
  await closeCitation()

  await openCitation(3)
  await settled()
  assertRefused(await modal(), 'citation_refused', 'a non-JSON server error')
  await closeCitation()

  await openCitation(4)
  await settled()
  assertRefused(await modal(), 'citation_unreachable', 'a dropped connection')
  await closeCitation()

  await openCitation(6)
  await settled()
  assertRefused(await modal(), 'citation_unverifiable', 'a pre-F2a document')
  await closeCitation()

  const beforeUnbound = resolveRequests.length
  await openCitation(7)
  await settled()
  assertRefused(await modal(), 'citation_unverifiable', 'a citation with no source binding')
  assert.equal(resolveRequests.length, beforeUnbound, 'an unbound citation is refused without asking the server')
  await closeCitation()

  /* ---- 4. a marker with no citation behind it does nothing -------------- */
  const beforeMissing = resolveRequests.length
  await page.click(pill(9))
  await new Promise((done) => setTimeout(done, 300))
  assert.equal(await page.$('.citation-modal'), null, 'a dangling marker opens no viewer')
  assert.equal(resolveRequests.length, beforeMissing, 'and verifies nothing')

  /* ---- 5. a slow verification cannot overwrite the next citation -------- */
  await openCitation(5)
  const pending = await modal()
  assert.match(pending.badge, /Verifying/, 'verification is visibly in flight')
  assert.equal(pending.span, null, 'nothing is highlighted before verification finishes')
  assert.doesNotMatch(pending.text, new RegExp(STALE), 'and the stale excerpt is not shown as a placeholder')
  await closeCitation()
  await openCitation(2)
  await settled()
  const deadline = Date.now() + SLOW_MS * 3
  while (!slowAnswered && Date.now() < deadline) await new Promise((done) => setTimeout(done, 50))
  assert.ok(slowAnswered, 'the slow verification did answer')
  await new Promise((done) => setTimeout(done, 300))
  const afterRace = await modal()
  assertRefused(afterRace, 'citation_source_corrupted', 'the citation opened last')
  assert.match(afterRace.title, /corrupted\.txt/, 'the viewer still describes the citation opened last')
  assert.doesNotMatch(afterRace.text, /SLOW-RESPONSE-SPAN/, 'the late answer for the closed citation is discarded')
  await closeCitation()

  /* ---- 6. citations survive a reload and are re-verified ---------------- */
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector(pill(1), { timeout: 30000 })
  const beforeReload = resolveRequests.filter((r) => r.doc_id === 'doc-ok').length
  await openCitation(1)
  await settled()
  const reloaded = await modal()
  assert.match(reloaded.badge, /Verified/, 'a citation from a restored conversation still verifies')
  assert.equal(
    resolveRequests.filter((r) => r.doc_id === 'doc-ok').length,
    beforeReload + 1,
    'every opening asks the server again; nothing is cached',
  )
  await page.click('.citation-modal-overlay', { offset: { x: 5, y: 5 } })
  await page.waitForFunction(() => !document.querySelector('.citation-modal'), { timeout: 10000 })

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')

  console.log('citations browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
