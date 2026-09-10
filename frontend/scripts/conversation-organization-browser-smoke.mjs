#!/usr/bin/env node
/* Browser-level acceptance for pinning, archiving and tagging.
 *
 * Requires `npm run build` first. The pure smoke owns the rules; this owns the
 * wiring, which is where an organization feature actually fails: a control
 * that updates storage but not the list, a pin that still falls off the end,
 * a filter that hides everything, an archive with no way back.
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
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
function isFile(path) { try { return statSync(path).isFile() } catch { return false } }

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: 'full', version: 'organization-smoke', build: 'organization-smoke',
        backend: 'llama', model_family: 'qwen3', loaded_now: true, generation_ready: true,
        active_model_id: MODEL_FILENAME, active_context_length: 4096, max_prompt_tokens: 4096, max_generation_tokens: 8192,
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
    const filePath = resolve(distDir, `.${path}`)
    if (path !== '/' && isFile(filePath)) return sendFile(res, filePath)
    return sendFile(res, resolve(distDir, 'index.html'))
  } catch (error) {
    if (!res.writableEnded) sendJson(res, 500, { error: String(error) })
  }
})

await new Promise((done) => server.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${server.address().port}`

/* Eight seeded conversations: more than the rail's six-recent limit, so
   "pinned survives the limit" is a real assertion and not a coincidence. */
const SEEDED = Array.from({ length: 8 }, (_, i) => ({
  id: `conversation-seed-${i}`,
  title: `Seeded chat ${i}`,
  created_at: `2026-09-0${(i % 8) + 1}T09:00:00.000Z`,
  updated_at: `2026-09-0${(i % 8) + 1}T10:00:00.000Z`,
  messages: [
    { id: `s${i}-u`, role: 'user', content: `question ${i}` },
    { id: `s${i}-a`, role: 'assistant', content: `answer ${i}`, finish_reason: 'stop' },
  ],
}))
const OLDEST = 'Seeded chat 0'

const browser = await launchBrowser({ purpose: 'the conversation organization browser smoke', headless: 'new' })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 1000, deviceScaleFactor: 1 })
page.on('pageerror', (error) => pageErrors.push(String(error)))
await page.setRequestInterception(true)
page.on('request', (request) => {
  const url = request.url()
  if (url.startsWith('data:') || url.startsWith('blob:')) return request.continue()
  try { if (new URL(url).origin === origin) return request.continue() } catch { /* abort below */ }
  externalRequests.push(url)
  return request.abort()
})
await page.evaluateOnNewDocument((seeded) => {
  if (window.sessionStorage.getItem('camelid.orgSmokeInitialized')) return
  window.localStorage.clear()
  window.localStorage.setItem('camelid.conversations', JSON.stringify(seeded))
  window.sessionStorage.setItem('camelid.orgSmokeInitialized', 'true')
}, SEEDED)

const railTitles = () => page.$$eval('.rail-convo__main', (nodes) => nodes.map((n) => n.textContent))
const groupLabels = () => page.$$eval('.rail__group-label', (nodes) => nodes.map((n) => n.textContent))

async function openMenuFor(titleFragment) {
  const opened = await page.evaluate((fragment) => {
    const rows = [...document.querySelectorAll('.rail-convo')]
    const row = rows.find((node) => node.textContent.includes(fragment))
    const button = row?.querySelector('.rail-convo__menu-btn')
    if (!button) return false
    button.click()
    return true
  }, titleFragment)
  assert.equal(opened, true, `no menu button for ${titleFragment}`)
  await page.waitForSelector('.rail-menu', { timeout: 10000 })
}

async function clickMenuItem(label) {
  const clicked = await page.evaluate((text) => {
    const item = [...document.querySelectorAll('.rail-menu__item')].find((n) => n.textContent.trim() === text)
    if (!item) return false
    item.click()
    return true
  }, label)
  assert.equal(clicked, true, `no menu item labelled ${label}`)
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('.rail-convo', { timeout: 30000 })

  /* ---- baseline: the rail is capped and the oldest is off the end ------- */
  const initial = await railTitles()
  assert.equal(initial.length, 6, 'the rail shows six recent conversations')
  assert.equal(initial.some((t) => t.includes(OLDEST)), false, 'the oldest chat has fallen off the list — the problem pinning exists to solve')
  assert.equal((await groupLabels()).includes('Pinned'), false, 'no Pinned group before anything is pinned')

  /* ---- pin the oldest: it must come back and stay ---------------------- */
  await page.evaluate(() => {
    // It is off the rail, so reach it the way a user would: search.
    const input = document.querySelector('.rail__search-input')
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
    setter.call(input, 'Seeded chat 0')
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await page.waitForFunction(() => document.querySelectorAll('.rail-convo').length === 1, { timeout: 10000 })
  await openMenuFor(OLDEST)
  await clickMenuItem('Pin to top')
  await page.evaluate(() => {
    const input = document.querySelector('.rail__search-input')
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
    setter.call(input, '')
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await page.waitForFunction(() => (
    [...document.querySelectorAll('.rail__group-label')].some((n) => n.textContent === 'Pinned')
  ), { timeout: 10000 })

  const afterPin = await railTitles()
  assert.equal(afterPin[0].includes(OLDEST), true, 'the pinned chat is now first')
  assert.equal(
    afterPin.length,
    7,
    'a pin is NOT subject to the six-recent limit — it would be worthless if it fell off again',
  )
  assert.equal((await groupLabels())[0], 'Pinned', 'and it has its own group above the date buckets')

  /* ---- tag it, and filter by that tag ---------------------------------- */
  await openMenuFor(OLDEST)
  await page.waitForSelector('.rail-menu__tag-input', { timeout: 10000 })
  await page.type('.rail-menu__tag-input', 'CUDA')
  await page.keyboard.press('Enter')
  await page.waitForFunction(() => (
    [...document.querySelectorAll('.rail-convo__tag')].some((n) => n.textContent === 'cuda')
  ), { timeout: 10000 })
  const tagInputAfter = await page.$eval('.rail-menu__tag-input', (n) => n.value)
  assert.equal(tagInputAfter, '', 'the tag box clears so a second tag can be typed without reopening the menu')
  await page.keyboard.press('Escape')

  await page.waitForSelector('.rail__tag-filter', { timeout: 10000 })
  const chips = await page.$$eval('.rail__tag-chip', (nodes) => nodes.map((n) => n.textContent))
  assert.deepEqual(chips, ['cuda'], 'the tag appears in the filter row, lowercased')

  await page.click('.rail__tag-chip')
  await page.waitForFunction(() => document.querySelectorAll('.rail-convo').length === 1, { timeout: 10000 })
  const filtered = await railTitles()
  assert.equal(filtered[0].includes(OLDEST), true, 'filtering by the tag narrows the list to it')
  await page.click('.rail__tag-chip--clear')
  await page.waitForFunction(() => document.querySelectorAll('.rail-convo').length > 1, { timeout: 10000 })

  /* ---- archive: gone from the list, with a way back -------------------- */
  const archiveTarget = 'Seeded chat 7'
  await openMenuFor(archiveTarget)
  await clickMenuItem('Archive')
  await page.waitForFunction((title) => (
    ![...document.querySelectorAll('.rail-convo__main')].some((n) => n.textContent.includes(title))
  ), { timeout: 10000 }, archiveTarget)

  await page.waitForSelector('.rail__archived-toggle', { timeout: 10000 })
  const toggleLabel = await page.$eval('.rail__archived-toggle', (n) => n.textContent)
  assert.match(toggleLabel, /Show archived \(1\)/, 'the archived count is visible — archiving must not look like deletion')

  await page.click('.rail__archived-toggle')
  await page.waitForFunction((title) => (
    [...document.querySelectorAll('.rail-convo__main')].some((n) => n.textContent.includes(title))
  ), { timeout: 10000 }, archiveTarget)

  /* ---- it all survives a reload ---------------------------------------- */
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('.rail-convo', { timeout: 30000 })
  const stored = await page.evaluate(() => {
    const list = JSON.parse(localStorage.getItem('camelid.conversations') || '[]')
    return {
      pinned: list.filter((c) => c.pinned).map((c) => c.title),
      archived: list.filter((c) => c.archived).map((c) => c.title),
      tags: list.find((c) => c.pinned)?.tags || [],
      // Organizing is not editing: it must not reshuffle recency.
      pinnedUpdatedAt: list.find((c) => c.pinned)?.updated_at,
    }
  })
  assert.deepEqual(stored.pinned, [OLDEST], 'the pin survives a reload')
  assert.deepEqual(stored.archived, ['Seeded chat 7'], 'so does the archive')
  assert.deepEqual(stored.tags, ['cuda'], 'so does the tag')
  assert.equal(
    stored.pinnedUpdatedAt,
    '2026-09-01T10:00:00.000Z',
    'pinning and tagging leave updated_at alone — organizing a thread is not editing it',
  )
  assert.equal((await groupLabels())[0], 'Pinned', 'and the Pinned group is still first after a reload')


  /* ---- export -> import round trip through the real controls ------------ */
  /* Navigate the way a user does, through the rail, rather than by poking the
     hash: it also proves the History nav item still reaches this view. */
  const wentToHistory = await page.evaluate(() => {
    const item = [...document.querySelectorAll('.rail__nav-item')].find((n) => n.textContent.includes('Chat history'))
    if (!item) return false
    item.click()
    return true
  })
  assert.equal(wentToHistory, true, 'the rail offers a Chat history entry')
  await page.waitForSelector('.history-view__tools', { timeout: 30000 })

  const exportFile = resolve(tmpdir(), `camelid-import-smoke-${process.pid}.json`)
  writeFileSync(exportFile, JSON.stringify({
    format: 'camelid.conversations/v1',
    conversations: [{
      /* Deliberately carries an id that already exists here plus a local path
         the exporter never writes: the import must take neither. */
      id: 'conversation-seed-0',
      title: 'IMPORTED-THREAD',
      model_path: '/home/someone/.ssh/id_rsa',
      tags: ['Imported'],
      messages: [
        { role: 'user', content: 'imported question' },
        { role: 'assistant', content: 'imported answer', usage: { prompt_tokens: 4, completion_tokens: 6 }, usage_source: 'backend' },
      ],
    }],
  }))

  const beforeImport = await page.evaluate(() => JSON.parse(localStorage.getItem('camelid.conversations') || '[]').length)
  const chooser = await page.$('.history-view__import-input')
  await chooser.uploadFile(exportFile)
  await page.waitForFunction((count) => (
    JSON.parse(localStorage.getItem('camelid.conversations') || '[]').length === count + 1
  ), { timeout: 15000 }, beforeImport)

  const importedState = await page.evaluate(() => {
    const list = JSON.parse(localStorage.getItem('camelid.conversations') || '[]')
    const imported = list.find((c) => c.title === 'IMPORTED-THREAD')
    return {
      total: list.length,
      seedZeroIntact: list.filter((c) => c.id === 'conversation-seed-0').length,
      id: imported?.id,
      tags: imported?.tags,
      hasPath: Object.prototype.hasOwnProperty.call(imported || {}, 'model_path'),
      usageSource: imported?.messages?.[1]?.usage_source,
      messages: (imported?.messages || []).map((m) => m.content),
    }
  })
  assert.equal(importedState.total, beforeImport + 1, 'the file adds exactly one conversation')
  assert.equal(importedState.seedZeroIntact, 1, 'the existing conversation whose id the file claimed is untouched')
  assert.notEqual(importedState.id, 'conversation-seed-0', 'the import got a fresh id instead of overwriting')
  assert.deepEqual(importedState.messages, ['imported question', 'imported answer'], 'the messages arrived')
  assert.deepEqual(importedState.tags, ['imported'], 'tags arrive normalized')
  assert.equal(importedState.hasPath, false, 'a field the exporter never writes cannot ride in on an import')
  assert.equal(importedState.usageSource, 'client_estimate', 'imported counts cannot claim to be this backend’s reported usage')
  rmSync(exportFile, { force: true })

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')

  console.log('conversation organization browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
