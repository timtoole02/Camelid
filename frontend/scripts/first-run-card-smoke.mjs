#!/usr/bin/env node
/* Component-level coverage for the first-run card's two hard transitions:
 * retrying AFTER the artifact landed, and cancelling when the backend disagrees
 * that anything was cancelled.
 *
 * These are races between the card's polling and the download lifecycle, so they
 * cannot be proved by rendering markup or by calling the pure helpers -- the point
 * is what the mounted component DOES over time. This drives the real component in a
 * real browser against a scripted backend, so every branch is deterministic: the
 * test decides exactly what `/catalog/cancel` answers and exactly when the file
 * appears in the local scan.
 *
 * Requires `npm run build` first (it serves frontend/dist) and Chrome/Edge.
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, readFileSync } from 'node:fs'
import { extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const distDir = resolve(scriptDir, '../dist')
const ledgerPath = resolve(scriptDir, '../../ledger/camelid-ledger.json')

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const executablePath = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter(Boolean).find(existsSync)
if (!executablePath) throw new Error('Chrome or Edge is required for the first-run card smoke')

/* The real shipped contract, so the card's supported-row derivation is not a fixture. */
const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = {
  ...ledger.capabilities,
  model_compatibility: ledger.model_rows.map((row) => row.contract),
}

const FILENAME = 'Qwen3-0.6B-Q8_0.gguf'
const CATALOG_ID = 'qwen3_0_6b_instruct_q8_0'
const CATALOG_ITEM = {
  catalog_id: CATALOG_ID,
  name: 'Qwen3 0.6B Q8_0',
  repo_id: 'Qwen/Qwen3-0.6B-GGUF',
  filename: FILENAME,
  size_bytes: 639446688,
  downloads: 0,
  likes: 0,
  quant: 'Q8_0',
  architecture: 'qwen3',
  license: 'apache-2.0',
  oracle_qualified: true,
  group: 'curated',
  arch_detected: true,
  fit: 'cpu_only_ok',
  task_tags: ['general'],
  fit_confidence: 'exact',
  arch_support: 'unknown',
}

/* Everything the scenario controls. Reset between scenarios. */
let stub
function resetStub() {
  stub = {
    downloads: [],            // what /catalog/downloads returns
    localModels: [],          // what /models/local returns
    cancelResponse: { status: 200, body: 'Download canceled' },
    onCancelRequest: null,      // fires server-side when /catalog/cancel arrives
    cancelDelayMs: 0,           // holds the cancel open so other work can interleave
    downloadsStatus: 200,     // force a failed probe
    localStatus: 200,
    loadStatus: 200,
    healthGenerationReady: false,
    activeModelId: null,
    requests: [],             // every method+path the app issued
  }
}
resetStub()

const countRequests = (path) => stub.requests.filter((entry) => entry.path === path).length

/* Wait on the SERVER's view of what the app did, for assertions about requests the
   card must or must not issue. */
async function waitUntil(predicate, description, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (predicate()) return
    await new Promise((done) => setTimeout(done, 100))
  }
  throw new Error(`timed out waiting: ${description}`)
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.json': 'application/json', '.png': 'image/png' }

function sendJson(res, status, body) {
  const payload = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) })
  res.end(payload)
}

async function readBody(req) {
  const chunks = []
  for await (const chunk of req) chunks.push(chunk)
  if (!chunks.length) return {}
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')) } catch { return {} }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1')
  const path = url.pathname
  stub.requests.push({ method: req.method, path })

  if (path === '/v1/health') {
    return sendJson(res, 200, {
      ok: true,
      engine: 'camelid',
      loaded_now: Boolean(stub.activeModelId),
      generation_ready: stub.healthGenerationReady,
      active_model_id: stub.activeModelId,
    })
  }
  if (path === '/v1/models') return sendJson(res, 200, { object: 'list', data: [] })
  if (path === '/api/capabilities') return sendJson(res, 200, capabilities)
  if (path === '/api/models/catalog') return sendJson(res, 200, { items: [CATALOG_ITEM], next_cursor: null })
  if (path === '/api/models/catalog/downloads') {
    if (stub.downloadsStatus !== 200) return sendJson(res, stub.downloadsStatus, { error: { message: 'probe failed' } })
    return sendJson(res, 200, stub.downloads)
  }
  if (path === '/api/models/local') {
    if (stub.localStatus !== 200) return sendJson(res, stub.localStatus, { error: { message: 'scan failed' } })
    return sendJson(res, 200, { models_dir: 'models', models: stub.localModels })
  }
  if (path === '/api/models/catalog/install') {
    await readBody(req)
    stub.downloads = [{
      id: CATALOG_ID, repo_id: CATALOG_ITEM.repo_id, filename: FILENAME,
      continuation_mode: 'start', total_bytes: CATALOG_ITEM.size_bytes,
      bytes_downloaded: 1024, status: 'downloading',
    }]
    return sendJson(res, 200, { started: true })
  }
  if (path === '/api/models/catalog/cancel') {
    await readBody(req)
    // Lets a scenario make the download COMPLETE exactly while the cancel is in
    // flight -- the real race, and the only way to click Cancel before the card
    // has already noticed the file and unmounted the button.
    stub.onCancelRequest?.()
    if (stub.cancelDelayMs) await new Promise((done) => setTimeout(done, stub.cancelDelayMs))
    const { status, body } = stub.cancelResponse
    if (status === 200) stub.downloads = []
    return sendJson(res, status, typeof body === 'string' ? { message: body } : body)
  }
  if (path === '/api/models/catalog/ack') { await readBody(req); res.writeHead(204); return res.end() }
  if (path === '/api/models/inspect') { await readBody(req); return sendJson(res, 200, { architecture: 'qwen3' }) }
  if (path === '/api/models/load') {
    await readBody(req)
    if (stub.loadStatus !== 200) {
      return sendJson(res, stub.loadStatus, { error: { code: 'model_io_error', message: 'simulated load failure' } })
    }
    stub.activeModelId = FILENAME
    stub.healthGenerationReady = true
    return sendJson(res, 200, { id: FILENAME })
  }
  if (path === '/api/models/current') {
    if (!stub.activeModelId) return sendJson(res, 404, { error: { message: 'no model' } })
    return sendJson(res, 200, { path: `models/${FILENAME}` })
  }
  if (path === '/v1/chat/completions') { await readBody(req); return sendJson(res, 200, { choices: [{ message: { content: 'ok' } }] }) }

  // Static app
  const filePath = join(distDir, path === '/' ? 'index.html' : path.replace(/^\//, ''))
  if (existsSync(filePath) && !filePath.includes('..')) {
    const body = readFileSync(filePath)
    res.writeHead(200, { 'Content-Type': MIME[extname(filePath)] || 'application/octet-stream' })
    return res.end(body)
  }
  res.writeHead(404, { 'Content-Type': 'application/json' })
  res.end('{}')
})

await new Promise((done) => server.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${server.address().port}`

const browser = await puppeteer.launch({ executablePath, headless: 'new' })
const pageErrors = []

async function openCard() {
  const page = await browser.newPage()
  page.on('pageerror', (error) => pageErrors.push(String(error)))
  await page.goto(origin, { waitUntil: 'networkidle2' })
  await page.waitForSelector('.cxfirstrun__primary', { timeout: 30000 })
  return page
}

const cardText = () => {
  const card = document.querySelector('.cxfirstrun')
  return {
    stage: card?.querySelector('.cxfirstrun__stage')?.textContent?.trim() || null,
    error: card?.querySelector('.cxfirstrun__error')?.textContent?.trim() || null,
    notice: card?.querySelector('.cxfirstrun__notice')?.textContent?.trim() || null,
    primary: card?.querySelector('.cxfirstrun__primary')?.textContent?.trim() || null,
    progress: Boolean(card?.querySelector('.cxfirstrun__progress')),
    gone: !card,
  }
}

/* Land the download: the record completes and the file appears in the scan. */
function landArtifact() {
  stub.downloads = [{ ...stub.downloads[0], status: 'completed', bytes_downloaded: CATALOG_ITEM.size_bytes }]
  stub.localModels = [{
    filename: FILENAME, size_bytes: CATALOG_ITEM.size_bytes, architecture: 'qwen3',
    quantization: 'Q8_0', admitted: true, oracle_qualified: true, lane_class: 'supported',
    chat_capable: true, runnable_receipt_present: false,
  }]
}

const results = []
async function scenario(name, body) {
  resetStub()
  const page = await openCard()
  try {
    await body(page)
    results.push(`ok   ${name}`)
  } catch (error) {
    results.push(`FAIL ${name}: ${error.message}`)
    throw error
  } finally {
    await page.close()
  }
}

try {
  /* ---- 1. A failure after the file landed must retry ACTIVATION ------------ */
  await scenario('retry after a landed download re-activates instead of re-downloading', async (page) => {
    stub.loadStatus = 500
    await page.click('.cxfirstrun__primary')
    await page.waitForFunction(() => document.querySelector('.cxfirstrun__progress'), { timeout: 15000 })
    landArtifact()

    await page.waitForFunction(
      () => document.querySelector('.cxfirstrun__error'),
      { timeout: 20000, polling: 250 },
    )
    const failed = await page.evaluate(cardText)
    assert.match(failed.error, /simulated load failure/, 'the typed load failure is surfaced')
    assert.equal(failed.primary, 'Retry setup', 'a landed artifact retries setup, not the download')
    assert.match(failed.notice, /already downloaded/, 'the card says the file is already there')

    const installsBefore = countRequests('/api/models/catalog/install')
    const loadsBefore = countRequests('/api/models/load')
    assert.equal(installsBefore, 1, 'exactly one install so far')

    /* The retry button has to still BE there to be useful. Once the artifact lands
       the host stops looking like a fresh install, so the app-level mount condition
       drops the card unless the card reports that it still owns the flow -- and the
       dashboard refreshes on a 2.5s cadence, so a test that clicks retry immediately
       passes whether or not that is true. Dwell past two refreshes and check. */
    await new Promise((done) => setTimeout(done, 7000))
    const afterDwell = await page.evaluate(cardText)
    assert.equal(afterDwell.gone, false, 'the failed card must survive the dashboard refresh that sees the landed file')
    assert.equal(afterDwell.primary, 'Retry setup', 'and must keep offering the retry, not strand a downloaded artifact')

    // The retry succeeds this time.
    stub.loadStatus = 200
    await page.click('.cxfirstrun__primary')

    /* Assert the SHAPE of the retry before waiting for it to finish: `startDownload`
       POSTs /install immediately on click, so if the retry took that path it shows up
       here. Waiting for completion first would only ever fail as a timeout, which
       hides which of the two paths ran. */
    await waitUntil(
      () => countRequests('/api/models/load') > loadsBefore || countRequests('/api/models/catalog/install') > installsBefore,
      'the retry issued no request at all',
    )
    assert.equal(
      countRequests('/api/models/catalog/install'),
      installsBefore,
      'retrying a landed artifact must NOT issue a second install (that would refetch 610 MB)',
    )
    assert.ok(
      countRequests('/api/models/load') > loadsBefore,
      'the retry must actually re-run the load it failed on',
    )

    await page.waitForFunction(() => !document.querySelector('.cxfirstrun'), { timeout: 30000, polling: 250 })
    assert.equal(countRequests('/api/models/catalog/install'), installsBefore, 'and still no second install after it completes')
    assert.ok(countRequests('/api/models/catalog/ack') >= 1, 'the completed download is acknowledged')
  })

  /* ---- 2. Cancel that lost the race (409) must not claim "nothing installed" */
  await scenario('cancel after completion activates the artifact instead of reporting a false cancel', async (page) => {
    await page.click('.cxfirstrun__primary')
    await page.waitForFunction(() => document.querySelector('.cxfirstrun__progress'), { timeout: 15000 })

    /* The download finishes while the cancel is in flight: the backend refuses the
       cancel with 409 and KEEPS the file (cancel is not delete). */
    stub.cancelResponse = { status: 409, body: { error: { code: 'download_already_completed', message: 'already finished' } } }
    stub.onCancelRequest = () => landArtifact()

    await page.click('.cxfirstrun__progress .cxfirstrun__secondary')
    await page.waitForFunction(
      () => !document.querySelector('.cxfirstrun') || document.querySelector('.cxfirstrun__error'),
      { timeout: 30000, polling: 250 },
    )
    const state = await page.evaluate(cardText)
    assert.equal(state.error, null, 'a 409 cancel over a completed file must not report a failure')
    assert.ok(state.gone, 'the artifact must be activated, not discarded')
    assert.ok(countRequests('/api/models/load') >= 1, 'the already-downloaded artifact is routed into activation')
    assert.equal(countRequests('/api/models/catalog/install'), 1, 'and it is never re-downloaded')
  })

  /* ---- 2b. Cancel and settlement racing for the SAME activation ------------
     The cancel is held open while the file lands, so the settlement poll claims the
     activation first and the cancel's own reconciliation arrives second. Both want to
     activate; only one may. Without single-flight this runs two loads and two
     acknowledgements for one file. */
  await scenario('a cancel racing settlement activates the artifact exactly once', async (page) => {
    await page.click('.cxfirstrun__primary')
    await page.waitForFunction(() => document.querySelector('.cxfirstrun__progress'), { timeout: 15000 })

    stub.cancelResponse = { status: 409, body: { error: { code: 'download_already_completed', message: 'already finished' } } }
    stub.onCancelRequest = () => landArtifact()   // lands immediately...
    stub.cancelDelayMs = 2500                      // ...but the cancel answers late

    await page.click('.cxfirstrun__progress .cxfirstrun__secondary')
    await page.waitForFunction(() => !document.querySelector('.cxfirstrun'), { timeout: 30000, polling: 250 })
    // Let any second activation that was going to happen actually happen.
    await new Promise((done) => setTimeout(done, 2000))

    assert.equal(countRequests('/api/models/load'), 1, 'the model must be loaded exactly once')
    assert.equal(countRequests('/api/models/catalog/ack'), 1, 'and acknowledged exactly once')
    assert.equal(countRequests('/api/models/catalog/install'), 1, 'and never re-downloaded')
  })

  /* ---- 3. Cancel the backend did not honour must keep watching ------------- */
  await scenario('a cancel that did not stop the download resumes instead of freezing on failed', async (page) => {
    await page.click('.cxfirstrun__primary')
    await page.waitForFunction(() => document.querySelector('.cxfirstrun__progress'), { timeout: 15000 })

    // 404: no such download -- but it is in fact still running.
    stub.cancelResponse = { status: 404, body: { error: { message: 'Download not found' } } }
    await page.click('.cxfirstrun__progress .cxfirstrun__secondary')

    await page.waitForFunction(
      () => document.querySelector('.cxfirstrun__notice'),
      { timeout: 20000, polling: 250 },
    )
    const state = await page.evaluate(cardText)
    assert.match(state.notice, /did not stop/, 'the card admits the cancel did not take')
    assert.equal(state.error, null, 'and must not report a failure')
    assert.ok(state.progress, 'the download stays visible and watched')

    // Proof it is still genuinely watched: it can still complete.
    landArtifact()
    await page.waitForFunction(() => !document.querySelector('.cxfirstrun'), { timeout: 30000, polling: 250 })
  })

  /* ---- 4. Unreadable probes must never produce a false all-clear ----------- */
  await scenario('a cancel with unreadable probes refuses to claim anything', async (page) => {
    await page.click('.cxfirstrun__primary')
    await page.waitForFunction(() => document.querySelector('.cxfirstrun__progress'), { timeout: 15000 })

    stub.cancelResponse = { status: 500, body: { error: { message: 'boom' } } }
    stub.downloadsStatus = 500
    stub.localStatus = 500
    await page.click('.cxfirstrun__progress .cxfirstrun__secondary')

    await page.waitForFunction(
      () => document.querySelector('.cxfirstrun__notice'),
      { timeout: 20000, polling: 250 },
    )
    const state = await page.evaluate(cardText)
    assert.match(state.notice, /could not confirm/, 'an unverifiable cancel says so')
    assert.equal(state.error, null, 'it must not claim "nothing was installed" without looking')
    assert.ok(state.progress, 'and keeps watching')
  })

  /* ---- 5. A genuine cancel still reports a clean, retryable state ---------- */
  await scenario('a confirmed cancel reports nothing installed and offers a fresh download', async (page) => {
    await page.click('.cxfirstrun__primary')
    await page.waitForFunction(() => document.querySelector('.cxfirstrun__progress'), { timeout: 15000 })

    stub.cancelResponse = { status: 200, body: 'Download canceled' }
    await page.click('.cxfirstrun__progress .cxfirstrun__secondary')

    await page.waitForFunction(
      () => document.querySelector('.cxfirstrun__error'),
      { timeout: 20000, polling: 250 },
    )
    const state = await page.evaluate(cardText)
    assert.match(state.error, /Download canceled\. Nothing was installed/)
    assert.equal(state.primary, 'Try again', 'nothing landed, so the retry is a fresh download')
    assert.equal(state.notice, null, 'and no "already downloaded" claim')

    const installsBefore = countRequests('/api/models/catalog/install')
    await page.click('.cxfirstrun__primary')
    await page.waitForFunction(() => document.querySelector('.cxfirstrun__progress'), { timeout: 15000 })
    assert.equal(
      countRequests('/api/models/catalog/install'),
      installsBefore + 1,
      'with nothing on disk the retry must issue a real install',
    )
  })

  assert.deepEqual(pageErrors, [], 'the card must not raise page errors')
  for (const line of results) console.log(line)
  console.log('first-run card smoke: all checks passed')
} catch (error) {
  for (const line of results) console.log(line)
  if (pageErrors.length) console.error('page errors:', pageErrors)
  console.error(error)
  process.exitCode = 1
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
