#!/usr/bin/env node
/* End-to-end UI acceptance using deterministic, local-only API fixtures.
 * This verifies wiring; it makes no claim about a real model's tool quality. */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, readFileSync, statSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { extname, resolve } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import { launchBrowser } from './lib/launch-browser.mjs'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const distDir = resolve(scriptDir, '../dist')
const ledgerPath = resolve(scriptDir, '../../ledger/camelid-ledger.json')
const MODEL_FILENAME = 'Qwen3-0.6B-Q8_0.gguf'
const MODEL_ID = 'qwen3_0_6b_instruct_q8_0'

const MIME = {
  '.css': 'text/css',
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = {
  ...ledger.capabilities,
  model_compatibility: ledger.model_rows.map((row) => ({ ...row.contract, tool_capable: true })),
}

const tool = { key: 'mcp_fixture_echo', name: 'echo', description: 'Returns the text you provide.', input_schema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] } }
const connection = { config: { id: 'a'.repeat(32), name: 'Local test tools', transport: 'stdio', command: 'test-fixture' }, connected: true, tools: [tool] }
const decisions = [], prepared = []
const chatRequests = []
const htmlOutput = '<h1>Preview works</h1><p>Generated report</p><script>parent.__escapedPreview = true; fetch("https://example.invalid/probe")</script><img src="https://example.invalid/image.png">'
const htmlSource = htmlOutput + '\n'
const markdown = 'Here are your files.\n\n```html\n' + htmlOutput + '\n```\n\n```csv\nname,count\nCamelid,3\n```\n\n```json\n{"ok":true}\n```\n\n```markdown\n# Formatted report\n\nThis is **ready**.\n```'
const reviews = []
let currentFile = '<h1>Before</h1>', fileWrites = 0
const downloads = mkdtempSync(resolve(tmpdir(), 'camelid-outputs-downloads-'))
mkdirSync(resolve(scriptDir, '../../target'), { recursive: true })
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

function sendChatCompletion(res, content, finishReason, completionTokens) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
  res.socket?.setNoDelay(true)
  res.flushHeaders()
  const frame = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)
  frame({ choices: [{ delta: { role: 'assistant' } }] })
  frame({ choices: [{ delta: { content } }] })
  frame({
    choices: [{ delta: {}, finish_reason: finishReason }],
    usage: { prompt_tokens: 40, completion_tokens: completionTokens, total_tokens: 40 + completionTokens },
  })
  res.write('data: [DONE]\n\n')
  res.end()
}

function isFile(path) {
  try { return statSync(path).isFile() } catch { return false }
}

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/api/changes') {
      if (req.method === 'GET') return sendJson(res, 200, { reviews })
      const body = await readJsonBody(req)
      const review = { id: String(reviews.length + 1).padStart(32,'0'), workspace: body.workspace, path: body.path, before: currentFile, after: body.content, source: body.source, status: 'pending', created_at: Date.now(), created: false, before_bytes: Buffer.byteLength(currentFile), after_bytes: Buffer.byteLength(body.content), diff: '- ' + currentFile + '\n+ ' + body.content }
      reviews.unshift(review)
      return sendJson(res, 200, review)
    }
    if (path.startsWith('/api/changes/')) {
      const [, , , id, action] = path.split('/')
      const review = reviews.find(r => r.id === id)
      if (!review) return sendJson(res, 404, {error:{message:'Review not found'}})
      if (action === 'decision') {
        const body = await readJsonBody(req)
        if (review.status !== 'pending') return sendJson(res, 409, {error:{message:'Already decided'}})
        if (body.approved) { currentFile = review.after; fileWrites++; review.status = 'applied'; return sendJson(res,503,{error:{message:'The action response was lost.'}}) } else review.status = 'rejected'
      }
      if (action === 'undo') {
        if (currentFile !== review.after) return sendJson(res,409,{error:{message:'The file changed after this version was reviewed. Nothing was overwritten; create a new review.'}})
        currentFile=review.before; fileWrites++; review.status='undone'
      }
      return sendJson(res,200,review)
    }
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true,
        engine: 'camelid',
        api_surface: 'full',
        version: 'chat-continuation-browser-smoke',
        build: 'chat-continuation-browser-smoke',
        backend: 'llama',
        model_family: 'qwen3',
        loaded_now: true,
        generation_ready: true,
        active_model_id: MODEL_ID,
        active_context_length: 4096,
        max_prompt_tokens: 4096,
        max_generation_tokens: 8192,
      })
    }
    if (path === '/v1/models') {
      return sendJson(res, 200, {
        object: 'list',
        data: [{
          id: MODEL_ID,
          object: 'model',
          created: 0,
          owned_by: 'camelid',
          meta: { n_ctx_train: 32768, n_params: 600000000, size: 639446688 },
        }],
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
        id: MODEL_ID,
        path: `models/${MODEL_FILENAME}`,
        gguf: { metadata: { general: { architecture: 'qwen3', file_type: 7 } } },
        tokenizer: { status: 'available' },
      })
    }
    if (path === '/api/web/research' && req.method === 'POST') {
      return sendJson(res, 200, { status: 'skipped', triggered: false, reason: 'not_needed', sources: [], warnings: [] })
    }
    if (path === '/api/mcp/connections') return sendJson(res, 200, { connections: [connection] })
    if (path === '/api/mcp/calls' && req.method === 'POST') {
      const body = await readJsonBody(req); prepared.push(body)
      return sendJson(res, 201, { id: 'approval-1', status: 'pending', connection_name: connection.config.name, tool: tool.name, arguments: body.arguments })
    }
    if (path.endsWith('/decision')) {
      const body = await readJsonBody(req); decisions.push(body)
      return sendJson(res, 202, { status: body.approved ? 'complete' : 'denied', result: body.approved ? { content: [{ type: 'text', text: 'Hello from MCP' }] } : null })
    }
    if (path === '/v1/chat/completions' && req.method === 'POST') return sendChatCompletion(res, markdown, 'stop', 150)

    const filePath = resolve(distDir, `.${path}`)
    if (path !== '/' && isFile(filePath)) return sendFile(res, filePath)
    return sendFile(res, resolve(distDir, 'index.html'))
  } catch (error) {
    if (!res.writableEnded) sendJson(res, 500, { error: String(error) })
  }
})

await new Promise((done) => server.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${server.address().port}`

const browser = await launchBrowser({ purpose: 'output previews and change review browser smoke', headless: 'new' })
const page = await browser.newPage()
const cdp = await page.createCDPSession()
await cdp.send('Page.setDownloadBehavior', { behavior: 'allow', downloadPath: downloads })
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
  if (window.top !== window) return
  if (window.sessionStorage.getItem('camelid.continuationSmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.continuationSmokeInitialized', 'true')
})

const clickText = async (selector, text) => {
  const handle = await page.waitForFunction((selector, text) => [...document.querySelectorAll(selector)].find(e => e.textContent.trim() === text), {}, selector, text)
  await handle.asElement().click()
}
try {
  await page.goto(origin + '/#chat', { waitUntil: 'networkidle0' })
  await page.type('textarea[aria-label="Message Camelid"]', 'Create a report and data files.')
  await page.waitForSelector('button[aria-label="Send message"]:not([disabled])')
  await page.click('button[aria-label="Send message"]')
  await page.waitForSelector('.message-code-card .output-actions button:not([disabled])', { timeout: 20000 })
  await page.waitForFunction(() => document.querySelectorAll('.message-code-card').length === 4)
  await page.click('.message-code-card .output-actions button')
  await page.waitForSelector('.output-modal iframe')
  assert.equal(await page.$eval('.output-modal iframe', e => e.getAttribute('sandbox')), '')
  await page.waitForFunction(() => [...document.querySelectorAll('iframe')].length > 0)
  const frame = page.frames().find(f => f !== page.mainFrame())
  await frame.waitForSelector('h1')
  assert.equal(await frame.$eval('h1', e => e.textContent), 'Preview works')
  assert.equal(await page.evaluate(() => window.__escapedPreview), undefined)
  await page.$eval('input[aria-label="Output filename"]', e => { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set; setter.call(e,'report.html'); e.dispatchEvent(new Event('input',{bubbles:true})) })
  await clickText('.output-modal button','Download file')
  for (let i=0; i<100 && !existsSync(resolve(downloads,'report.html')); i++) await new Promise(r => setTimeout(r,50))
  assert.equal(readFileSync(resolve(downloads,'report.html'),'utf8'),htmlSource)
  await new Promise(r => setTimeout(r, 300))
  await page.screenshot({path:resolve(scriptDir,'../../target/output-preview-desktop.png'),fullPage:true})
  await page.click('.output-modal button[aria-label="Close"]')
  for (const index of [1,2,3]) {
    const cards = await page.$$('.message-code-card')
    await (await cards[index].$('.output-actions button')).click()
    await page.waitForSelector('.output-modal')
    if (index === 1) assert.equal(await page.$eval('.output-table td', e=>e.textContent), 'name')
    if (index === 2) assert.match(await page.$eval('.output-preview pre',e=>e.textContent), /"ok": true/)
    if (index === 3) { await page.waitForSelector('.output-markdown h1, .output-markdown h2, .output-markdown h3'); assert.equal(await page.$eval('.output-markdown h1, .output-markdown h2, .output-markdown h3',e=>e.textContent),'Formatted report') }
    await page.click('.output-modal button[aria-label="Close"]')
  }
  await clickText('.message-code-card .output-actions button','Review file change')
  await page.waitForSelector('.change-proposal')
  await page.type('.change-folder input','/workspace')
  await page.$eval('.change-proposal input[placeholder="src/example.js"]', e => { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set; setter.call(e,'report.html'); e.dispatchEvent(new Event('input',{bubbles:true})) })
  await clickText('button','Prepare review')
  await page.waitForSelector('.change-detail')
  assert.equal(fileWrites,0,'preparation must not apply a file')
  assert.equal(currentFile,'<h1>Before</h1>')
  await clickText('button','Before')
  assert.match(await page.$eval('.change-source',e=>e.textContent), /Before/)
  await clickText('button','After')
  assert.equal(await page.$eval('.change-source',e=>e.textContent),htmlSource)
  await clickText('button','Diff summary')
  await new Promise(r => setTimeout(r, 300))
  await page.screenshot({path:resolve(scriptDir,'../../target/change-review-desktop.png'),fullPage:true})
  await clickText('button','Approve & apply')
  await page.waitForFunction(()=>document.querySelector('.change-status')?.textContent==='Applied')
  assert.equal(fileWrites,1); assert.equal(currentFile,htmlSource)
  await page.reload({waitUntil:'networkidle0'})
  await page.waitForSelector('.change-list-item'); await page.click('.change-list-item')
  await page.waitForFunction(()=>document.querySelector('.change-status')?.textContent==='Applied')
  currentFile='manual edit'
  await clickText('button','Undo this change')
  await page.waitForSelector('.changes-error')
  assert.equal(currentFile,'manual edit'); assert.equal(fileWrites,1)
  currentFile=htmlSource
  await clickText('button','Undo this change')
  await page.waitForFunction(()=>document.querySelector('.change-status')?.textContent==='Undone')
  assert.equal(currentFile,'<h1>Before</h1>'); assert.equal(fileWrites,2)
  await page.setViewport({width:390,height:844,deviceScaleFactor:1})
  await new Promise(r => setTimeout(r, 300))
  await page.screenshot({path:resolve(scriptDir,'../../target/change-review-mobile.png'),fullPage:true})
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'mobile Changes must not overflow')
  await page.goto(origin+'/?output-mobile=1#chat',{waitUntil:'networkidle0'})
  await page.waitForSelector('.message-code-card .output-actions button')
  await page.click('.message-code-card .output-actions button')
  await page.waitForSelector('.output-modal iframe')
  await new Promise(r => setTimeout(r, 300))
  await page.screenshot({path:resolve(scriptDir,'../../target/output-preview-mobile.png'),fullPage:true})
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'mobile preview must not overflow')
  assert.deepEqual(pageErrors,[])
  assert.deepEqual(externalRequests,[],'generated previews cannot make external requests')
  console.log('Output/change browser smoke passed: isolated preview, exact download, proposal, approval, restored history, undo conflict, undo, and mobile layouts.')
} catch(error) {
  await new Promise(r => setTimeout(r, 300))
  await page.screenshot({path:resolve(scriptDir,'../../target/output-changes-failure.png'),fullPage:true})
  console.error(pageErrors); throw error
} finally {
  await browser.close(); await new Promise(r=>server.close(r)); rmSync(downloads,{recursive:true,force:true})
}
