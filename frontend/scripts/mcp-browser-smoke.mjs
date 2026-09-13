#!/usr/bin/env node
/* End-to-end UI acceptance using deterministic, local-only API fixtures.
 * This verifies wiring; it makes no claim about a real model's tool quality. */
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
const readTool = { ...tool, key: 'mcp_fixture_read', name: 'read_file', description: 'Read a project file.' }
const connection = { config: { id: 'a'.repeat(32), name: 'Local test tools', transport: 'stdio', command: 'test-fixture' }, connected: true, tools: [tool, readTool] }
const largeConnection = { config: { id: 'b'.repeat(32), name: 'Repository tools', transport: 'stdio', command: 'test-fixture' }, connected: true,
  tools: Array.from({ length: 17 }, (_, index) => ({ ...tool, key: 'mcp_repository_' + index, name: 'repository_tool_' + index })) }
const remoteTool = { ...tool, key: 'mcp_remote_search', name: 'search_docs' }
const remoteConnection = { config: { id: 'c'.repeat(32), name: 'Remote docs', transport: 'http', url: 'https://docs.example/mcp' }, connected: false, tools: [] }
const connections = [connection, largeConnection, remoteConnection]
const connectionRequests = [], cancellations = []
const decisions = [], prepared = []
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
    if (path === '/api/mcp/connections' && req.method === 'POST') {
      const config = { ...await readJsonBody(req), id: 'd'.repeat(32) }
      connectionRequests.push(config)
      connections.push({ config, connected: false, tools: [] })
      return sendJson(res, 201, { saved: true })
    }
    if (path === '/api/mcp/connections') return sendJson(res, 200, { connections })
    if (path.startsWith('/api/mcp/connections/')) {
      const found = connections.find(item => path.includes(item.config.id))
      if (found && path.endsWith('/connect')) {
        found.connected = true
        if (found === remoteConnection) found.tools = [remoteTool]
      } else if (found && path.endsWith('/disconnect')) found.connected = false
      return sendJson(res, 200, { ok: true })
    }
    if (path === '/api/mcp/calls' && req.method === 'POST') {
      const body = await readJsonBody(req); prepared.push(body)
      return sendJson(res, 201, { id: 'approval-1', status: 'pending', connection_name: connection.config.name, tool: tool.name, arguments: body.arguments })
    }
    if (path.endsWith('/decision')) {
      const body = await readJsonBody(req); decisions.push(body)
      return sendJson(res, 202, { status: body.approved ? 'complete' : 'denied', result: body.approved ? { content: [{ type: 'text', text: 'Hello from MCP' }] } : null })
    }
    if (path.startsWith('/api/mcp/calls/') && req.method === 'DELETE') {
      cancellations.push(path)
      return sendJson(res, 200, { status: 'cancelled' })
    }
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      const body = await readJsonBody(req); chatRequests.push(body)
      if (body.messages.at(-1).role === 'tool') return sendChatCompletion(res, 'The connected tool returned: Hello from MCP.', 'stop', 12)
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      const data = { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: tool.key, arguments: '{"text":"hello"}' } }] } }] }
      res.write(`data: ${JSON.stringify(data)}\n\n`)
      res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'tool_calls' }], usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 } })}\n\n`)
      return res.end('data: [DONE]\n\n')
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

const browser = await launchBrowser({ purpose: 'the chat continuation browser smoke', headless: 'new' })
const page = await browser.newPage()
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
  if (window.sessionStorage.getItem('camelid.continuationSmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.continuationSmokeInitialized', 'true')
})

const click = async selector => (await page.waitForSelector(selector, { visible: true })).click()
const clickText = async (selector, text) => {
  const handle = await page.waitForFunction((selector, text) => [...document.querySelectorAll(selector)].find(e => e.textContent.trim() === text), {}, selector, text)
  await handle.asElement().click()
}
try {
  await page.goto(`${origin}/#connections`, { waitUntil: 'networkidle0' })
  await page.waitForSelector('.mcp-connection')
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-connections-desktop.png'), fullPage: true })
  await clickText('button', 'Add server')
  await page.waitForSelector('.mcp-form')
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-add-desktop.png'), fullPage: true })
  await clickText('button', 'Cancel')
  await clickText('button', 'Chat')
  await page.waitForSelector('.mcp-trigger')
  await click('.mcp-trigger')
  await page.waitForSelector('.mcp-tool-options input:not([disabled])', { timeout: 20000 })
  await click('.mcp-tool-options input')
  assert.equal(await page.$eval('.mcp-trigger', element => Boolean(element.closest('.cxcomposer__toolbar'))), true, 'Tools belongs inside the composer toolbar')
  assert.equal(await page.$eval('.mcp-picker', element => element.parentElement === document.body), true, 'picker escapes scrolling composer ancestors')
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-picker-desktop.png'), fullPage: true })
  await click('[aria-label="Select all tools from Repository tools"]')
  await page.waitForFunction(() => document.querySelector('.mcp-picker').textContent.includes('Choose individual tools'))
  assert.equal(await page.$eval('.mcp-trigger__count', element => element.textContent), '1', 'oversized group is rejected atomically')
  await clickText('button', 'Clear selection')
  await clickText('.mcp-group-toggle', 'Repository tools')
  for (let index = 0; index < 16; index += 1) await click('[data-tool-key="mcp_repository_' + index + '"]')
  assert.equal(await page.$eval('.mcp-trigger__count', element => element.textContent), '16')
  assert.equal(await page.$eval('[data-tool-key="mcp_repository_16"]', element => element.disabled), true, 'the 17th individual selection is blocked')
  await clickText('button', 'Clear selection')
  await clickText('.mcp-group-toggle', 'Local test tools')
  await click('[data-tool-key="mcp_fixture_echo"]')
  await page.type('[aria-label="Search tools"]', 'read a project')
  await page.waitForFunction(() => document.querySelectorAll('.mcp-tool-options input').length === 1)
  assert.equal(await page.$eval('.mcp-tool-options input', element => element.dataset.toolKey), readTool.key)
  await click('.mcp-tool-options input')
  await page.$eval('[aria-label="Search tools"]', input => { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; setter.call(input, ''); input.dispatchEvent(new Event('input', { bubbles: true })) })
  await click('[aria-label="Save selection as a tool set"]')
  await page.type('[aria-label="Tool set name"]', 'Code review')
  await clickText('button', 'Save set')
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('camelid.mcpToolSets') || '[]').length === 1)
  const setId = await page.evaluate(() => JSON.parse(localStorage.getItem('camelid.mcpToolSets'))[0].id)
  await clickText('button', 'Clear selection')
  await page.select('[aria-label="Tool set"]', setId)
  assert.equal(await page.$eval('.mcp-trigger__count', element => element.textContent), '2')
  await page.keyboard.press('Escape')
  assert.equal(await page.$('.mcp-picker'), null)
  assert.equal(await page.$eval('.mcp-trigger', element => element === document.activeElement), true, 'Escape restores trigger focus')
  await page.reload({ waitUntil: 'networkidle0' })
  await click('.mcp-trigger')
  await page.waitForSelector('[aria-label="Tool set"] option[value="' + setId + '"]')
  await page.select('[aria-label="Tool set"]', setId)
  assert.equal(await page.$eval('.mcp-trigger__count', element => element.textContent), '2', 'saved sets restore exact tools after reload')
  await click('.mcp-manual summary')
  await click('.mcp-manual__toggle input')
  assert.equal(await page.$('.mcp-tool-options input:checked'), null, 'manual definitions clear connected selections')
  await page.waitForSelector('textarea[aria-label="Tool definitions"]')
  await click('[data-tool-key="mcp_fixture_echo"]')
  assert.equal(await page.$eval('.mcp-manual__toggle input', element => element.checked), false, 'connected selection turns manual definitions off')
  // A storage refusal must not produce a false saved confirmation.
  await page.evaluate(() => {
    window.mcpOriginalSetItem = Storage.prototype.setItem
    Storage.prototype.setItem = function (key, value) {
      if (key === 'camelid.mcpToolSets') throw new DOMException('Quota exceeded', 'QuotaExceededError')
      return window.mcpOriginalSetItem.call(this, key, value)
    }
  })
  await click('[aria-label="Save selection as a tool set"]')
  await page.type('[aria-label="Tool set name"]', 'Unsaved')
  await clickText('button', 'Save set')
  await page.waitForFunction(() => document.querySelector('.mcp-picker').textContent.includes('Could not save tool sets'))
  await page.evaluate(() => { Storage.prototype.setItem = window.mcpOriginalSetItem })
  await clickText('.mcp-save-set button', 'Cancel')
  await clickText('.mcp-picker button', 'Connect')
  await page.waitForFunction(() => document.querySelector('.mcp-picker').textContent.includes('Remote docs') && !document.querySelector('.mcp-disconnected'))
  await page.type('[aria-label="Search tools"]', 'search_docs')
  await click('[data-tool-key="mcp_remote_search"]')
  await clickText('.mcp-picker button', 'Manage connections')
  await click('[aria-label="Details for Remote docs"]')
  await clickText('button', 'Disconnect')
  await page.waitForFunction(() => [...document.querySelectorAll('.mcp-connection')].find(element => element.textContent.includes('Remote docs')).textContent.includes('Disconnected'))
  await clickText('button', 'Chat')
  await click('.mcp-trigger')
  await page.waitForFunction(() => document.querySelector('.mcp-picker').textContent.includes('selected tool is unavailable'))
  await clickText('button', 'Remove unavailable')
  assert.equal(await page.$eval('.mcp-trigger__count', element => element.textContent), '1')
  await click('textarea[aria-label="Message Camelid"]')
  assert.equal(await page.$('.mcp-picker'), null, 'clicking back into the composer dismisses the picker')
  await page.type('textarea[aria-label="Message Camelid"]', 'Say hello using the echo tool.')
  await click('button[aria-label="Send message"]')
  await page.waitForSelector('.mcp-approval', { timeout: 20000 })
  assert.equal(decisions.length, 0, 'execution requires a human decision')
  assert.equal(prepared.length, 1)
  assert.equal(await page.$eval('.mcp-approval', element => Boolean(element.closest('.cxchat__thread'))), true, 'approval is part of the conversation')
  assert.equal((await page.$$('.mcp-trigger')).length, 1, 'starting a conversation keeps exactly one tool picker')
  assert.equal(await page.$eval('.mcp-trigger', element => element.disabled), true, 'tool selection is locked throughout approval')
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-approval-desktop.png'), fullPage: true })
  await clickText('button', 'Allow once')
  await page.waitForFunction(() => document.body.textContent.includes('The connected tool returned: Hello from MCP.'), { timeout: 20000 })
  await page.waitForFunction(() => !document.querySelector('.mcp-run'), { timeout: 20000 })
  assert.equal(chatRequests.length, 2)
  assert.equal(decisions.length, 1)
  assert.equal(await page.$('.mcp-picker'), null, 'completing a turn never reopens the picker')
  const payload = chatRequests[1]
  assert.equal(payload.messages.at(-1).role, 'tool')
  assert.equal(payload.messages.at(-1).tool_call_id, 'call_1')
  assert.equal(payload.messages.at(-2).tool_calls[0].function.name, tool.key)
  assert.equal(payload.messages.filter(m => m.role === 'user').length, 1, 'continuation does not duplicate the prompt')
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('camelid.conversations')))
  assert.ok(stored.some(c => c.mcp_tools.includes('mcp_fixture_echo') && c.messages.some(m => m.role === 'tool')))
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-result-desktop.png'), fullPage: true })
  await page.type('textarea[aria-label="Message Camelid"]', 'Run the tool again, but let me decline.')
  await click('button[aria-label="Send message"]')
  await page.waitForSelector('.mcp-approval', { timeout: 20000 })
  await clickText('button', 'New chat')
  await page.waitForSelector('.camelid-notice-slot .mcp-approval')
  assert.equal(await page.$('.cxchat__thread .mcp-approval'), null, 'another conversation never inherits the approval card')
  assert.equal((await page.$$('.mcp-approval')).length, 1, 'approval remains reachable exactly once after switching conversations')
  await clickText('#camelid-sidebar button', 'Say hello using the echo tool.')
  await page.waitForSelector('.cxchat__thread .mcp-approval')
  assert.equal(await page.$('.camelid-notice-slot .mcp-approval'), null, 'returning restores the inline approval')
  await clickText('button', 'Connections')
  await page.waitForSelector('.mcp-view')
  await page.waitForSelector('.camelid-notice-slot .mcp-approval')
  await clickText('button', 'Chat')
  await page.waitForSelector('.cxchat__thread .mcp-approval')
  await clickText('button', 'Deny')
  await page.waitForFunction(() => !document.querySelector('.mcp-run'), { timeout: 20000 })
  assert.equal(decisions.at(-1).approved, false)
  assert.match(chatRequests.at(-1).messages.at(-1).content, /denied/)
  await page.type('textarea[aria-label="Message Camelid"]', 'Prepare another tool request.')
  await click('button[aria-label="Send message"]')
  await page.waitForSelector('.mcp-approval', { timeout: 20000 })
  await clickText('.mcp-run button', 'Stop')
  await page.waitForFunction(() => !document.querySelector('.mcp-run'), { timeout: 20000 })
  assert.equal(decisions.length, 2, 'Stop must not submit an approval')
  assert.ok(cancellations.length)
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 })
  await page.goto(`${origin}/?mcp-mobile=1#connections`, { waitUntil: 'networkidle0' })
  await page.waitForSelector('.mcp-connection')
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-connections-mobile.png'), fullPage: true })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'mobile layout must not overflow')
  await clickText('button', 'Add server')
  await page.waitForSelector('.mcp-form')
  await page.type('.mcp-form input', 'Design docs')
  await page.select('.mcp-form select', 'http')
  await page.type('.mcp-form input[type="url"]', 'https://docs.example/mcp')
  await click('.mcp-form__advanced summary')
  await page.type('.mcp-form__advanced input', 'MY_MCP_TOKEN')
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-add-mobile.png'), fullPage: true })
  await clickText('button', 'Save server')
  await page.waitForSelector('.mcp-connection-modal', { hidden: true })
  assert.equal(connectionRequests.length, 1)
  assert.equal(connectionRequests[0].bearer_env, 'MY_MCP_TOKEN')
  assert.equal(connections.at(-1).connected, false, 'saving never connects or launches a server')
  await page.goto(origin + '/?picker-mobile=1#chat', { waitUntil: 'networkidle0' })
  await click('.mcp-trigger')
  await page.waitForSelector('.mcp-picker.is-mobile')
  assert.equal(await page.$eval('.mcp-picker', element => element.getAttribute('aria-modal')), 'true')
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-picker-mobile.png'), fullPage: true })
  await clickText('.mcp-picker button', 'Manage connections')
  await page.goto(origin + '/?picker-mobile=2#chat', { waitUntil: 'networkidle0' })
  await click('.mcp-trigger')
  await page.$eval('.mcp-picker__footer button', element => element.focus())
  await page.keyboard.press('Tab')
  assert.equal(await page.$eval('.mcp-picker', element => element.contains(document.activeElement)), true, 'mobile dialog traps keyboard focus')
  for (const [width, height] of [[320, 640], [390, 420], [800, 400], [1280, 900]]) {
    await page.setViewport({ width, height, deviceScaleFactor: 1 })
    await page.waitForFunction(() => {
      const panel = document.querySelector('.mcp-picker')?.getBoundingClientRect()
      return panel && panel.left >= 0 && panel.top >= 0 && panel.right <= innerWidth + 1 && panel.bottom <= innerHeight + 1
    })
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'picker never expands page width')
  }
  await page.evaluate(() => { document.documentElement.dataset.theme = 'light' })
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-picker-light.png'), fullPage: true })
  await page.keyboard.press('Escape')
  await page.waitForSelector('.mcp-picker', { hidden: true })
  assert.deepEqual(pageErrors, [])
  assert.deepEqual(externalRequests, [])
  console.log('MCP browser smoke passed: picker limits/search, saved sets/storage failure, manual mode, reconnect, allow/deny/stop, conversation history, server setup, keyboard dismissal, and responsive layout.')
} catch (error) {
  await page.screenshot({ path: resolve(scriptDir, '../../target/mcp-browser-failure.png'), fullPage: true })
  console.error(pageErrors)
  throw error
} finally {
  await browser.close()
  await new Promise(resolve => server.close(resolve))
}
