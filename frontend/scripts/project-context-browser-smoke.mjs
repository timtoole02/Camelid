#!/usr/bin/env node
/* End-to-end UI acceptance using deterministic, local-only API fixtures.
 * This verifies wiring; it makes no claim about a real model's tool quality. */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, readFileSync, statSync, mkdirSync, writeFileSync } from 'node:fs'
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
const connection = { config: { id: 'a'.repeat(32), name: 'Local test tools', transport: 'stdio', command: 'test-fixture' }, connected: true, tools: [tool] }
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
    if (path === '/api/mcp/connections') return sendJson(res, 200, { connections: [connection] })
    if (path === '/api/mcp/calls' && req.method === 'POST') {
      const body = await readJsonBody(req); prepared.push(body)
      return sendJson(res, 201, { id: 'approval-1', status: 'pending', connection_name: connection.config.name, tool: tool.name, arguments: body.arguments })
    }
    if (path.endsWith('/decision')) {
      const body = await readJsonBody(req); decisions.push(body)
      return sendJson(res, 202, { status: body.approved ? 'complete' : 'denied', result: body.approved ? { content: [{ type: 'text', text: 'Hello from MCP' }] } : null })
    }
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      const body = await readJsonBody(req); chatRequests.push(body)
      if (!body.tools?.length) return sendChatCompletion(res, 'Context received.', 'stop', 12)
      if (body.messages.some(m => m.role === 'tool')) return sendChatCompletion(res, 'The connected tool returned: Hello from MCP.', 'stop', 12)
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
  if (window.top !== window) return
  if (window.sessionStorage.getItem('camelid.continuationSmokeInitialized')) return
  window.localStorage.clear()
  window.localStorage.setItem('camelid.conversations', JSON.stringify([{ id: 'legacy-chat', title: 'Legacy chat', messages: [{ id: 'legacy-user', role: 'user', content: 'Legacy question' }, { id: 'legacy-answer', role: 'assistant', content: 'Legacy answer' }] }]))
  window.localStorage.setItem('camelid.systemPrompt', 'Global: keep answers concise.')
  window.localStorage.setItem('camelid.webResearchEnabled', 'false')
  window.sessionStorage.setItem('camelid.continuationSmokeInitialized', 'true')
})

const clickText = async (selector, text) => {
  const handle = await page.waitForFunction((selector, text) => [...document.querySelectorAll(selector)].find(e => e.textContent.trim() === text), {}, selector, text)
  await handle.asElement().click()
}
const artifactsDir = resolve(scriptDir, '../../target')
mkdirSync(artifactsDir, { recursive: true })
const projectFile = resolve(artifactsDir, 'context-brief.md')
const chatFile = resolve(artifactsDir, 'context-notes.txt')
writeFileSync(projectFile, 'Launch date: Friday.\nAudience: designers.\n')
writeFileSync(chatFile, 'Reviewer: Alex.\n')
const fill = async (selector, value) => page.$eval(selector, (element, value) => {
  const setter = Object.getOwnPropertyDescriptor(element.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, 'value').set
  setter.call(element, value)
  element.dispatchEvent(new Event('input', { bubbles: true }))
}, value)
const click = async selector => { await page.waitForSelector(selector, { visible: true }); await page.click(selector) }
const saveContext = async () => { await clickText('button', 'Save context'); await page.waitForSelector('.context-modal', { hidden: true }) }
const openContext = async () => { await click('[aria-label="Edit conversation context"]'); await page.waitForSelector('[aria-label="Conversation project"]') }
const send = async content => {
  const before = chatRequests.length
  await fill('textarea[aria-label="Message Camelid"]', content)
  await click('button[aria-label="Send message"]')
  await page.waitForFunction(before => JSON.parse(localStorage.getItem('camelid.conversations') || '[]').some(c => c.messages.some(m => m.content === 'Context received.' && !m.streaming)) && !document.querySelector('.cxcomposer__stop'), {}, before)
  assert.equal(chatRequests.length, before + 1)
  return chatRequests.at(-1)
}
const currentConversation = () => page.evaluate(() => JSON.parse(localStorage.getItem('camelid.conversations') || '[]').find(c => c.id === localStorage.getItem('camelid.selectedConversationId')))
try {
  await page.goto(`${origin}/#projects`, { waitUntil: 'networkidle0' })
  await clickText('button', 'New project')
  await fill('[aria-label="Project name"]', 'Design launch')
  await fill('[aria-label="Project instructions"]', 'Project: write for designers.')
  await (await page.$('input[aria-label="Add reference files"]')).uploadFile(projectFile)
  await page.waitForFunction(() => document.querySelector('.context-modal')?.textContent.includes('context-brief.md'))
  await clickText('button', 'Save project')
  await page.waitForSelector('[aria-label="New chat in Design launch"]')
  await page.screenshot({ path: resolve(artifactsDir, 'projects-desktop.png'), fullPage: true })
  await click('[aria-label="New chat in Design launch"]')
  assert.ok(!new URL(page.url()).hash, 'new project chat navigates out of the projects hash')
  await openContext()
  assert.equal(await page.$eval('[aria-label="Conversation project"]', element => element.selectedOptions[0].textContent), 'Design launch')
  await fill('[aria-label="Conversation instructions"]', 'Conversation: answer in Spanish.')
  await (await page.$('input[aria-label="Add reference files"]')).uploadFile(chatFile)
  await page.waitForFunction(() => document.querySelector('.context-modal')?.textContent.includes('context-notes.txt'))
  await new Promise(resolve => setTimeout(resolve, 350))
  await page.screenshot({ path: resolve(artifactsDir, 'conversation-context-desktop.png'), fullPage: true })
  await saveContext()
  await page.reload({ waitUntil: 'networkidle0' })
  await page.waitForSelector('[aria-label="Edit conversation context"]')
  assert.ok(await page.$eval('.context-trigger', element => element.textContent.includes('Design launch')), 'unsent draft context survives reload')
  const first = await send('Summarize the brief.')
  assert.equal(first.messages.length, 6)
  assert.deepEqual(first.messages.map(m => m.role), ['system', 'system', 'system', 'user', 'user', 'user'])
  assert.ok(first.messages[0].content.includes('Global:'))
  assert.ok(first.messages[1].content.includes('write for designers'))
  assert.ok(first.messages[2].content.includes('Spanish'))
  assert.ok(first.messages[3].content.includes('Friday'))
  assert.ok(first.messages[4].content.includes('Alex'))
  assert.equal(first.messages[5].content, 'Summarize the brief.')
  const original = await currentConversation()
  assert.ok(original.context.project_id)
  assert.equal(original.messages.length, 3, 'context stays out of transcript (bootstrap, question, reply)')

  await openContext()
  await page.evaluate(() => {
    for (const text of ['Use global instructions', 'Use project instructions', 'context-brief.md']) {
      const label = [...document.querySelectorAll('.context-check')].find(item => item.textContent.trim() === text)
      label.querySelector('input').click()
    }
  })
  await saveContext()
  const second = await send('What should Alex review?')
  assert.equal(second.messages.filter(m => m.role === 'system').length, 1)
  assert.ok(!JSON.stringify(second.messages).includes('Friday'))
  assert.ok(!JSON.stringify(second.messages).includes('Global:'))
  assert.ok(JSON.stringify(second.messages).includes('Reviewer: Alex'))

  // Restore inheritance, then verify MCP continuation uses one frozen prefix.
  await openContext()
  await page.evaluate(() => { for (const input of document.querySelectorAll('.context-check input')) if (!input.checked) input.click() })
  await saveContext()
  await click('.mcp-picker summary')
  await page.waitForSelector('.mcp-tool-options input:not([disabled])')
  await click('.mcp-tool-options input')
  await fill('textarea[aria-label="Message Camelid"]', 'Say hello using the echo tool.')
  await click('button[aria-label="Send message"]')
  await page.waitForSelector('.mcp-approval')
  await openContext()
  assert.equal(await page.$eval('[aria-label="Conversation instructions"]', element => element.matches(':disabled')), true, 'context is locked during a tool run')
  await clickText('.context-modal button', 'Cancel')
  await clickText('button', 'Allow once')
  await page.waitForFunction(() => !document.querySelector('.mcp-run') && document.body.textContent.includes('The connected tool returned: Hello from MCP.'))
  assert.equal(chatRequests.length, 4)
  assert.deepEqual(chatRequests[2].messages.slice(0, 5), chatRequests[3].messages.slice(0, 5), 'MCP carries the exact context prefix once')
  assert.equal(chatRequests[3].messages.filter(m => m.content?.includes?.('Reviewer: Alex')).length, 1)

  await clickText('.conversation-context button', 'Projects')
  await click('[aria-label="Edit Design launch"]')
  await fill('[aria-label="Project instructions"]', 'Project: use the updated design brief.')
  await clickText('button', 'Save project')
  await page.waitForSelector('.context-modal', { hidden: true })
  await click('.project-chats summary')
  await click('.project-chats button')
  await openContext()
  assert.ok(await page.$eval('.context-sources', element => element.textContent.includes('updated design brief')))
  await clickText('.context-modal button', 'Cancel')

  await clickText('.conversation-context button', 'Projects')
  await clickText('button', 'New project')
  await fill('[aria-label="Project name"]', 'Research notes')
  await fill('[aria-label="Project instructions"]', 'Project: focus on research methods.')
  await clickText('button', 'Save project')
  await page.waitForSelector('[aria-label="New chat in Research notes"]')
  await click('[aria-label="New chat in Research notes"]')
  const other = await send('Explain the approach.')
  assert.ok(JSON.stringify(other.messages).includes('research methods'))
  assert.ok(!JSON.stringify(other.messages).includes('Alex'))
  assert.ok(!JSON.stringify(other.messages).includes('Spanish'))
  assert.ok(!JSON.stringify(other.messages).includes('Friday'))
  assert.equal(other.messages.length, 3, 'new chat contains only global, its project, and the new question')
  await page.reload({ waitUntil: 'networkidle0' })
  await page.waitForSelector('[aria-label="Edit conversation context"]')
  assert.ok(await page.$eval('.context-trigger', element => element.textContent.includes('Research notes')))

  // Context alone can fill the model window: prevent the request before fetch.
  await openContext()
  await fill('[aria-label="Conversation instructions"]', '字'.repeat(2500))
  await saveContext()
  await fill('textarea[aria-label="Message Camelid"]', 'Hello')
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]').disabled)
  assert.ok(await page.$('.cxcomposer__budget-error'))
  assert.equal(chatRequests.length, 5)
  await openContext()
  await fill('[aria-label="Conversation instructions"]', '')
  await saveContext()

  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 })
  await new Promise(resolve => setTimeout(resolve, 350))
  await openContext()
  await new Promise(resolve => setTimeout(resolve, 350))
  await page.screenshot({ path: resolve(artifactsDir, 'conversation-context-mobile.png'), fullPage: true })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'mobile page has no horizontal overflow')
  assert.ok(await page.$eval('.context-modal', element => { const rect = element.getBoundingClientRect(); return rect.left >= 0 && rect.right <= innerWidth && rect.top >= 0 && rect.bottom <= innerHeight + 1 }), 'dialog fits phone viewport')
  await clickText('.context-modal button', 'Cancel')
  await clickText('.conversation-context button', 'Projects')
  await new Promise(resolve => setTimeout(resolve, 350))
  await page.screenshot({ path: resolve(artifactsDir, 'projects-mobile.png'), fullPage: true })
  const beforeDelete = await page.evaluate(() => JSON.parse(localStorage.getItem('camelid.conversations')).length)
  await click('[aria-label="Delete Design launch"]')
  await clickText('.cx-modal button', 'Delete project')
  await page.waitForSelector('[aria-label="Delete Design launch"]', { hidden: true })
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('camelid.conversations')).length), beforeDelete)
  await page.evaluate(id => localStorage.setItem('camelid.selectedConversationId', id), original.id)
  // A fresh navigation applies the route and selected conversation together.
  await page.goto(`${origin}/?restored=1#chat`, { waitUntil: 'networkidle0' })
  await openContext()
  assert.ok(await page.$eval('.context-form', element => element.textContent.includes('This project was removed')))
  assert.ok(!await page.$eval('.context-sources', element => element.textContent.includes('Friday')))
  await clickText('.context-modal button', 'Cancel')
  await page.evaluate(() => localStorage.setItem('camelid.selectedConversationId', 'legacy-chat'))
  await page.reload({ waitUntil: 'networkidle0' })
  await openContext()
  assert.equal(await page.$eval('[aria-label="Conversation project"]', element => element.value), '')
  assert.equal(await page.$$eval('.context-sources details', elements => elements.length), 1, 'legacy chats keep only global defaults, never an unrelated draft project')
  await clickText('.context-modal button', 'Cancel')
  assert.deepEqual(pageErrors, [])
  assert.deepEqual(externalRequests, [])
  console.log('Project context browser smoke passed: project CRUD, files, request payloads, inheritance, MCP continuity, isolation, reload, budget gate, mobile layout, and deletion.')
} catch (error) {
  await page.screenshot({ path: resolve(artifactsDir, 'project-context-browser-failure.png'), fullPage: true })
  console.error(pageErrors)
  throw error
} finally {
  await browser.close()
  await new Promise(resolve => server.close(resolve))
}
