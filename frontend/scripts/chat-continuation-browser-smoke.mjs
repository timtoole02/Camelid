#!/usr/bin/env node
/* Browser-level acceptance for continuing a length-truncated reply.
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app plus deterministic API fixtures; the browser aborts every
 * cross-origin request, so a green run cannot depend on anything external.
 *
 * The pure smoke covers the join and usage arithmetic. What only a real render
 * can prove is the transcript bookkeeping, which is where this feature would
 * fail silently:
 *
 *   - the continuation lands on the SAME assistant message, not a second bubble
 *   - the request-only instruction never enters the stored transcript
 *   - the request that continues actually carries the truncated text plus the
 *     instruction, in that order
 *   - a later ordinary turn sends ONE merged assistant message, with no
 *     "continue" turn left behind to be replayed forever
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

const PROMPT = 'Write out the deployment checklist in full.'
const FOLLOW_UP = 'Thanks, now summarize it.'
/* Ends mid-word on purpose: a rejoin that inserts a space here is visible. */
const TRUNCATED = 'Checklist step one is to verify the build. Step two is to conf'
const RESUMED = 'irm the signature. Step three is to publish.'
const MERGED = `${TRUNCATED}${RESUMED}`
const FOLLOW_UP_ANSWER = 'Three steps: build, signature, publish.'

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
  model_compatibility: ledger.model_rows.map((row) => row.contract),
}

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
        active_model_id: MODEL_FILENAME,
        active_context_length: 4096,
        max_prompt_tokens: 4096,
        max_generation_tokens: 8192,
      })
    }
    if (path === '/v1/models') {
      return sendJson(res, 200, {
        object: 'list',
        data: [{
          id: MODEL_FILENAME,
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
        id: MODEL_FILENAME,
        path: `models/${MODEL_FILENAME}`,
        gguf: { metadata: { general: { architecture: 'qwen3', file_type: 7 } } },
        tokenizer: { status: 'available' },
      })
    }
    if (path === '/api/web/research' && req.method === 'POST') {
      return sendJson(res, 200, { status: 'skipped', triggered: false, reason: 'not_needed', sources: [], warnings: [] })
    }
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      const body = await readJsonBody(req)
      chatRequests.push(body)
      const lastUser = [...(body?.messages || [])].reverse().find((message) => message?.role === 'user')
      const text = typeof lastUser?.content === 'string' ? lastUser.content : ''
      if (text.startsWith('Continue your previous reply')) return sendChatCompletion(res, RESUMED, 'stop', 12)
      if (text === FOLLOW_UP) return sendChatCompletion(res, FOLLOW_UP_ANSWER, 'stop', 9)
      // The opening turn deliberately runs out of budget.
      return sendChatCompletion(res, TRUNCATED, 'length', 16)
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

const assistantTexts = () => page.$$eval(
  'main[data-view="chat"] .cxturn--assistant .cxturn__body',
  (nodes) => nodes.map((node) => node.textContent),
)
/* The stored transcript also holds the "Conversation created." bootstrap row,
   which the chat view filters out of the rendered thread. Filter it the same
   way here so the counts below describe what the reader actually sees. */
const storedMessages = () => page.evaluate(() => {
  const conversations = JSON.parse(localStorage.getItem('camelid.conversations') || '[]')
  const selected = localStorage.getItem('camelid.selectedConversationId')
  const conversation = conversations.find((item) => item.id === selected) || conversations[0]
  return (conversation?.messages || [])
    .filter((message) => !String(message.content || '').startsWith('Conversation created.'))
    .map((message) => ({
      role: message.role,
      content: String(message.content || ''),
      finish_reason: message.finish_reason || null,
      continuation_count: message.continuation_count || 0,
      usage: message.usage || null,
    }))
})

async function sendPrompt(prompt) {
  await page.$eval('textarea[aria-label="Message Camelid"]:not([disabled])', (textarea, value) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, value)
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
  }, prompt)
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true'
  ), { timeout: 30000 })
  await page.click('button[aria-label="Send message"]')
}

async function waitForIdleAnswer(text) {
  await page.waitForFunction((expected) => (
    [...document.querySelectorAll('.cxturn--assistant .cxturn__body')].some((node) => node.textContent.includes(expected))
  ), { timeout: 30000 }, text)
  await page.waitForFunction(() => (
    !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming')
  ), { timeout: 30000 })
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') !== null
  ), { timeout: 30000 })

  /* ---- 1. a reply that ran out of budget offers Continue ---------------- */
  await sendPrompt(PROMPT)
  await waitForIdleAnswer(TRUNCATED)
  await page.waitForSelector('.cxturn__warning-action', { timeout: 30000 })
  const warningText = await page.$eval('.cxturn__warning', (node) => node.textContent)
  assert.match(warningText, /Stopped at the response budget/, 'the budget stop is explained')
  assert.match(warningText, /Continue/, 'Continue is offered next to that explanation')
  assert.equal((await assistantTexts()).length, 1, 'one reply so far')

  /* ---- 2. continuing merges into the SAME message ---------------------- */
  await page.click('.cxturn__warning-action')
  await waitForIdleAnswer(MERGED)

  const afterTexts = await assistantTexts()
  assert.equal(afterTexts.length, 1, 'continuing must not open a second assistant bubble')
  assert.match(afterTexts[0], /Step two is to confirm the signature/, 'the mid-word resume rejoins without a stray space')

  const stored = await storedMessages()
  const assistantMessages = stored.filter((message) => message.role === 'assistant')
  assert.equal(assistantMessages.length, 1, 'exactly one assistant message is stored')
  assert.equal(assistantMessages[0].content, MERGED, 'the stored reply is the merged text')
  assert.equal(assistantMessages[0].finish_reason, 'stop', 'the merged reply carries the resumed finish reason')
  assert.equal(assistantMessages[0].continuation_count, 1, 'the resume is recorded on the message')
  assert.equal(
    assistantMessages[0].usage.completion_tokens,
    28,
    'output tokens accumulate across the two requests (16 + 12)',
  )

  const userMessages = stored.filter((message) => message.role === 'user')
  assert.equal(userMessages.length, 1, 'the continuation instruction must not be stored as a user turn')
  assert.equal(userMessages[0].content, PROMPT, 'the only stored user turn is the one that was typed')

  const visibleTranscript = await page.$eval('main[data-view="chat"]', (node) => node.textContent)
  assert.doesNotMatch(
    visibleTranscript,
    /Continue your previous reply/,
    'the request-only instruction must never be visible in the transcript',
  )

  /* ---- 3. the continuation request carried the right payload ------------ */
  assert.equal(chatRequests.length, 2, 'continuing issues exactly one more request')
  const continuation = chatRequests[1]
  const roles = continuation.messages.map((message) => message.role)
  assert.equal(roles.at(-1), 'user', 'the continuation instruction is the final turn')
  assert.equal(roles.at(-2), 'assistant', 'the truncated reply is sent immediately before it')
  assert.equal(
    continuation.messages.at(-2).content,
    TRUNCATED,
    'the model is shown exactly the text it must resume from',
  )
  assert.match(
    continuation.messages.at(-1).content,
    /Do not repeat any text you already wrote/,
    'the instruction tells the model not to restate',
  )

  /* ---- 4. Continue is withdrawn once the reply is complete -------------- */
  assert.equal(
    await page.$('.cxturn__warning-action'),
    null,
    'a completed reply no longer offers Continue',
  )
  const footer = await page.$eval('.cxturn--assistant .cxturn__meta', (node) => node.textContent)
  assert.match(footer, /continued/, 'the footer discloses that the reply was resumed')

  /* ---- 5. the next ordinary turn sends one clean assistant message ------ */
  await sendPrompt(FOLLOW_UP)
  await waitForIdleAnswer(FOLLOW_UP_ANSWER)
  const followUp = chatRequests[2]
  const assistantTurnsSent = followUp.messages.filter((message) => message.role === 'assistant')
  assert.equal(assistantTurnsSent.length, 1, 'the follow-up sends one merged assistant turn, not two fragments')
  assert.equal(assistantTurnsSent[0].content, MERGED, 'and it is the merged text')
  assert.equal(
    followUp.messages.filter((message) => (
      typeof message.content === 'string' && message.content.startsWith('Continue your previous reply')
    )).length,
    0,
    'the continuation instruction must not be replayed on later turns',
  )

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')

  console.log('chat continuation browser smoke passed')
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
