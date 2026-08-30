#!/usr/bin/env node
/* Browser-level Web Auto acceptance with no model process and no public web.
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app and deterministic API fixtures. The browser aborts every request
 * to a different origin, so a green run cannot depend on GitHub or a search
 * provider being reachable.
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
const BRANCH_URL = 'https://github.com/acme/widgets/tree/feature/web-research'
const BRANCH_CHUNK_URL = 'https://github.com/acme/widgets/blob/feature/web-research/README.md'
const BLOB_URL = 'https://github.com/acme/widgets/blob/feature/web-research/src/api/mod.rs'
const OFF_PROMPT = 'Summarize https://example.test/web-auto-must-stay-off'
const LOCAL_PROMPT = 'Rewrite this sentence in a friendlier tone.'
const PRIVACY_PROMPT = 'Stay offline and do not use online sources or network access; compare the current versions mentioned at https://example.test/private.'
const SECOND_CHAT_PROMPT = 'Explain local conversation isolation in one sentence.'
const OVERFULL_PROMPT = 'overfull active context '.repeat(5_000)
const LINKED_PROMPT = `Read ${BRANCH_URL} and ${BLOB_URL}`
const STOP_PROMPT = 'Search the web for the deterministic stop fixture'
const HELD_STREAM_PROMPT = 'Explain deterministic stream pacing in one sentence.'
const HELD_STREAM_TAIL = 'HELD_STREAM_UNREVEALED_TAIL_SENTINEL'
const HELD_STREAM_CONTENT = `Held stream prefix is visible. The paced remainder includes ${HELD_STREAM_TAIL}.`
const HELD_STREAM_PREFIX = [...HELD_STREAM_CONTENT].slice(0, 32).join('')
const PARTIAL_WARNING = 'One supplemental result timed out.'
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
const partialResearch = {
  status: 'partial',
  triggered: true,
  reason: 'linked_urls',
  query: LINKED_PROMPT,
  sources: [
    {
      id: 1,
      title: 'Feature branch',
      url: BRANCH_URL,
      excerpt: 'BRANCH_EXCERPT_SECRET branch evidence',
      chunks: [{ path: 'README.md', url: BRANCH_CHUNK_URL, text: 'BRANCH_EXCERPT_SECRET branch evidence for Web Auto.' }],
    },
    {
      id: 2,
      title: 'API module blob',
      url: BLOB_URL,
      excerpt: 'BLOB_EXCERPT_SECRET blob evidence',
      chunks: [{ path: 'src/api/mod.rs', url: BLOB_URL, text: 'BLOB_EXCERPT_SECRET blob evidence for Web Auto.' }],
    },
  ],
  warnings: [PARTIAL_WARNING],
}

const researchRequests = []
const chatRequests = []
const heldChatStreams = []
const researchScenarios = []
const pageErrors = []
const externalRequests = []

function sendJson(res, status, body) {
  const payload = JSON.stringify(body)
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload),
  })
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

function sendChatCompletion(res, content) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
  })
  res.socket?.setNoDelay(true)
  res.flushHeaders()
  const frame = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)
  frame({ choices: [{ delta: { role: 'assistant' } }] })
  frame({ choices: [{ delta: { content } }] })
  frame({
    choices: [{ delta: {}, finish_reason: 'stop' }],
    usage: { prompt_tokens: 32, completion_tokens: 4, total_tokens: 36 },
  })
  res.write('data: [DONE]\n\n')
  res.end()
}

function sendHeldChatCompletion(res, content) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
  })
  res.socket?.setNoDelay(true)
  res.flushHeaders()
  const frame = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)

  let settle = () => {}
  const done = new Promise((resolveDone) => { settle = resolveDone })
  const entry = {
    aborted: false,
    released: false,
    release: () => {},
  }
  entry.release = () => {
    if (entry.released) return
    entry.released = true
    if (!entry.aborted && !res.destroyed && !res.writableEnded) {
      frame({
        choices: [{ delta: {}, finish_reason: 'stop' }],
        usage: { prompt_tokens: 32, completion_tokens: 1, total_tokens: 33 },
      })
      res.write('data: [DONE]\n\n')
      res.end()
    }
    settle()
  }
  res.once('close', () => {
    if (res.writableEnded || entry.released) return
    entry.aborted = true
    entry.release()
  })
  heldChatStreams.push(entry)

  // One deliberately large content event, then no finish/usage/DONE until
  // the test releases it. This proves the UI's first-prefix path independently
  // from its terminal stream flush.
  frame({ choices: [{ delta: { role: 'assistant' } }] })
  frame({ choices: [{ delta: { content } }] })
  return done
}

function isFile(path) {
  try {
    return statSync(path).isFile()
  } catch {
    return false
  }
}

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true,
        engine: 'camelid',
        api_surface: 'full',
        version: 'web-research-browser-smoke',
        build: 'web-research-browser-smoke',
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
      const body = await readJsonBody(req)
      const scenario = researchScenarios.shift() || {
        hold: false,
        result: { status: 'skipped', triggered: false, reason: 'not_needed', sources: [], warnings: [] },
      }
      const entry = {
        body,
        scenario,
        aborted: false,
        released: false,
        release: () => {},
      }
      researchRequests.push(entry)

      let releaseWait = () => {}
      const held = new Promise((done) => { releaseWait = done })
      entry.release = () => {
        if (entry.released) return
        entry.released = true
        releaseWait()
      }
      const markAborted = () => {
        if (res.writableEnded) return
        entry.aborted = true
        entry.release()
      }
      req.once('aborted', markAborted)
      res.once('close', markAborted)

      if (scenario.hold) await held
      if (entry.aborted || res.destroyed) return
      return sendJson(res, 200, scenario.result)
    }
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      const body = await readJsonBody(req)
      chatRequests.push(body)
      const prompt = [...(body?.messages || [])].reverse().find((message) => message?.role === 'user')?.content
      const content = prompt === OFF_PROMPT
        ? 'Web Off answer.'
        : prompt === LOCAL_PROMPT
          ? 'Local-only answer.'
          : prompt === PRIVACY_PROMPT
            ? 'Privacy-local answer.'
            : prompt === SECOND_CHAT_PROMPT
              ? 'Second chat answer.'
              : prompt === HELD_STREAM_PROMPT
                ? HELD_STREAM_CONTENT
          : 'Grounded branch answer.'
      if (prompt === HELD_STREAM_PROMPT) return sendHeldChatCompletion(res, content)
      return sendChatCompletion(res, content)
    }
    if (path === '/api/telemetry/stream') {
      res.writeHead(204)
      return res.end()
    }
    if (path.startsWith('/api/') || path.startsWith('/v1/')) {
      return sendJson(res, 404, { error: { code: 'not_found', message: 'not found in browser smoke' } })
    }

    const relative = path === '/' ? 'index.html' : decodeURIComponent(path.replace(/^\//, ''))
    const filePath = resolve(distDir, relative)
    if ((filePath === distDir || filePath.startsWith(`${distDir}/`)) && isFile(filePath)) return sendFile(res, filePath)
    return sendFile(res, resolve(distDir, 'index.html'))
  } catch (error) {
    if (!res.headersSent) return sendJson(res, 500, { error: { message: error.message } })
    res.destroy(error)
  }
})

async function waitUntil(predicate, label, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await predicate()) return
    await new Promise((done) => setTimeout(done, 20))
  }
  throw new Error(`timed out waiting for ${label}`)
}

await new Promise((done) => server.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${server.address().port}`
const browser = await launchBrowser({ purpose: 'the Web Auto browser smoke', headless: 'new' })
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
    // Record and abort malformed or otherwise non-local requests below.
  }
  externalRequests.push(url)
  return request.abort()
})
await page.evaluateOnNewDocument(() => {
  if (window.sessionStorage.getItem('camelid.webResearchBrowserSmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.webResearchBrowserSmokeInitialized', 'true')
})

async function waitForComposer() {
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') !== null
  ), { timeout: 30000 })
}

async function setComposerText(prompt) {
  await page.$eval('textarea[aria-label="Message Camelid"]:not([disabled])', (textarea, value) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, value)
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
  }, prompt)
}

async function sendPrompt(prompt) {
  await setComposerText(prompt)
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true'
  ), { timeout: 30000 })
  await page.click('button[aria-label="Send message"]')
}

async function waitForAnswer(text) {
  await page.waitForFunction((expected) => (
    [...document.querySelectorAll('.cxturn--assistant .cxturn__body')]
      .some((node) => node.textContent.includes(expected))
  ), { timeout: 30000 }, text)
  await page.waitForFunction(() => (
    !document.querySelector('.cxcomposer__stop')
      && !document.querySelector('.cxturn--assistant.is-streaming')
  ), { timeout: 30000 })
}

async function selectConversationByTitle(titleFragment, expectedAnswer) {
  const result = await page.$$eval('.rail-convo__main', (items, fragment) => {
    const item = items.find((candidate) => candidate.textContent.includes(fragment))
    item?.click()
    return { clicked: Boolean(item), titles: items.map((candidate) => candidate.textContent) }
  }, titleFragment)
  assert.equal(result.clicked, true, `missing conversation titled with ${titleFragment}; found ${result.titles.join(' | ')}`)
  await page.waitForFunction((answer) => (
    [...document.querySelectorAll('main[data-view="chat"] .cxturn__body')]
      .some((node) => node.textContent.includes(answer))
  ), { timeout: 30000 }, expectedAnswer)
}

let animationFramesHeld = false
async function holdAnimationFrames() {
  await page.evaluate(() => {
    if (window.__camelidHeldAnimationFrames) throw new Error('animation frames are already held')
    const callbacks = new Map()
    let nextId = 1
    window.__camelidHeldAnimationFrames = {
      callbacks,
      request: window.requestAnimationFrame,
      cancel: window.cancelAnimationFrame,
    }
    window.requestAnimationFrame = (callback) => {
      const id = nextId
      nextId += 1
      callbacks.set(id, callback)
      return id
    }
    window.cancelAnimationFrame = (id) => callbacks.delete(id)
  })
  animationFramesHeld = true
}

async function restoreAnimationFrames() {
  if (!animationFramesHeld) return
  animationFramesHeld = false
  await page.evaluate(() => {
    const held = window.__camelidHeldAnimationFrames
    if (!held) return
    held.callbacks.clear()
    window.requestAnimationFrame = held.request
    window.cancelAnimationFrame = held.cancel
    delete window.__camelidHeldAnimationFrames
  })
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await waitForComposer()

  // Web Off is a hard zero-request mode, even when the prompt contains a URL.
  await page.waitForSelector('button[aria-label="Turn off automatic web research"]')
  await page.click('button[aria-label="Turn off automatic web research"]')
  await page.waitForSelector('button[aria-label="Turn on automatic web research"]')
  const researchCountBeforeOffSend = researchRequests.length
  const chatCountBeforeOffSend = chatRequests.length
  await sendPrompt(OFF_PROMPT)
  await waitForAnswer('Web Off answer.')
  assert.equal(researchRequests.length, researchCountBeforeOffSend, 'Web Off must make no research request')
  assert.equal(chatRequests.length, chatCountBeforeOffSend + 1, 'Web Off should still make one ordinary chat request')
  assert.equal(chatRequests.at(-1).max_tokens, 8192, 'the clean browser profile must retain Camelid\'s 8192-token requested default')
  assert.equal(
    chatRequests.at(-1).messages.some((message) => String(message?.content || '').includes('UNTRUSTED EXTERNAL DATA')),
    false,
    'Web Off must not inject a web evidence message',
  )

  // Hold the fixture so progress is observed before either sources or answer.
  await page.click('button[aria-label="Turn on automatic web research"]')
  await page.waitForSelector('button[aria-label="Turn off automatic web research"]')
  const researchCountBeforeLocalSend = researchRequests.length
  const chatCountBeforeLocalSend = chatRequests.length
  await sendPrompt(LOCAL_PROMPT)
  await waitForAnswer('Local-only answer.')
  assert.equal(researchRequests.length, researchCountBeforeLocalSend, 'Web Auto must skip its endpoint for a definite local-only prompt')
  assert.equal(chatRequests.length, chatCountBeforeLocalSend + 1, 'a local-only prompt should still use ordinary chat')

  // Freeze every app-scheduled animation frame, then hold a single large SSE
  // content event open. The bounded prefix and live out count must render from
  // the immediate path; no pacing callback or terminal stream flush can help.
  await setComposerText(HELD_STREAM_PROMPT)
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true'
  ), { timeout: 30000 })
  const researchCountBeforeHeldStream = researchRequests.length
  const chatCountBeforeHeldStream = chatRequests.length
  const heldStreamCountBeforeSend = heldChatStreams.length
  await holdAnimationFrames()
  await page.click('button[aria-label="Send message"]')
  await waitUntil(
    () => heldChatStreams.length === heldStreamCountBeforeSend + 1,
    'the held SSE content event',
  )
  const heldStream = heldChatStreams.at(-1)
  await page.waitForFunction((expectedPrefix) => {
    const turns = [...document.querySelectorAll('main[data-view="chat"] .cxturn--assistant')]
    const turn = turns.at(-1)
    const text = turn?.querySelector('.message-markdown')?.textContent || ''
    const usage = turn?.querySelector('.cxturn__meta-usage')?.textContent || ''
    const output = Number(usage.match(/out\s+(\d+)/)?.[1] || 0)
    return turn?.classList.contains('is-streaming') && text === expectedPrefix && output > 0
  }, { polling: 20, timeout: 30000 }, HELD_STREAM_PREFIX)
  const heldRender = await page.evaluate((tail) => {
    const turns = [...document.querySelectorAll('main[data-view="chat"] .cxturn--assistant')]
    const turn = turns.at(-1)
    const text = turn?.querySelector('.message-markdown')?.textContent || ''
    const usage = turn?.querySelector('.cxturn__meta-usage')?.textContent || ''
    return {
      callbackCount: window.__camelidHeldAnimationFrames?.callbacks?.size || 0,
      output: Number(usage.match(/out\s+(\d+)/)?.[1] || 0),
      streaming: turn?.classList.contains('is-streaming') || false,
      tailVisible: text.includes(tail),
      text,
    }
  }, HELD_STREAM_TAIL)
  assert.equal(heldStream.released, false, 'the SSE fixture must remain incomplete during the held-state assertions')
  assert.equal(heldRender.streaming, true, 'the bounded prefix must render before stream completion')
  assert.equal(heldRender.text, HELD_STREAM_PREFIX, 'only the bounded first prefix may render synchronously')
  assert.equal(heldRender.tailVisible, false, 'the unrevealed first-chunk tail must remain absent while animation frames are held')
  assert.ok(heldRender.output > 0, 'live usage must advance above out 0 before animation frames or stream completion')
  assert.ok(heldRender.callbackCount > 0, 'the unrevealed tail must be waiting on a held pacing callback')
  assert.equal(researchRequests.length, researchCountBeforeHeldStream, 'the local held-stream prompt must not invoke Web Auto')
  assert.equal(chatRequests.length, chatCountBeforeHeldStream + 1, 'the held fixture must use one ordinary chat request')

  await restoreAnimationFrames()
  heldStream.release()
  await waitForAnswer(HELD_STREAM_CONTENT)
  const completedHeldText = await page.$$eval(
    'main[data-view="chat"] .cxturn--assistant .message-markdown',
    (messages) => messages.at(-1)?.textContent || '',
  )
  assert.equal(completedHeldText, HELD_STREAM_CONTENT, 'releasing the stream must render the complete byte-identical answer')

  // Explicit privacy language wins over every competing trigger in the same
  // prompt: a URL, current-version wording, and phrases about online/network.
  const researchCountBeforePrivacySend = researchRequests.length
  const chatCountBeforePrivacySend = chatRequests.length
  await sendPrompt(PRIVACY_PROMPT)
  await waitForAnswer('Privacy-local answer.')
  assert.equal(researchRequests.length, researchCountBeforePrivacySend, 'privacy veto must suppress Web Auto despite competing triggers')
  assert.equal(chatRequests.length, chatCountBeforePrivacySend + 1, 'privacy-vetoed prompts must still use ordinary local chat')
  assert.equal(
    chatRequests.at(-1).messages.some((message) => String(message?.content || '').includes('UNTRUSTED EXTERNAL DATA')),
    false,
    'privacy-vetoed chat must not receive injected web evidence',
  )

  // The model advertises a 32K training context but is actively running at 4K.
  // Enter reaches the hook even while the button is disabled by the composer
  // estimate, proving the send path itself fails closed before either endpoint.
  await setComposerText(OVERFULL_PROMPT)
  await page.waitForFunction(() => (
    document.querySelector('.cxcomposer__budget-error')?.textContent.includes('4,096-token context')
  ), { timeout: 30000 })
  const researchCountBeforeOverfullSend = researchRequests.length
  const chatCountBeforeOverfullSend = chatRequests.length
  await page.focus('textarea[aria-label="Message Camelid"]:not([disabled])')
  await page.keyboard.press('Enter')
  await page.waitForFunction(() => (
    document.body.textContent.includes('active 4,096-token runtime context')
      || document.body.textContent.includes("server's 4,096-token prompt limit")
  ), { timeout: 30000 })
  assert.equal(researchRequests.length, researchCountBeforeOverfullSend, 'a known-overfull active context must block before Web Auto')
  assert.equal(chatRequests.length, chatCountBeforeOverfullSend, 'a known-overfull active context must block before model chat')
  await setComposerText('')

  // Populate a second conversation before holding research in the first. This
  // catches global-loading UI leaking into a real, non-empty selected chat.
  await page.click('button.rail__new-chat')
  await sendPrompt(SECOND_CHAT_PROMPT)
  await waitForAnswer('Second chat answer.')
  await selectConversationByTitle('example.test/web-auto', 'Web Off answer.')

  const partialScenario = { hold: true, result: partialResearch }
  researchScenarios.push(partialScenario)
  const researchCountBeforeLinkedSend = researchRequests.length
  const chatCountBeforeLinkedSend = chatRequests.length
  await sendPrompt(LINKED_PROMPT)
  await waitUntil(
    () => researchRequests.length === researchCountBeforeLinkedSend + 1,
    'the held linked-source research request',
  )
  const linkedRequest = researchRequests.at(-1)
  assert.deepEqual(linkedRequest.body, { prompt: LINKED_PROMPT }, 'the browser must preserve exact branch/blob URLs in the research prompt')
  await page.waitForFunction(() => (
    document.body.textContent.includes('Reading relevant web sources…')
      && Boolean(document.querySelector('button[aria-label="Stop web research"]'))
  ), { timeout: 30000 })
  assert.equal(chatRequests.length, chatCountBeforeLinkedSend, 'model chat must wait for the research preflight')

  // Research progress belongs to the conversation that started it. Switching
  // to populated chat B must show neither a phantom answer nor an abort-A
  // control, while the global active request still disables a second send.
  await selectConversationByTitle('local conversation isolation', 'Second chat answer.')
  await page.waitForFunction(() => (
    !document.body.textContent.includes('Reading relevant web sources…')
      && !document.querySelector('button[aria-label="Stop web research"]')
      && !document.querySelector('button[aria-label="Stop Camelid generation"]')
      && !document.querySelector('.cxturn--assistant.is-streaming')
      && document.querySelector('button[aria-label="Send message"]')?.disabled
  ), { timeout: 30000 })
  assert.equal(linkedRequest.aborted, false, 'switching chats must not cancel the originating research request')
  const selectedConversationB = await page.evaluate(() => localStorage.getItem('camelid.selectedConversationId'))
  linkedRequest.release()
  await waitUntil(
    () => chatRequests.length === chatCountBeforeLinkedSend + 1,
    'the background research completion to start its model request',
  )
  await page.waitForFunction((answer) => {
    const conversations = JSON.parse(localStorage.getItem('camelid.conversations') || '[]')
    return conversations.some((conversation) => (
      (conversation.messages || []).some((message) => String(message?.content || '').includes(answer))
    ))
  }, { timeout: 30000 }, 'Grounded branch answer.')
  assert.equal(linkedRequest.aborted, false, 'the released linked-source request should complete normally')
  assert.equal(chatRequests.length, chatCountBeforeLinkedSend + 1, 'linked research should feed one ordinary chat completion')
  assert.equal(
    await page.evaluate(() => localStorage.getItem('camelid.selectedConversationId')),
    selectedConversationB,
    'background completion in chat A must not steal selection from populated chat B',
  )
  const visibleBackgroundCompletionState = await page.$eval('main[data-view="chat"]', (main) => ({
    hasSecondChat: main.textContent.includes('Second chat answer.'),
    hasBackgroundAnswer: main.textContent.includes('Grounded branch answer.'),
  }))
  assert.equal(visibleBackgroundCompletionState.hasSecondChat, true, 'chat B must remain visible while A finishes in storage')
  assert.equal(visibleBackgroundCompletionState.hasBackgroundAnswer, false, 'chat A must not render as a phantom answer inside selected chat B')

  await selectConversationByTitle('example.test/web-auto', 'Web Off answer.')
  await waitForAnswer('Grounded branch answer.')
  assert.ok(
    chatRequests.at(-1).max_tokens > 0 && chatRequests.at(-1).max_tokens < 8192,
    'the 4K active runtime must jointly fit evidence and reply below the 8192 requested default',
  )
  const evidenceMessage = chatRequests.at(-1).messages[0]
  assert.equal(evidenceMessage.role, 'system')
  assert.match(evidenceMessage.content, /UNTRUSTED EXTERNAL DATA/)
  assert.ok(evidenceMessage.content.includes(BRANCH_URL), 'chat evidence must retain the exact GitHub branch URL')
  assert.ok(evidenceMessage.content.includes(BRANCH_CHUNK_URL), 'chat evidence must retain the exact README chunk URL')
  assert.ok(evidenceMessage.content.includes(BLOB_URL), 'chat evidence must retain the exact GitHub blob URL')

  await page.waitForSelector('.cxturn__web-sources')
  await page.click('.cxturn__web-sources summary')
  const renderedSources = await page.$eval('.cxturn__web-sources', (details) => ({
    text: details.textContent,
    hrefs: [...details.querySelectorAll('a')].map((link) => link.href),
  }))
  assert.ok(renderedSources.text.includes('2'), 'the source panel should show both fitted sources')
  assert.ok(renderedSources.text.includes('Partial'), 'the source panel should disclose a partial result')
  assert.ok(renderedSources.text.includes(PARTIAL_WARNING), 'the source panel should render the actionable warning')
  assert.deepEqual(renderedSources.hrefs, [BRANCH_CHUNK_URL, BLOB_URL], 'source links must use exact evidence chunk URLs')

  const storedBeforeReload = await page.evaluate(() => JSON.parse(localStorage.getItem('camelid.conversations') || '[]'))
  const persistedReply = storedBeforeReload
    .flatMap((conversation) => conversation.messages || [])
    .find((message) => String(message?.content || '').includes('Grounded branch answer.'))
  assert.ok(persistedReply?.web_research, 'the completed reply should persist its Web source metadata')
  assert.deepEqual(
    persistedReply.web_research.sources.map((source) => source.url),
    [BRANCH_CHUNK_URL, BLOB_URL],
    'persisted provenance should retain exact injected chunk URLs',
  )
  assert.deepEqual(persistedReply.web_research.warnings, [PARTIAL_WARNING])
  for (const source of persistedReply.web_research.sources) {
    assert.deepEqual(Object.keys(source).sort(), ['title', 'url'], 'persisted sources must omit fetched text and chunks')
  }
  const persistedJson = JSON.stringify(storedBeforeReload)
  assert.equal(persistedJson.includes('BRANCH_EXCERPT_SECRET'), false, 'branch excerpts must not enter conversation storage')
  assert.equal(persistedJson.includes('BLOB_EXCERPT_SECRET'), false, 'blob excerpts must not enter conversation storage')

  const researchCountBeforeReload = researchRequests.length
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 30000 })
  await waitForComposer()
  await page.waitForFunction((answer) => document.body.textContent.includes(answer), { timeout: 30000 }, 'Grounded branch answer.')
  await page.waitForSelector('.cxturn__web-sources')
  const reloadedSources = await page.$eval('.cxturn__web-sources', (details) => ({
    text: details.textContent,
    hrefs: [...details.querySelectorAll('a')].map((link) => link.href),
  }))
  assert.equal(researchRequests.length, researchCountBeforeReload, 'reload must restore metadata without replaying research')
  assert.ok(reloadedSources.text.includes('Partial'))
  assert.ok(reloadedSources.text.includes(PARTIAL_WARNING))
  assert.deepEqual(reloadedSources.hrefs, [BRANCH_CHUNK_URL, BLOB_URL])

  // Stop uses the same AbortController as research and must prevent model chat.
  const stopScenario = { hold: true, result: partialResearch }
  researchScenarios.push(stopScenario)
  const researchCountBeforeStop = researchRequests.length
  const chatCountBeforeStop = chatRequests.length
  await sendPrompt(STOP_PROMPT)
  await waitUntil(
    () => researchRequests.length === researchCountBeforeStop + 1,
    'the held research request that will be stopped',
  )
  const stoppedRequest = researchRequests.at(-1)
  await page.waitForFunction(() => (
    document.body.textContent.includes('Reading relevant web sources…')
      && Boolean(document.querySelector('button[aria-label="Stop web research"]'))
  ), { timeout: 30000 })
  await page.click('button[aria-label="Stop web research"]')
  await waitUntil(() => stoppedRequest.aborted, 'the browser to abort held research')
  await page.waitForFunction(() => (
    !document.querySelector('.cxcomposer__stop')
      && !document.querySelector('button[aria-label="Turn off automatic web research"]:disabled')
  ), { timeout: 30000 })
  assert.equal(chatRequests.length, chatCountBeforeStop, 'stopping research must prevent a model completion request')
  assert.equal(
    await page.evaluate(() => document.body.textContent.includes('Reading relevant web sources…')),
    false,
    'research progress should clear after cancellation',
  )

  assert.deepEqual(externalRequests, [], `the smoke attempted non-loopback requests: ${externalRequests.join(', ')}`)
  assert.deepEqual(pageErrors, [], `browser errors: ${pageErrors.join('\n')}`)

  console.log('WEB_RESEARCH_BROWSER_SMOKE_PASS')
  console.log(JSON.stringify({ researchRequests: researchRequests.length, chatRequests: chatRequests.length }))
} finally {
  await restoreAnimationFrames().catch(() => {})
  for (const request of researchRequests) request.release()
  for (const stream of heldChatStreams) stream.release()
  await page.close().catch(() => {})
  await browser.close().catch(() => {})
  await new Promise((done) => server.close(done))
}
