#!/usr/bin/env node
/* Browser-level acceptance for memory.
 *
 * Requires `npm run build` first. One ephemeral loopback server serves the
 * compiled app and a fixture engine; every cross-origin request is aborted.
 * The fixture records every request, so each claim below is checked against
 * what was actually sent:
 *
 *   - memory is off until the user turns it on: nothing is suggested and
 *     nothing is sent
 *   - after a reply, the side request carries only the user's own message,
 *     with a JSON schema, and the suggestions appear under the reply
 *   - nothing is kept until the user chooses Remember; Edit, Not now and Undo
 *     do what they say; a kept memory records its conversation and turn
 *   - kept memories reach the next request; a memory not in use does not
 *   - the next message cancels a suggestion request still running
 *   - a chat can turn memory off for itself
 *   - the Memory page edits, adds, makes a memory from a note, and opens the
 *     chat at the message a memory came from
 *   - a lane that refuses the schema is asked again plainly
 *   - Forget everything forgets everything; saved notes never reach the model
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdirSync } from 'node:fs'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { extname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { launchBrowser } from './lib/launch-browser.mjs'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const distDir = resolve(scriptDir, '../dist')
const ledgerPath = resolve(scriptDir, '../../ledger/camelid-ledger.json')
const MODEL_FILENAME = 'Qwen3-0.6B-Q8_0.gguf'
const REPLY = 'FIXTURE-REPLY: here is an idea.'
const NOTE_TEXT = 'Passport, charger, NOTE-SECRET-TEXT'
const MEMORY_MARKER = 'Things the user asked you to remember'

const MIME = {
  '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = { ...ledger.capabilities, model_compatibility: ledger.model_rows.map((row) => row.contract) }

const MSG = {
  off: 'I am Priya and I am a nurse. Any quick dinner ideas?',
  sam: 'I am Sam, I live in Leeds and I am allergic to peanuts. What should I cook tonight?',
  dessert: 'What is a good dessert?',
  dog: 'I also have a dog called Biscuit.',
  thanks: 'Thanks!',
  colour: 'My favourite colour is green.',
  hello: 'Hello again.',
  japanese: 'I am learning Japanese.',
  cello: 'I play the cello in an orchestra.',
  last: 'One more question.',
}
const FACTS = {
  [MSG.off]: ['The user is named Priya.'],
  [MSG.sam]: ["The user's name is Sam.", 'The user lives in Leeds.', 'The user is allergic to peanuts.'],
  [MSG.dessert]: [],
  [MSG.dog]: ['The user has a dog called Biscuit.'],
  [MSG.thanks]: [],
  [MSG.colour]: ["The user's favourite colour is green."],
  [MSG.hello]: [],
  [MSG.japanese]: ['The user is learning Japanese.'],
  [MSG.cello]: ['The user plays the cello.'],
  [MSG.last]: [],
}
const fixture = { refuseSchema: false, delayFor: null }
/* Set MEMORY_SMOKE_SCREENSHOTS to a folder to keep a picture of each surface. */
const shotDir = process.env.MEMORY_SMOKE_SCREENSHOTS || ''
if (shotDir) mkdirSync(shotDir, { recursive: true })

const chatRequests = []
const extractionRequests = []
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
function sendChatCompletion(res, content) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' })
  res.socket?.setNoDelay(true)
  res.flushHeaders()
  const frame = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`)
  frame({ choices: [{ delta: { role: 'assistant' } }] })
  frame({ choices: [{ delta: { content } }] })
  frame({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 } })
  res.write('data: [DONE]\n\n')
  res.end()
}
const isFile = (path) => { try { return statSync(path).isFile() } catch { return false } }
const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const isExtraction = (body) => body?.messages?.[0]?.role === 'system'
  && String(body.messages[0].content).startsWith('You decide which facts about a user')
const messageFor = (body) => Object.keys(FACTS).find((text) => body.messages[1].content.endsWith(JSON.stringify(text)))

const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://127.0.0.1').pathname
    if (path === '/v1/health') {
      return sendJson(res, 200, {
        ok: true, engine: 'camelid', api_surface: 'full',
        version: 'memory-browser-smoke', build: 'memory-browser-smoke',
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
    if (path === '/v1/chat/completions' && req.method === 'POST') {
      const body = await readJsonBody(req)
      if (!isExtraction(body)) {
        chatRequests.push({ body, at: Date.now() })
        return sendChatCompletion(res, REPLY)
      }
      const message = messageFor(body)
      const record = { body, message, at: Date.now(), aborted: false, answered: false }
      extractionRequests.push(record)
      res.on('close', () => { if (!res.writableEnded) record.aborted = true })
      if (fixture.refuseSchema && body.response_format) {
        record.answered = true
        return sendJson(res, 400, { error: { message: 'structured-output constrained decoding is not supported on this model\'s serve lane yet', type: 'invalid_request_error', code: 'unsupported_parameter', param: 'response_format' } })
      }
      if (fixture.delayFor === message) {
        for (let waited = 0; waited < 4000 && !record.aborted; waited += 50) await sleep(50)
        if (record.aborted) return undefined
      }
      const facts = FACTS[message] ?? []
      const content = body.response_format
        ? JSON.stringify({ facts })
        : `Here is what I found:\n\`\`\`json\n${JSON.stringify({ facts })}\n\`\`\``
      record.answered = true
      return sendJson(res, 200, {
        id: 'fixture', object: 'chat.completion', model: body.model,
        choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 50, completion_tokens: 20, total_tokens: 70 },
      })
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

const browser = await launchBrowser({ purpose: 'the memory browser smoke', headless: 'new' })
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
await page.evaluateOnNewDocument((noteText) => {
  if (window.sessionStorage.getItem('camelid.memorySmokeInitialized')) return
  window.localStorage.clear()
  window.localStorage.setItem('camelid.webResearchEnabled', 'false')
  // A saved note from before memory existed: it must stay a note.
  window.localStorage.setItem('camelid.memories', JSON.stringify([{
    id: 'note-1', title: 'Packing list', body: noteText, scope: 'Travel',
    created_at: '2026-10-01T09:00:00.000Z', updated_at: '2026-10-01T09:00:00.000Z',
  }]))
  window.sessionStorage.setItem('camelid.memorySmokeInitialized', 'true')
}, NOTE_TEXT)

/* ---- helpers -------------------------------------------------------------- */
const storage = (key) => page.evaluate((key) => window.localStorage.getItem(key), key)
const storedMemories = async () => JSON.parse(await storage('camelid.userMemories') || '[]')
const currentConversation = () => page.evaluate(() => JSON.parse(localStorage.getItem('camelid.conversations') || '[]')
  .find((conversation) => conversation.id === localStorage.getItem('camelid.selectedConversationId')))
const memoryMessage = (body) => body.messages.find((m) => m.role === 'system' && String(m.content).includes(MEMORY_MARKER))
const clickVisible = async (selector, text = null) => {
  const handle = await page.waitForFunction((selector, text) => [...document.querySelectorAll(selector)]
    .find((element) => element.offsetParent !== null && (text === null || element.textContent.trim() === text)), { timeout: 15000 }, selector, text)
  await handle.asElement().click()
}
const fill = (selector, value) => page.$eval(selector, (element, value) => {
  const setter = Object.getOwnPropertyDescriptor(element.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, 'value').set
  setter.call(element, value)
  element.dispatchEvent(new Event('input', { bubbles: true }))
}, value)
const idle = () => page.waitForFunction(() => (
  !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming')
), { timeout: 30000 })
async function send(text) {
  const before = chatRequests.length
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await fill('textarea[aria-label="Message Camelid"]', text)
  await page.waitForFunction(() => (
    document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true'
  ), { timeout: 30000 })
  await page.click('button[aria-label="Send message"]')
  await idle()
  assert.equal(chatRequests.length, before + 1, `"${text}" sends one chat request`)
  return chatRequests.at(-1).body
}
const extractionFor = (message) => extractionRequests.filter((record) => record.message === message)
const waitForExtraction = async (message, count = 1) => {
  for (let i = 0; i < 200 && extractionFor(message).length < count; i += 1) await sleep(50)
  assert.equal(extractionFor(message).length, count, `${count} suggestion request(s) for "${message}"`)
}
const suggestionRows = () => page.$$eval('.memsug .memsug__item', (items) => items.map((item) => ({
  text: item.querySelector('p')?.textContent || item.querySelector('input')?.value || '',
  saved: item.classList.contains('is-saved'),
})))
/* Through the sidebar, as a user does: a hash change alone does not navigate. */
const navTo = async (label) => {
  const handle = await page.waitForFunction((label) => [...document.querySelectorAll('button')].find((button) => button.offsetParent !== null
    && (button.getAttribute('aria-label') === label || (button.classList.contains('rail__nav-item') && button.textContent.trim() === label))), { timeout: 15000 }, label)
  await handle.asElement().click()
}
const newChat = async () => {
  const handle = await page.waitForFunction(() => [...document.querySelectorAll('button')].find((button) => button.offsetParent !== null
    && (button.getAttribute('aria-label') === 'New chat' || button.classList.contains('rail__new-chat'))), { timeout: 15000 })
  await handle.asElement().click()
}
const openChatReady = async () => {
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
}

const shot = async (name, selector = null, { mobile = false } = {}) => {
  if (!shotDir) return
  const capture = async (file) => {
    await sleep(350)
    if (selector) await page.$eval(selector, (element) => element.scrollIntoView({ block: 'center' }))
    await page.screenshot({ path: resolve(shotDir, file) })
  }
  await capture(`${name}.png`)
  if (!mobile) return
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 })
  await capture(`${name}-mobile.png`)
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
  await sleep(200)
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await openChatReady()

  /* ---- 1. off until turned on ------------------------------------------- */
  const offRequest = await send(MSG.off)
  await sleep(800)
  assert.equal(extractionRequests.length, 0, 'memory off: nothing is asked of the model about the user')
  assert.equal(await page.$('.memsug'), null, 'memory off: nothing is suggested')
  assert.equal(memoryMessage(offRequest), undefined, 'memory off: no memory is sent')
  assert.equal(await storage('camelid.memoryEnabled'), null, 'nothing was switched on behind the user’s back')

  /* ---- 2. the switch in Settings ---------------------------------------- */
  await navTo('Settings')
  await page.waitForSelector('input[aria-label="Remember things about me"]', { timeout: 30000 })
  assert.equal(await page.$eval('input[aria-label="Remember things about me"]', (input) => input.checked), false, 'off by default')
  await page.click('input[aria-label="Remember things about me"]')
  await page.waitForFunction(() => localStorage.getItem('camelid.memoryEnabled') === 'true', { timeout: 5000 })
  await shot('0-settings', 'input[aria-label="Remember things about me"]')

  /* ---- 3. suggestions after a reply ------------------------------------- */
  await navTo('Chat')
  await openChatReady()
  await newChat()
  await openChatReady()
  await send(MSG.sam)
  await waitForExtraction(MSG.sam)
  await page.waitForFunction(() => document.querySelectorAll('.memsug .memsug__item').length === 3, { timeout: 10000 })
  const [samExtraction] = extractionFor(MSG.sam)
  const samChat = chatRequests.at(-1)
  assert.ok(samExtraction.at >= samChat.at, 'the side request follows the reply, never ahead of it')
  assert.equal(samExtraction.body.stream, false)
  assert.equal(samExtraction.body.response_format?.type, 'json_schema', 'asked with a schema first')
  assert.equal(samExtraction.body.messages.length, 2, 'just the instructions and the user’s message')
  assert.ok(!JSON.stringify(samExtraction.body).includes('FIXTURE-REPLY'), 'the reply is not read for facts about the user')
  assert.equal(samExtraction.body.model, MODEL_FILENAME, 'the same model as the chat')
  assert.deepEqual((await suggestionRows()).map((row) => row.text), FACTS[MSG.sam], 'the suggestions are shown under the reply')
  assert.deepEqual(await storedMemories(), [], 'nothing is kept before the user chooses')
  assert.match(await page.$eval('.memsug__head', (head) => head.textContent), /Remember this\?/)
  const costs = JSON.parse(await storage('camelid.memorySuggestionMs') || '{}')
  assert.ok(Number.isFinite(costs[MODEL_FILENAME]) && costs[MODEL_FILENAME] < 15000, 'how long the suggestion took is kept, per model')
  await shot('1-suggestions', '.memsug', { mobile: true })

  /* ---- 4. Remember, Edit, Not now, Undo --------------------------------- */
  await page.click('.memsug .memsug__item:nth-child(1) .memsug__primary')
  await page.waitForFunction(() => document.querySelector('.memsug .memsug__item:nth-child(1)')?.classList.contains('is-saved'), { timeout: 5000 })
  await clickVisible('.memsug .memsug__item:nth-child(2) button', 'Edit')
  await page.waitForSelector('input[aria-label="Edit what to remember"]')
  await fill('input[aria-label="Edit what to remember"]', 'The user lives in Leeds, UK.')
  await page.focus('input[aria-label="Edit what to remember"]')
  await page.keyboard.press('Enter')
  await page.waitForFunction(() => document.querySelectorAll('.memsug .memsug__item.is-saved').length === 2, { timeout: 5000 })
  await clickVisible('.memsug .memsug__item button', 'Not now')
  await page.waitForFunction(() => document.querySelectorAll('.memsug .memsug__item').length === 2, { timeout: 5000 })
  assert.match(await page.$eval('.memsug__head', (head) => head.textContent), /Remembered/, 'with nothing pending, the card says what was kept')
  await shot('2-remembered', '.memsug')

  let memories = await storedMemories()
  assert.deepEqual(memories.map((memory) => memory.text).sort(), ["The user's name is Sam.", 'The user lives in Leeds, UK.'].sort(), 'exactly what the user kept, as edited')
  const conversation = await currentConversation()
  const samMessage = conversation.messages.find((message) => message.role === 'user' && message.content === MSG.sam)
  for (const memory of memories) {
    assert.deepEqual(memory.source, { kind: 'chat', conversation_id: conversation.id, message_id: samMessage.id, turn: 1, conversation_title: conversation.title },
      'each memory records the conversation and the turn it came from')
    assert.equal(memory.enabled, true)
  }

  await clickVisible('.memsug .memsug__item:nth-child(1) button', 'Undo')
  await page.waitForFunction(() => document.querySelector('.memsug .memsug__item:nth-child(1) .memsug__primary'), { timeout: 5000 })
  assert.equal((await storedMemories()).length, 1, 'Undo forgets what was just kept')
  await page.click('.memsug .memsug__item:nth-child(1) .memsug__primary')
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('camelid.userMemories') || '[]').length === 2, { timeout: 5000 })

  /* ---- 5. kept memories reach the next request --------------------------- */
  const dessertRequest = await send(MSG.dessert)
  const sent = memoryMessage(dessertRequest)
  assert.ok(sent, 'kept memories are sent with the next message')
  assert.match(sent.content, /- The user's name is Sam\./)
  assert.match(sent.content, /- The user lives in Leeds, UK\./)
  assert.doesNotMatch(sent.content, /peanuts/, 'a dismissed suggestion is never sent')
  await waitForExtraction(MSG.dessert)
  assert.match(extractionFor(MSG.dessert)[0].body.messages[1].content, /already known:[\s\S]*The user lives in Leeds, UK\./, 'known facts are not suggested again')
  await sleep(300)
  assert.equal(await page.$$eval('.memsug', (cards) => cards.length), 1, 'nothing new to remember adds no card')

  /* ---- 6. the next message cancels a suggestion request ------------------ */
  fixture.delayFor = MSG.dog
  await send(MSG.dog)
  await waitForExtraction(MSG.dog)
  await send(MSG.thanks)
  await waitForExtraction(MSG.thanks)
  const [dogExtraction] = extractionFor(MSG.dog)
  assert.equal(dogExtraction.aborted, true, 'sending a message cancels the suggestion request still running')
  assert.equal(dogExtraction.answered, false)
  assert.ok(chatRequests.at(-1).at - dogExtraction.at < 4000, 'and the reply did not wait for it')
  assert.ok(!(await suggestionRows()).some((row) => row.text.includes('Biscuit')), 'a cancelled request suggests nothing')
  fixture.delayFor = null

  /* ---- 7. a chat can turn memory off ------------------------------------- */
  await clickVisible('[aria-label="Edit conversation context"]')
  await page.waitForSelector('.context-modal')
  const useMemory = await page.waitForFunction(() => [...document.querySelectorAll('.context-modal label.context-check')]
    .find((label) => label.textContent.trim() === 'Use memory in this chat')?.querySelector('input'))
  assert.equal(await useMemory.evaluate((input) => input.checked), true, 'a chat uses memory by default once it is on')
  assert.ok(await page.$eval('.context-modal', (modal) => modal.textContent.includes('Memory · 2 facts')), 'the dialog lists memory with what is sent')
  await shot('3-context-dialog')
  await useMemory.asElement().click()
  await clickVisible('button', 'Save context')
  await page.waitForSelector('.context-modal', { hidden: true })
  const colourRequest = await send(MSG.colour)
  assert.equal(memoryMessage(colourRequest), undefined, 'this chat no longer sends memory')
  await sleep(800)
  assert.equal(extractionFor(MSG.colour).length, 0, 'nor looks for anything to remember')

  /* ---- 8. the Memory page ------------------------------------------------ */
  await navTo('Memory')
  await page.waitForSelector('.memory-list .memory-item', { timeout: 30000 })
  assert.equal(await page.$eval('#memory-tab-memories', (tab) => tab.getAttribute('aria-selected')), 'true')
  await shot('4-memory-page', null, { mobile: true })
  const items = await page.$$eval('.memory-item', (rows) => rows.map((row) => ({
    text: row.querySelector('.memory-item__text')?.textContent,
    source: row.querySelector('.memory-item__source')?.textContent.trim(),
  })))
  assert.equal(items.length, 2)
  for (const item of items) assert.match(item.source, /^From “.+”, message 1$/, 'each says which chat and which message it came from')

  // Not in use: kept, but not sent.
  // Rows are found by their memory text: the source line names the chat, whose title mentions both.
  const samRow = await page.waitForFunction(() => [...document.querySelectorAll('.memory-item')]
    .find((row) => row.querySelector('.memory-item__text')?.textContent.includes('Sam')))
  await (await samRow.asElement().$('.memory-item__use input')).click()
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('camelid.userMemories')).find((m) => m.text.includes('Sam'))?.enabled === false)
  assert.ok(await samRow.asElement().evaluate((row) => row.classList.contains('is-off')))

  // Edit.
  await page.evaluate(() => [...document.querySelectorAll('.memory-item')]
    .find((row) => row.querySelector('.memory-item__text')?.textContent.includes('Leeds'))
    .querySelector('.memory-item__actions button').click())
  await page.waitForSelector('input[aria-label="Edit memory"]')
  await fill('input[aria-label="Edit memory"]', 'The user lives in Leeds, England.')
  await clickVisible('.memory-item__edit button', 'Save')
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('camelid.userMemories')).some((m) => m.text === 'The user lives in Leeds, England.' && m.edited === true && m.enabled))
  assert.ok((await storedMemories()).some((m) => m.text === "The user's name is Sam." && !m.edited), 'only the edited memory changed')

  // Add one by hand.
  await fill('input[aria-label="Something to remember"]', 'I prefer metric units.')
  await clickVisible('.memory-add button', 'Remember')
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('camelid.userMemories')).some((m) => m.text === 'I prefer metric units.' && m.source.kind === 'manual'))

  // A note becomes a memory only when the user writes it as one.
  await page.click('#memory-tab-notes')
  await page.waitForFunction(() => document.querySelector('#memory-panel-notes')?.textContent.includes('Packing list'))
  await shot('5-saved-notes')
  await clickVisible('#memory-panel-notes button', 'Make a memory')
  await page.waitForSelector('#memory-panel-memories input[aria-label="Something to remember"]')
  assert.equal(await page.$eval('input[aria-label="Something to remember"]', (input) => input.value), NOTE_TEXT, 'the note is offered as a draft')
  assert.equal((await storedMemories()).length, 3, 'and nothing is kept until the user saves it')
  await fill('input[aria-label="Something to remember"]', 'The user always packs a charger.')
  await clickVisible('.memory-add button', 'Remember')
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('camelid.userMemories'))
    .some((m) => m.text === 'The user always packs a charger.' && m.source.kind === 'note' && m.source.note_title === 'Packing list'))
  assert.ok(await page.evaluate(() => JSON.parse(localStorage.getItem('camelid.memories')).some((note) => note.id === 'note-1')), 'the note itself stays')

  // Where it came from: the chat opens at that message.
  await clickVisible('button.memory-item__source')
  await page.waitForSelector(`.cxturn.is-focused[data-message-id="${samMessage.id}"]`, { timeout: 10000 })
  await shot('6-opened-at-source')

  /* ---- 9. only memories in use are sent ---------------------------------- */
  await newChat()
  await openChatReady()
  const helloRequest = await send(MSG.hello)
  const helloMemory = memoryMessage(helloRequest)?.content || ''
  assert.match(helloMemory, /- The user lives in Leeds, England\./, 'as edited')
  assert.match(helloMemory, /- I prefer metric units\./)
  assert.match(helloMemory, /- The user always packs a charger\./)
  assert.doesNotMatch(helloMemory, /Sam/, 'a memory not in use is not sent')

  /* ---- 10. a lane that refuses the schema -------------------------------- */
  fixture.refuseSchema = true
  await send(MSG.japanese)
  await waitForExtraction(MSG.japanese, 2)
  const [schemaTry, plainTry] = extractionFor(MSG.japanese)
  assert.equal(schemaTry.body.response_format?.type, 'json_schema')
  assert.equal(plainTry.body.response_format, undefined, 'asked again without the schema')
  assert.match(plainTry.body.messages[1].content, /Answer with only the JSON object/)
  await page.waitForFunction(() => [...document.querySelectorAll('.memsug .memsug__item p')].some((p) => p.textContent === 'The user is learning Japanese.'), { timeout: 10000 })
  fixture.refuseSchema = false

  /* ---- 10b. a model too slow to ask on its own waits for the user ------- */
  await page.evaluate((model) => localStorage.setItem('camelid.memorySuggestionMs', JSON.stringify({ [model]: 60000 })), MODEL_FILENAME)
  await send(MSG.cello)
  await sleep(800)
  assert.equal(extractionFor(MSG.cello).length, 0, 'a slow model is not asked on its own')
  await shot('7-offer', '.memsug--offer')
  await clickVisible('.memsug--offer button', 'Look for things to remember')
  await waitForExtraction(MSG.cello)
  await page.waitForFunction(() => [...document.querySelectorAll('.memsug .memsug__item p')].some((p) => p.textContent === 'The user plays the cello.'), { timeout: 10000 })
  assert.equal(await page.$('.memsug--offer'), null, 'the offer is gone once the model has been asked')
  const celloMessage = (await currentConversation()).messages.find((message) => message.content === MSG.cello)
  assert.equal(celloMessage.memory_suggestion_offer, false)
  assert.ok(JSON.parse(await storage('camelid.memorySuggestionMs'))[MODEL_FILENAME] < 15000, 'a quick answer makes it automatic again')

  /* ---- 11. Forget everything ---------------------------------------------- */
  await navTo('Memory')
  await page.waitForSelector('.memory-list .memory-item', { timeout: 30000 })
  await clickVisible('.cxv-toolbar button', 'Forget everything')
  await clickVisible('[role="dialog"] button', 'Forget everything')
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('camelid.userMemories') || '[]').length === 0, { timeout: 5000 })

  /* ---- 12. off again -------------------------------------------------------- */
  await page.click('input[aria-label="Remember things about me"]')
  await page.waitForFunction(() => localStorage.getItem('camelid.memoryEnabled') === 'false', { timeout: 5000 })
  await navTo('Chat')
  await openChatReady()
  await newChat()
  await openChatReady()
  const lastRequest = await send(MSG.last)
  await sleep(800)
  assert.equal(memoryMessage(lastRequest), undefined)
  assert.equal(extractionFor(MSG.last).length, 0, 'memory off again: nothing is asked')
  await clickVisible('[aria-label="Edit conversation context"]')
  await page.waitForSelector('.context-modal')
  assert.ok(!(await page.$eval('.context-modal', (modal) => modal.textContent.includes('Use memory in this chat'))), 'with memory off, the chat offers no memory setting')
  await clickVisible('.context-modal button', 'Cancel')

  /* ---- 13. notes never reach the model ------------------------------------ */
  const everything = JSON.stringify([...chatRequests, ...extractionRequests].map((record) => record.body))
  assert.ok(!everything.includes('NOTE-SECRET-TEXT'), 'a saved note is never sent')

  assert.deepEqual(pageErrors, [], 'the page must not raise errors')
  assert.deepEqual(externalRequests, [], 'the smoke must not reach anything off-origin')
  console.log(`memory browser smoke passed (${chatRequests.length} chat and ${extractionRequests.length} suggestion requests checked)`)
} finally {
  await browser.close()
  await new Promise((done) => server.close(done))
}
