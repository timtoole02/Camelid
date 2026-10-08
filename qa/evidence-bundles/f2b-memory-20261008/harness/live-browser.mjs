/* Memory end to end against a real model: the built UI served by camelid
 * serve, driven in Chrome. Turns memory on, chats, waits for the model's own
 * suggestions, keeps them, and checks the next request carries them.
 *
 * Usage: node live-browser.mjs <port> <frontend dir> <out dir>
 */
import assert from 'node:assert/strict'
import { mkdirSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const [port, frontendDir, outDir, mode = 'auto'] = process.argv.slice(2)
const { launchBrowser } = await import(pathToFileURL(resolve(frontendDir, 'scripts/lib/launch-browser.mjs')).href)
mkdirSync(outDir, { recursive: true })
const origin = `http://127.0.0.1:${port}`
const FIRST = "Hi, I'm Priya. I work night shifts as a nurse in Leeds and I'm vegetarian. Can you suggest a quick dinner?"
const SECOND = 'What could I take to work for a snack?'
const MARKER = 'Things the user asked you to remember'

const requests = []
const pageErrors = []
const browser = await launchBrowser({ purpose: 'the live memory run', headless: 'new', protocolTimeout: 20 * 60 * 1000 })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
page.on('pageerror', (error) => pageErrors.push(String(error)))
page.on('request', (request) => {
  if (request.url().endsWith('/v1/chat/completions') && request.method() === 'POST') {
    try { requests.push({ at: Date.now(), body: JSON.parse(request.postData() || '{}') }) } catch { /* not JSON */ }
  }
})
await page.evaluateOnNewDocument(() => {
  if (window.sessionStorage.getItem('camelid.liveMemoryInitialized')) return
  window.localStorage.clear()
  window.localStorage.setItem('camelid.webResearchEnabled', 'false')
  // Settings → Response length at its minimum, so a CPU reply is short.
  window.localStorage.setItem('camelid.maxTokens', '256')
  window.sessionStorage.setItem('camelid.liveMemoryInitialized', 'true')
})

const isExtraction = (body) => String(body?.messages?.[0]?.content || '').startsWith('You decide which facts about a user')
const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const shot = async (name, selector = null) => {
  await sleep(400)
  if (selector) await page.$eval(selector, (element) => element.scrollIntoView({ block: 'center' })).catch(() => {})
  await page.screenshot({ path: resolve(outDir, `${name}.png`) })
}
const navTo = async (label) => {
  const handle = await page.waitForFunction((label) => [...document.querySelectorAll('button')].find((button) => button.offsetParent !== null
    && (button.getAttribute('aria-label') === label || (button.classList.contains('rail__nav-item') && button.textContent.trim() === label))), { timeout: 30000 }, label)
  await handle.asElement().click()
}
const fill = (selector, value) => page.$eval(selector, (element, value) => {
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
  setter.call(element, value)
  element.dispatchEvent(new Event('input', { bubbles: true }))
}, value)
async function send(text) {
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 120000 })
  await fill('textarea[aria-label="Message Camelid"]', text)
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 120000 })
  const started = Date.now()
  await page.click('button[aria-label="Send message"]')
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming'), { timeout: 600000 })
  return Date.now() - started
}

const result = {}
try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 60000 })
  await page.waitForSelector('main[data-view="chat"]', { timeout: 60000 })

  await navTo('Settings')
  await page.waitForSelector('input[aria-label="Remember things about me"]', { timeout: 30000 })
  await page.click('input[aria-label="Remember things about me"]')
  await page.waitForFunction(() => localStorage.getItem('camelid.memoryEnabled') === 'true')
  await navTo('Chat')

  result.first_reply_ms = await send(FIRST)
  const firstReply = await page.$$eval('.cxturn--assistant .cxturn__body .cxturn__md, .cxturn--assistant .cxturn__body', (turns) => turns.at(-1)?.textContent || '')
  result.first_reply = firstReply.slice(0, 600)
  result.mode_expected = mode
  result.first_token_ms = await page.evaluate(() => {
    const conversation = JSON.parse(localStorage.getItem('camelid.conversations') || '[]').find((c) => c.id === localStorage.getItem('camelid.selectedConversationId'))
    return conversation?.messages?.filter((m) => m.role === 'assistant').at(-1)?.first_content_ms ?? null
  })
  if (mode === 'offer') {
    await page.waitForSelector('.memsug--offer button', { timeout: 30000 })
    assert.equal(requests.filter((request) => isExtraction(request.body)).length, 0, 'a slow model is not asked on its own')
    await shot('live-0-offer', '.memsug--offer')
    await page.click('.memsug--offer button')
  }
  const suggestionStarted = Date.now()
  await page.waitForFunction(() => document.querySelectorAll('.memsug .memsug__item').length > 0, { timeout: 900000 })
  result.suggestions_after_reply_ms = Date.now() - suggestionStarted
  result.suggestions = await page.$$eval('.memsug .memsug__item p', (items) => items.map((p) => p.textContent))
  const extraction = requests.find((request) => isExtraction(request.body))
  result.suggestion_requests = requests.filter((request) => isExtraction(request.body)).map((request) => request.body.response_format?.type || 'plain')
  result.extraction_request = {
    model: extraction.body.model, stream: extraction.body.stream, response_format: extraction.body.response_format?.type || null,
    messages: extraction.body.messages.length,
    carries_reply: firstReply.trim().length > 40 && JSON.stringify(extraction.body).includes(JSON.stringify(firstReply.trim().slice(0, 40)).slice(1, -1)),
  }
  assert.equal(JSON.parse(await page.evaluate(() => localStorage.getItem('camelid.userMemories') || '[]')).length, 0, 'nothing kept before Remember')
  await shot('live-1-suggestions', '.memsug')

  // Keep every suggestion.
  while (await page.$('.memsug .memsug__item:not(.is-saved) .memsug__primary')) {
    await page.click('.memsug .memsug__item:not(.is-saved) .memsug__primary')
    await sleep(150)
  }
  const kept = JSON.parse(await page.evaluate(() => localStorage.getItem('camelid.userMemories') || '[]'))
  result.kept = kept.map((memory) => ({ text: memory.text, source: memory.source }))
  assert.ok(kept.length > 0)
  for (const memory of kept) assert.equal(memory.source.kind, 'chat')
  await shot('live-2-kept', '.memsug')

  const before = requests.length
  result.second_reply_ms = await send(SECOND)
  if (mode === 'offer') await sleep(2000)
  const nextChat = requests.slice(before).find((request) => !isExtraction(request.body))
  const memoryMessage = nextChat.body.messages.find((message) => message.role === 'system' && String(message.content).includes(MARKER))
  assert.ok(memoryMessage, 'the next request carries memory')
  for (const memory of kept) assert.ok(memoryMessage.content.includes(memory.text), `"${memory.text}" is sent`)
  result.sent_memory_source = memoryMessage.content
  result.second_reply = await page.$$eval('.cxturn--assistant .cxturn__body', (turns) => turns.at(-1)?.textContent?.slice(0, 600) || '')
  await shot('live-3-second-reply')

  await navTo('Memory')
  await page.waitForSelector('.memory-list .memory-item', { timeout: 30000 })
  await shot('live-4-memory-page')

  result.page_errors = pageErrors
  result.requests = requests.map((request) => ({ extraction: isExtraction(request.body), model: request.body.model, stream: request.body.stream }))
  writeFileSync(resolve(outDir, 'live-browser.json'), JSON.stringify(result, null, 1))
  console.log(JSON.stringify({ ...result, sent_memory_source: undefined, requests: result.requests.length }, null, 1))
} finally {
  await browser.close()
}
