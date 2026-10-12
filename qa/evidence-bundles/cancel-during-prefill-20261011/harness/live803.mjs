// Live check for #803 in the real chat UI: stop a long message while the model is
// still reading it, then ask something short and see how long the answer takes.
// node live803.mjs <ui origin> <label> <out dir> <filler sentences> <stop after ms> <give up after ms>
import { writeFileSync, mkdirSync } from 'node:fs'
import { launchBrowser } from './scripts/lib/launch-browser.mjs'

const [origin, label, outDir, fillerArg, stopAfterArg, giveUpArg] = process.argv.slice(2)
const stopAfterMs = Number(stopAfterArg)
const giveUpMs = Number(giveUpArg)
mkdirSync(outDir, { recursive: true })

const subjects = ['The river', 'A lantern', 'The old bridge', 'Each harvest', 'The north road', 'A quiet bell', 'The mill', 'Every winter']
const verbs = ['carried', 'lit', 'connected', 'brought', 'led to', 'rang for', 'ground', 'covered']
const objects = ['the valley towns', 'the market square', 'two farming villages', 'grain and cider', 'the coastal fort', 'the evening prayers', 'barley for bread', 'the hills in snow']
let filler = ''
for (let i = 0; i < Number(fillerArg); i++) filler += `${subjects[i % 8]} ${verbs[(i * 3) % 8]} ${objects[(i * 5) % 8]} in year ${1500 + i}. `
const LONG = `${filler}\n\nIn one sentence, what is this text about?`
const SHORT = 'Say hi in three words.'

const browser = await launchBrowser({ purpose: 'the #803 live check', headless: 'new', protocolTimeout: 1800000 })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
const events = []
const t0 = Date.now()
const at = () => Date.now() - t0
const chatBodies = []
let shortRequest = null
let shortFinishedAt = null
page.on('request', (request) => {
  if (request.url().includes('/v1/chat/completions') && request.method() === 'POST') {
    chatBodies.push(request.postData())
    events.push({ at: at(), what: 'chat request sent' })
    if (!shortRequest && String(request.postData()).includes(SHORT)) shortRequest = request
  }
})
page.on('requestfinished', (request) => {
  if (request === shortRequest) {
    shortFinishedAt = at()
    events.push({ at: shortFinishedAt, what: 'short request finished (response fully received)' })
  }
})
page.on('requestfailed', (request) => {
  if (request.url().includes('/v1/chat/completions')) events.push({ at: at(), what: `chat request ended by the browser: ${request.failure()?.errorText}` })
})
await page.evaluateOnNewDocument(() => {
  if (window.sessionStorage.getItem('live803')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('live803', '1')
})
const composer = 'textarea[aria-label="Message Camelid"]:not([disabled])'
await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 60000 })
await page.waitForSelector(composer, { timeout: 60000 })
const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const type = async (text) => {
  await page.$eval(composer, (node, next) => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(node, next)
    node.dispatchEvent(new Event('input', { bubbles: true }))
  }, text)
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 30000 })
}
const shot = async (name) => {
  await page.screenshot({ path: `${outDir}/${label}-${name}.png` })
  events.push({ at: at(), what: `screenshot ${label}-${name}.png` })
}

await type(LONG)
await page.click('button[aria-label="Send message"]')
const longSentAt = at()
events.push({ at: longSentAt, what: 'long message sent' })
await page.waitForSelector('.cxcomposer__stop', { timeout: 30000 })
await sleep(stopAfterMs)
const longFirstReply = await page.$$eval('.cxturn--assistant', (nodes) => nodes.at(-1)?.innerText ?? '')
await shot('1-reading-long-message')
await page.click('.cxcomposer__stop')
events.push({ at: at(), what: 'Stop clicked' })
await page.waitForFunction(() => !document.querySelector('.cxcomposer__stop'), { timeout: 60000, polling: 200 })
events.push({ at: at(), what: 'Stop button gone' })
await sleep(1000)
await shot('2-stopped')

// A new chat, so the short question is sent alone: in the same chat the stopped
// message goes along as history and would have to be read again anyway.
const clicked = await page.$$eval('button', (nodes) => {
  const node = nodes.find((n) => n.textContent.trim() === 'New chat' && n.offsetParent !== null)
  node?.click()
  return Boolean(node)
})
if (!clicked) throw new Error('no visible New chat button')
await page.waitForFunction(() => document.querySelectorAll('.cxturn').length === 0, { timeout: 10000 })
events.push({ at: at(), what: 'New chat' })
await page.waitForSelector(composer, { timeout: 30000 })
await type(SHORT)
const bodiesBefore = chatBodies.length
await page.click('button[aria-label="Send message"]')
const shortSentAt = at()
events.push({ at: shortSentAt, what: 'short message sent' })
// On main the chat view was seen switching back to the stopped conversation while the
// short question waited, so show the short question's chat before each screenshot.
const showShortChat = async () => {
  const title = await page.$eval('h1, header', (n) => n.innerText).catch(() => '')
  if (title.includes(SHORT)) return
  const clicked = await page.$$eval('*', (nodes, text) => {
    const node = nodes.find((n) => n.children.length === 0 && n.textContent.trim() === text && n.offsetParent !== null)
    node?.click()
    return Boolean(node)
  }, SHORT)
  events.push({ at: at(), what: clicked ? 'opened the short question\'s chat from the sidebar' : 'short question\'s chat not found in the sidebar' })
  await sleep(800)
}
await sleep(20000)
await showShortChat()
await shot('3-20s-after-short-message')
const giveUpAt = Date.now() + giveUpMs
while (shortFinishedAt === null && Date.now() < giveUpAt) await sleep(250)
const answered = shortFinishedAt !== null
const shortDoneAt = answered ? shortFinishedAt : at()
if (!answered) events.push({ at: shortDoneAt, what: `gave up waiting after ${giveUpMs} ms` })
await sleep(1500)
await showShortChat()
const shortReply = await page.$$eval('.cxturn--assistant', (nodes) => nodes.at(-1)?.innerText ?? '')
await shot(answered ? '4-short-reply' : '4-still-waiting')
const shortBody = chatBodies.slice(bodiesBefore).map((b) => JSON.parse(b)).find((b) => String(b.messages?.at(-1)?.content ?? '').includes(SHORT))
const shortSent = shortBody?.messages ?? []
if (shortSent.filter((m) => m.role === 'user').length !== 1) {
  throw new Error(`the short request carried more than the short message: ${JSON.stringify(shortSent.map((m) => m.role))}`)
}
const result = {
  ui: label,
  short_request_roles: shortSent.map((m) => m.role),
  short_request_chars: shortSent.reduce((n, m) => n + String(m.content ?? '').length, 0),
  long_message_chars: LONG.length,
  stop_after_ms: stopAfterMs,
  long_turn_text_at_stop: longFirstReply,
  short_message: SHORT,
  short_reply: shortReply,
  short_answered: answered,
  short_wait_ms: shortDoneAt - shortSentAt,
  events,
}
writeFileSync(`${outDir}/${label}.json`, JSON.stringify(result, null, 1))
console.log(`${label}: short reply ${answered ? 'after' : 'NOT done after'} ${((shortDoneAt - shortSentAt) / 1000).toFixed(1)} s: ${shortReply.replace(/\s+/g, ' ').slice(0, 160)}`)
await browser.close()
