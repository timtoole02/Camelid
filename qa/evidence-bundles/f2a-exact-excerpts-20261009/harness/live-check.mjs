// Live check for #800: the real chat UI, the real server, the 550-document library.
// Usage: node live800.mjs <ui origin> <label> <out dir>
import { writeFileSync, mkdirSync } from 'node:fs'
import { launchBrowser } from './scripts/lib/launch-browser.mjs'

const [origin, label, outDir] = process.argv.slice(2)
mkdirSync(outDir, { recursive: true })
const QUESTIONS = [
  [21, 'How much did it cost when the primary mirror of the Ivarsdal observatory was recoated?'],
  [13, 'How much did it cost when the north clarifier of the Wyncroft water treatment plant was relined?'],
]
const browser = await launchBrowser({ purpose: 'the #800 live check', headless: 'new', protocolTimeout: 900000 })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
const chatBodies = []
page.on('request', (request) => {
  if (request.url().includes('/v1/chat/completions') && request.method() === 'POST') chatBodies.push(request.postData())
})
await page.evaluateOnNewDocument(() => {
  if (window.sessionStorage.getItem('live800')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('live800', '1')
})
const composer = 'textarea[aria-label="Message Camelid"]:not([disabled])'
await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 60000 })
await page.waitForSelector(composer, { timeout: 60000 })
const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const results = []
for (const [id, question] of QUESTIONS) {
  if (results.length) {
    const clicked = await page.$$eval('button', (nodes) => {
      const node = nodes.find((n) => n.textContent.trim() === 'New chat' && n.offsetParent !== null)
      node?.click()
      return Boolean(node)
    })
    if (!clicked) throw new Error('no visible New chat button')
    await page.waitForFunction(() => document.querySelectorAll('.cxturn').length === 0, { timeout: 10000 })
  }
  await page.waitForSelector(composer, { timeout: 30000 })
  await page.click('button[aria-label="Attach"]')
  await page.waitForSelector('button[aria-label="Knowledge collections"]', { timeout: 5000 })
  await page.click('button[aria-label="Knowledge collections"]')
  await page.waitForSelector('.knowledge-modal', { timeout: 5000 })
  await page.$$eval('.knowledge-modal button.knowledge-collection', (nodes) => nodes.find((n) => n.textContent.includes('F2a 500-PDF gate')).click())
  await page.$eval('.knowledge-search-toggle input', (input) => { if (!input.checked) input.click() })
  await page.waitForFunction(() => [...document.querySelectorAll('.knowledge-collection')].some((n) => n.textContent.includes('In this chat')), { timeout: 5000 })
  await page.click('.knowledge-modal .cx-modal__footer button')
  await page.waitForFunction(() => !document.querySelector('.knowledge-modal'), { timeout: 5000 })
  await page.$eval(composer, (node, next) => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(node, next)
    node.dispatchEvent(new Event('input', { bubbles: true }))
  }, question)
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 30000 })
  const before = chatBodies.length
  await page.click('button[aria-label="Send message"]')
  await page.waitForFunction((n) => document.querySelectorAll('.cxturn--assistant').length > 0, { timeout: 120000 })
  while (chatBodies.length === before) await sleep(100)
  await sleep(1000)
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__stop'), { timeout: 600000, polling: 500 })
  await sleep(1500)
  const reply = await page.$$eval('.cxturn--assistant', (nodes) => nodes.at(-1).innerText)
  const body = JSON.parse(chatBodies.at(-1))
  const messages = body.messages
  const { messages: _sent, ...params } = body
  if (messages.filter((m) => m.role === 'user').length !== 1) throw new Error('the chat carried earlier turns')
  const sent = messages.filter((m) => m.role === 'user').at(-1).content
  await page.screenshot({ path: `${outDir}/${label}-q${id}.png` })
  results.push({ id, question, params, roles_sent: messages.map((m) => m.role), prompt_sent: sent, reply })
  console.log(`${label} q${id}: ${reply.replace(/\s+/g, ' ').slice(0, 300)}`)
}
writeFileSync(`${outDir}/${label}.json`, JSON.stringify({ ui: label, results }, null, 1))
await browser.close()
