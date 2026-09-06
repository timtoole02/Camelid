/* Design evidence for the chat context meter.

   Captures the three states a reviewer needs to judge this change against a
   running dev server, in the repo's usual way (real Chrome via puppeteer-core,
   fixed viewport, dark theme):

     01  the panel open, showing the receipt-style breakdown
     02  the collapsed chip in the composer status line
     03  a window smaller than the configured reply limit -- the case that
         exposed the clamped-reservation defect

   Shot 03 patches the /health response in the page, because the released
   binary this was captured against predates `active_context_length` and always
   omits it. The value injected (4096) is what a server that loads a model at
   4096 actually reports.

   Usage: node scripts/capture-context-meter.mjs [--url http://127.0.0.1:4175]
*/
import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { launchBrowser } from './lib/launch-browser.mjs'

const args = new Map()
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1])
}
const baseUrl = args.get('url') || 'http://127.0.0.1:4175'
const outDir = args.get('out') || 'design-evidence/chat-context-meter'
const WIDTH = 1440
const HEIGHT = 900

const DRAFT = 'Here is the implementation plan for the new ingestion pipeline. It covers the batching strategy, the retry semantics we agreed on, dead-letter handling, and the migration path off the legacy worker. Summarise it and call out the riskiest assumptions, especially anything that could silently drop records during the cutover window.'

await mkdir(outDir, { recursive: true })
const captured = []

const browser = await launchBrowser({ purpose: 'the context-meter evidence capture', headless: 'new' })

async function newPage({ contextLength = null } = {}) {
  const page = await browser.newPage()
  await page.setViewport({ width: WIDTH, height: HEIGHT })
  await page.evaluateOnNewDocument(() => window.localStorage.setItem('camelid-theme', 'dark'))
  if (contextLength !== null) {
    await page.evaluateOnNewDocument((ctx) => {
      const orig = window.fetch
      window.fetch = async (...a) => {
        const res = await orig(...a)
        const url = typeof a[0] === 'string' ? a[0] : (a[0] && a[0].url) || ''
        if (!/health/.test(url)) return res
        try {
          const body = await res.clone().json()
          body.active_context_length = ctx
          return new Response(JSON.stringify(body), { status: res.status, headers: { 'content-type': 'application/json' } })
        } catch { return res }
      }
    }, contextLength)
  }
  await page.goto(`${baseUrl}/#chat`, { waitUntil: 'networkidle2', timeout: 30000 })
  await new Promise((r) => setTimeout(r, 1500))
  return page
}

async function typeDraft(page) {
  await page.evaluate((text) => {
    const ta = document.querySelector('textarea.cxcomposer__input')
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
    setter.call(ta, text)
    ta.dispatchEvent(new Event('input', { bubbles: true }))
  }, DRAFT)
  await new Promise((r) => setTimeout(r, 800))
}

async function setPanel(page, open) {
  const isOpen = await page.$eval('.ctxmeter__chip', (el) => el.getAttribute('aria-expanded') === 'true')
  if (isOpen !== open) {
    await page.$eval('.ctxmeter__chip', (el) => el.click())
    await new Promise((r) => setTimeout(r, 700))
  }
}

/* Screenshots are cropped to the composer so the meter is legible in a PR
   review, rather than a 1440px page with the panel a tenth of its height.
   The panel is absolutely positioned, so it is outside `.ctxmeter`'s own rect
   and the clip has to be the union of every part we want in frame. */
async function shot(page, name, selectors) {
  const boxes = await page.evaluate((sel) => sel
    .map((s) => document.querySelector(s))
    .filter(Boolean)
    .map((el) => {
      const r = el.getBoundingClientRect()
      return { left: r.left, top: r.top, right: r.right, bottom: r.bottom }
    }), selectors)
  if (!boxes.length) throw new Error(`nothing matched ${selectors.join(', ')}`)
  const pad = 20
  const left = Math.max(Math.round(Math.min(...boxes.map((b) => b.left)) - pad), 0)
  const top = Math.max(Math.round(Math.min(...boxes.map((b) => b.top)) - pad), 0)
  const right = Math.min(Math.round(Math.max(...boxes.map((b) => b.right)) + pad), WIDTH)
  const bottom = Math.min(Math.round(Math.max(...boxes.map((b) => b.bottom)) + pad), HEIGHT)
  const file = join(outDir, name)
  await page.screenshot({ path: file, clip: { x: left, y: top, width: right - left, height: bottom - top } })
  console.log(`captured ${file}`)
  captured.push(file)
}

try {
  {
    const page = await newPage()
    await typeDraft(page)
    await setPanel(page, true)
    await shot(page, '01-panel-open.png', ['.ctxmeter__panel', '.ctxmeter__chip'])
    await setPanel(page, false)
    await shot(page, '02-collapsed-chip.png', ['.cxcomposer__status'])
    await page.close()
  }
  {
    // A model loaded at 4096 while the reply limit is the 8192 default.
    const page = await newPage({ contextLength: 4096 })
    await typeDraft(page)
    await setPanel(page, true)
    const label = await page.$eval('.ctxmeter__chip', (el) => el.getAttribute('aria-label'))
    if (/100% used/.test(label)) {
      throw new Error(`clamped window still reports a full context: ${label}`)
    }
    await shot(page, '03-small-window.png', ['.ctxmeter__panel', '.ctxmeter__chip'])
    console.log(`  small-window chip reads: ${label}`)
    await page.close()
  }
} finally {
  await browser.close()
}

const lines = []
const seen = new Map()
for (const file of captured.sort()) {
  const digest = createHash('sha256').update(await readFile(file)).digest('hex')
  if (seen.has(digest)) {
    console.error(`capture self-check FAILED: ${file} is pixel-identical to ${seen.get(digest)}`)
    process.exit(1)
  }
  seen.set(digest, file)
  lines.push(`${digest}  ${file.split(/[\\/]/).pop()}`)
}
await writeFile(join(outDir, 'SHA256SUMS'), `${lines.join('\n')}\n`)
console.log(`capture self-check passed: ${captured.length} screenshots, all distinct`)
