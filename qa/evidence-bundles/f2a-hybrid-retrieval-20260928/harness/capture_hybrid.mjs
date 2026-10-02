#!/usr/bin/env node
// Capture hybrid-retrieval evidence from the shipped UI against a live `camelid serve` with real models.
// Usage: node capture_hybrid.mjs <phase: semantic|keyword-only> <origin> <out dir> <launch-browser.mjs> <policy> [progress doc]
import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import { pathToFileURL } from 'node:url'

const [phase, origin, outDir, launcherPath, policyPath, progressPath] = process.argv.slice(2)
const { launchBrowser } = await import(pathToFileURL(launcherPath).href)
const QUESTION = 'If hackers steal my information, when will you tell me?'
const EXPECTED = 'reported to the account owner within seventy-two hours'
const log = []
const summary = { phase, question: QUESTION }

mkdirSync(join(outDir, 'screenshots'), { recursive: true })

async function api(method, path, body) {
  const res = await fetch(origin + path, body === undefined ? { method } : {
    method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const json = await res.json().catch(() => null)
  return { status: res.status, json }
}
const brief = (result) => ({
  chunk_index: result.chunk_index, retrieval: result.retrieval, score: result.score,
  holds_expected_passage: result.excerpt.includes(EXPECTED),
  excerpt_sha256: createHash('sha256').update(result.excerpt).digest('hex'),
})

const browser = await launchBrowser({ purpose: 'hybrid retrieval evidence capture', headless: 'new' })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 })
const pageErrors = []
page.on('pageerror', (error) => pageErrors.push(String(error)))
const pause = (ms) => new Promise((done) => setTimeout(done, ms))
const shot = async (name, element = null) => {
  await pause(600)
  const path = join(outDir, 'screenshots', name)
  if (element) await element.screenshot({ path, type: 'png' })
  else await page.screenshot({ path, type: 'png' })
  console.log(`captured ${name}`)
}
const attach = async (path) => {
  const input = await page.$('input[type=file][accept*=".txt"]')
  await input.uploadFile(path)
  await page.waitForSelector(`.cxcomposer__doc-open[title="Open ${basename(path)}"]`, { timeout: 120000 })
  const attached = await page.evaluate(() => JSON.parse(localStorage.getItem('camelid.attachedDocuments') || '[]'))
  return attached.find((doc) => doc.filename === basename(path)).doc_id
}
const chunkLabel = (name) => page.$eval(`.cxcomposer__doc-open[title="Open ${name}"] .cxcomposer__doc-chunks`, (node) => node.textContent)

try {
  summary.build = (await api('GET', '/v1/health')).json?.build ?? null
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 60000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]', { timeout: 60000 })
  await page.waitForFunction(() => document.body.innerText.includes('is loaded and ready'), { timeout: 600000 })

  if (phase === 'keyword-only') {
    // The encoder is not in this server's models directory.
    await attach(policyPath)
    await page.waitForSelector('.cxcomposer__semantic-note', { timeout: 30000 })
    summary.note = await page.$eval('.cxcomposer__semantic-note', (node) => node.textContent)
    summary.chip = await chunkLabel(basename(policyPath))
    summary.index_status = (await api('GET', '/api/documents/index-status')).json.semantic
    await shot('04-keyword-only-without-encoder.png', await page.$('.cxcomposer__box'))
  } else {
    // 1. A larger upload shows background indexing progress; keyword search works meanwhile.
    const progressId = await attach(progressPath)
    await page.waitForFunction((name) => {
      const node = document.querySelector(`.cxcomposer__doc-open[title="Open ${name}"] .cxcomposer__doc-chunks--indexing`)
      const match = node?.textContent.match(/indexing (\d+)\/(\d+)/)
      return match && Number(match[1]) >= 4
    }, { timeout: 600000, polling: 500 }, basename(progressPath))
    summary.progress_chip = await chunkLabel(basename(progressPath))
    await shot('01-indexing-progress.png', await page.$('.cxcomposer__box'))
    const midway = await api('GET', '/api/documents/index-status')
    summary.progress_status = { semantic: midway.json.semantic, document: midway.json.documents.find((doc) => doc.id === progressId) }
    await page.click(`.cxcomposer__doc-pill button[aria-label="Remove ${basename(progressPath)}"]`)

    // 2. The policy, indexed. The chip reads "5 chunks" for a moment before its first status
    // poll returns, so wait on the server's own count first, then on the chip.
    const policyId = await attach(policyPath)
    const deadline = Date.now() + 600000
    for (;;) {
      const status = await api('GET', '/api/documents/index-status')
      const doc = status.json.documents.find((entry) => entry.id === policyId)
      if (doc && doc.indexed_chunks === doc.indexable_chunks) break
      if (Date.now() > deadline) throw new Error('policy not indexed in time')
      await pause(500)
    }
    await page.waitForFunction((name) => {
      const node = document.querySelector(`.cxcomposer__doc-open[title="Open ${name}"] .cxcomposer__doc-chunks`)
      return node && !node.classList.contains('cxcomposer__doc-chunks--indexing') && /chunks$/.test(node.textContent)
    }, { timeout: 600000, polling: 500 }, basename(policyPath))
    const ready = await api('GET', '/api/documents/index-status')
    summary.policy_status = ready.json.documents.find((doc) => doc.id === policyId)
    if (summary.policy_status.indexed_chunks !== summary.policy_status.indexable_chunks) throw new Error('policy not fully indexed')

    // 3. The same question through the API in both modes, recorded for the README.
    for (const mode of ['keyword', 'hybrid']) {
      const { status, json } = await api('POST', '/api/documents/search', { query: QUESTION, doc_ids: [policyId], top_k: 4, mode })
      log.push({ request: { mode, top_k: 4 }, status, retrieval: json.retrieval, results: json.results.map(brief) })
    }

    // 4. The real answer through the chat.
    await page.$eval('textarea[aria-label="Message Camelid"]:not([disabled])', (textarea, value) => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
      setter.call(textarea, value)
      textarea.dispatchEvent(new Event('input', { bubbles: true }))
    }, QUESTION)
    await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 300000 })
    await page.click('button[aria-label="Send message"]')
    await page.waitForSelector('.cxturn__user-doc', { timeout: 300000 })
    await page.waitForFunction(() => document.querySelectorAll('.cxturn--assistant').length > 0, { timeout: 300000 })
    await page.waitForFunction(() => !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming'), { timeout: 600000 })
    summary.answer = await page.evaluate(() => [...document.querySelectorAll('.cxturn--assistant')].at(-1).querySelector('.cxturn__body p')?.innerText)
    summary.message_chips = await page.$$eval('.cxturn__user-doc', (nodes) => nodes.map((node) => node.textContent))
    const citations = await page.evaluate(() => {
      const conversations = JSON.parse(localStorage.getItem('camelid.conversations') || '[]')
      const messages = conversations.flatMap((conversation) => conversation.messages || [])
      return messages.filter((message) => message.role === 'assistant').at(-1)?.citations || []
    })
    summary.chat_citations = citations.map(brief)
    await page.evaluate(() => document.querySelector('.cxturn--user').scrollIntoView({ block: 'start' }))
    await shot('02-answer-found-by-meaning.png')

    // 5. Open the citation that holds the incident passage.
    const pills = await page.$$eval('.cxturn--assistant button.citation-pill', (nodes) => nodes.map((node) => node.title))
    summary.pills = pills
    const target = citations.findIndex((citation) => citation.excerpt.includes(EXPECTED))
    summary.expected_citation_number = target >= 0 ? target + 1 : null
    const pill = pills.find((title) => title === `View source citation [${target + 1}]`)
    if (pill) {
      await page.click(`.cxturn--assistant button.citation-pill[title="${pill}"]`)
      await page.waitForFunction(() => document.querySelector('.citation-modal .citation-modal__badge')?.textContent.includes('Verified'), { timeout: 30000 })
      summary.citation_modal = await page.evaluate(() => ({
        badge: document.querySelector('.citation-modal__badge')?.textContent,
        found: document.querySelector('.citation-modal__found')?.textContent,
        span: document.querySelector('.citation-modal__span')?.textContent,
      }))
      await shot('03-citation-found-by-meaning.png')
      await page.click('.citation-modal__footer button')
    } else {
      summary.citation_modal = null
      console.log('the answer did not cite the incident passage with a pill; no citation screenshot')
    }
  }
  summary.page_errors = pageErrors
  writeFileSync(join(outDir, `capture-${phase}.json`), JSON.stringify(summary, null, 2))
  if (log.length) writeFileSync(join(outDir, 'api-transcript.json'), JSON.stringify(log, null, 2))
  console.log('CAPTURE_OK', JSON.stringify(summary))
} finally {
  await browser.close()
}
