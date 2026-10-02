#!/usr/bin/env node
// Capture whole-library search evidence from the shipped UI against a live `camelid serve` with real models.
// Usage: node capture_library.mjs <phase: encoder|no-encoder> <origin> <out dir> <launch-browser.mjs> [files to ingest...]
import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import { pathToFileURL } from 'node:url'

const [phase, origin, outDir, launcherPath, ...files] = process.argv.slice(2)
const { launchBrowser } = await import(pathToFileURL(launcherPath).href)
const QUESTION = 'How long do refunds take?'
const EXPECTED = 'processing takes five business days'
const HARD = 'If hackers steal my information, when will you tell me?'
const HARD_PASSAGE = 'reported to the account owner within seventy-two hours'
const UNRELATED = 'Tell me a joke about penguins.'
const summary = { phase, question: QUESTION, hard_question: HARD, unrelated: UNRELATED }
const uiSearches = []

mkdirSync(join(outDir, 'screenshots'), { recursive: true })

async function api(method, path, body) {
  const res = await fetch(origin + path, body === undefined ? { method } : {
    method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  const json = res.status === 204 ? null : await res.json().catch(() => null)
  return { status: res.status, json }
}
const sha = (text) => createHash('sha256').update(text).digest('hex')
const brief = (result) => ({
  filename: result.filename, chunk_index: result.chunk_index, retrieval: result.retrieval,
  similarity: result.similarity ?? null, holds_expected_passage: result.excerpt.includes(EXPECTED),
  holds_hard_passage: result.excerpt.includes(HARD_PASSAGE), excerpt_sha256: sha(result.excerpt),
})

const browser = await launchBrowser({ purpose: 'library search evidence capture', headless: 'new' })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 })
const pageErrors = []
page.on('pageerror', (error) => pageErrors.push(String(error)))
page.on('request', (request) => {
  if (request.url().endsWith('/api/documents/search') && request.method() === 'POST') uiSearches.push(JSON.parse(request.postData()))
})
const pause = (ms) => new Promise((done) => setTimeout(done, ms))
const shot = async (name, element = null) => {
  await pause(600)
  const path = join(outDir, 'screenshots', name)
  if (element) await element.screenshot({ path, type: 'png' })
  else await page.screenshot({ path, type: 'png' })
  console.log(`captured ${name}`)
}
const setValue = (selector, value) => page.$eval(selector, (node, next) => {
  const prototype = node.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype
  Object.getOwnPropertyDescriptor(prototype, 'value').set.call(node, next)
  node.dispatchEvent(new Event('input', { bubbles: true }))
}, value)
const texts = (selector) => page.$$eval(selector, (nodes) => nodes.map((node) => node.textContent.trim()))
const libraryChip = '.cxcomposer__doc-pill--library'
const lastTurn = (selector) => page.$$eval('.cxturn--user', (turns, s) => [...turns.at(-1).querySelectorAll(s)].map((node) => node.textContent.trim()), selector)
const storedCitations = () => page.evaluate(() => {
  const conversations = JSON.parse(localStorage.getItem('camelid.conversations') || '[]')
  const messages = conversations.flatMap((conversation) => conversation.messages || [])
  return messages.filter((message) => message.role === 'assistant').at(-1)?.citations || []
})
const lastAnswer = () => page.evaluate(() => [...document.querySelectorAll('.cxturn--assistant')].at(-1).querySelector('.cxturn__body')?.innerText)
const scrollToLastQuestion = () => page.evaluate(() => [...document.querySelectorAll('.cxturn--user')].at(-1).scrollIntoView({ block: 'start' }))

async function send(text) {
  const before = uiSearches.length
  const turns = await page.$$eval('.cxturn--assistant', (nodes) => nodes.length)
  await setValue('textarea[aria-label="Message Camelid"]:not([disabled])', text)
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 300000 })
  await page.click('button[aria-label="Send message"]')
  await page.waitForFunction((n) => document.querySelectorAll('.cxturn--assistant').length > n, { timeout: 300000 }, turns)
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming'), { timeout: 600000 })
  return uiSearches.slice(before)
}
async function record(key, text) {
  const searches = await send(text)
  const replay = await api('POST', '/api/documents/search', searches.at(-1))
  summary[key] = {
    searches,
    replay: { status: replay.status, retrieval: replay.json.retrieval, results: replay.json.results.map(brief) },
    citations: (await storedCitations()).map(brief),
    message_chips: await lastTurn('.cxturn__user-doc--library'),
    answer: await lastAnswer(),
  }
}

try {
  summary.build = (await api('GET', '/v1/health')).json?.build ?? null
  for (const path of files) {
    const { status, json } = await api('POST', '/api/documents/ingest', { filename: basename(path), content: readFileSync(path, 'utf8') })
    if (status !== 200) throw new Error(`ingest ${basename(path)} failed: ${status}`)
    summary[`library_${basename(path)}`] = { doc_id: json.doc_id, chunk_count: json.chunk_count, bytes: readFileSync(path).length, sha256: sha(readFileSync(path)) }
  }
  if (phase === 'encoder') {
    const deadline = Date.now() + 900000
    for (;;) {
      const status = await api('GET', '/api/documents/index-status')
      if (status.json.documents.every((doc) => doc.indexed_chunks === doc.indexable_chunks)) { summary.index_status = status.json; break }
      if (Date.now() > deadline) throw new Error('library not indexed in time')
      await pause(1000)
    }
  }
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 60000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]', { timeout: 60000 })
  await page.waitForFunction(() => document.body.innerText.includes('is loaded and ready'), { timeout: 600000 })
  await page.click('button[aria-label="Attach"]')
  await page.waitForSelector('button[aria-label="Search the whole library"]', { timeout: 10000 })
  summary.attach_menu = await texts('.composer-menu__attachments button')
  if (phase === 'encoder') await shot('01-attach-menu.png', await page.$('.composer-menu__attachments'))
  await page.click('button[aria-label="Search the whole library"]')
  await page.waitForSelector(libraryChip, { timeout: 10000 })

  if (phase === 'no-encoder') {
    await page.waitForSelector('.cxcomposer__semantic-note--library', { timeout: 30000 })
    summary.note = await page.$eval('.cxcomposer__semantic-note--library', (node) => node.textContent.trim())
    summary.chip = await texts(`${libraryChip} .cxcomposer__doc-open`)
    summary.index_status = (await api('GET', '/api/documents/index-status')).json.semantic
    await shot('07-without-the-encoder.png', await page.$('.cxcomposer__box'))
  } else {
    await page.waitForFunction((s) => /docs$/.test(document.querySelector(`${s} .cxcomposer__doc-chunks`)?.textContent || ''), { timeout: 60000 }, libraryChip)
    summary.chip = await texts(`${libraryChip} .cxcomposer__doc-open`)
    await shot('02-composer-whole-library.png', await page.$('.cxcomposer__box'))

    // A question the library answers, with nothing attached.
    await record('answered', QUESTION)
    await scrollToLastQuestion()
    await shot('03-answer-from-the-library.png')
    const citations = await storedCitations()
    const target = citations.findIndex((citation) => citation.excerpt.includes(EXPECTED))
    summary.expected_citation_number = target >= 0 ? target + 1 : null
    const pills = await page.$$eval('.cxturn--assistant button.citation-pill', (nodes) => nodes.map((node) => node.title))
    summary.pills = pills
    if (target >= 0 && pills.includes(`View source citation [${target + 1}]`)) {
      await page.click(`.cxturn--assistant button.citation-pill[title="View source citation [${target + 1}]"]`)
      await page.waitForFunction(() => document.querySelector('.citation-modal .citation-modal__badge')?.textContent.includes('Verified'), { timeout: 30000 })
      summary.citation_modal = await page.evaluate(() => ({
        badge: document.querySelector('.citation-modal__badge')?.textContent,
        found: document.querySelector('.citation-modal__found')?.textContent,
        span: document.querySelector('.citation-modal__span')?.textContent,
      }))
      await shot('04-citation-verified.png')
      await page.click('.citation-modal__footer button')
    } else {
      summary.citation_modal = null
      console.log('the answer did not cite the refunds passage with a pill; no citation screenshot')
    }

    // A question whose answer shares a chunk with other topics: whatever clears the floor is recorded.
    await record('hard', HARD)
    await scrollToLastQuestion()
    await shot('05-question-below-the-floor.png')

    // A message the library has nothing to say about.
    await record('unrelated', UNRELATED)
    await scrollToLastQuestion()
    await shot('06-unrelated-message.png')
  }
  summary.page_errors = pageErrors
  writeFileSync(join(outDir, `capture-${phase}.json`), JSON.stringify(summary, null, 2))
  console.log('CAPTURE_OK', JSON.stringify({ chip: summary.chip, answered: summary.answered?.message_chips, hard: summary.hard?.message_chips, unrelated: summary.unrelated?.message_chips, cited: summary.expected_citation_number, note: summary.note }))
} finally {
  await browser.close()
}
