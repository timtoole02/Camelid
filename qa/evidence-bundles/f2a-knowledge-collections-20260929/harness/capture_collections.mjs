#!/usr/bin/env node
// Capture knowledge-collection evidence from the shipped UI against a live `camelid serve` with real models.
// Usage: node capture_collections.mjs <origin> <out dir> <launch-browser.mjs> <policy> <escalation guide> <readme>
import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { basename, join } from 'node:path'
import { pathToFileURL } from 'node:url'

const [origin, outDir, launcherPath, policyPath, guidePath, readmePath] = process.argv.slice(2)
const { launchBrowser } = await import(pathToFileURL(launcherPath).href)
const QUESTION = 'If hackers steal my information, when will you tell me?'
const EXPECTED = 'reported to the account owner within seventy-two hours'
const COLLECTION = 'Customer support'
const PROJECT = 'Support desk'
const summary = { question: QUESTION, collection: COLLECTION, project: PROJECT }
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
  holds_expected_passage: result.excerpt.includes(EXPECTED), excerpt_sha256: sha(result.excerpt),
})

const browser = await launchBrowser({ purpose: 'knowledge collections evidence capture', headless: 'new' })
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
const clickText = async (selector, text) => {
  const clicked = await page.$$eval(selector, (nodes, wanted) => {
    const node = nodes.find((item) => item.textContent.trim() === wanted)
    node?.click()
    return Boolean(node)
  }, text)
  if (!clicked) throw new Error(`no ${selector} reads ${text}`)
}
const texts = (selector) => page.$$eval(selector, (nodes) => nodes.map((node) => node.textContent.trim()))
const library = '.knowledge-modal'
const openLibrary = async () => {
  await page.click('button[aria-label="Attach"]')
  await page.waitForSelector('button[aria-label="Knowledge collections"]', { timeout: 10000 })
  await page.click('button[aria-label="Knowledge collections"]')
  await page.waitForSelector(library, { timeout: 10000 })
}
const closeLibrary = async () => {
  await page.click(`${library} .cx-modal__footer button`)
  await page.waitForFunction((s) => !document.querySelector(s), { timeout: 10000 }, library)
}
const waitIndexed = async (docIds) => {
  const deadline = Date.now() + 900000
  for (;;) {
    const status = await api('GET', '/api/documents/index-status')
    const docs = status.json.documents.filter((doc) => docIds.includes(doc.id))
    if (docs.length === docIds.length && docs.every((doc) => doc.indexed_chunks === doc.indexable_chunks)) return docs
    if (Date.now() > deadline) throw new Error('collection not indexed in time')
    await pause(500)
  }
}

try {
  summary.build = (await api('GET', '/v1/health')).json?.build ?? null
  // The library already holds two documents before the UI opens.
  for (const path of [readmePath, policyPath]) {
    const { status, json } = await api('POST', '/api/documents/ingest', { filename: basename(path), content: readFileSync(path, 'utf8') })
    if (status !== 200) throw new Error(`ingest ${basename(path)} failed: ${status}`)
    summary[`library_${basename(path)}`] = { doc_id: json.doc_id, chunk_count: json.chunk_count, sha256: sha(readFileSync(path)) }
  }
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 60000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]', { timeout: 60000 })
  await page.waitForFunction(() => document.body.innerText.includes('is loaded and ready'), { timeout: 600000 })

  // 1. Create the collection, add the policy from the library, upload the guide straight into it.
  await openLibrary()
  await setValue('input[aria-label="New collection name"]', COLLECTION)
  await page.click('button[aria-label="Create collection"]')
  await page.waitForFunction((name) => document.querySelector('.knowledge-detail h3')?.textContent.includes(name), { timeout: 10000 }, COLLECTION)
  await page.click(`${library} .knowledge-add summary`)
  await page.waitForSelector('.knowledge-pick input[type="checkbox"]', { visible: true, timeout: 10000 })
  summary.library_picker = await texts('.knowledge-pick label')
  await page.$$eval('.knowledge-pick label', (labels, wanted) => labels.find((label) => label.textContent.startsWith(wanted)).querySelector('input').click(), basename(policyPath))
  await clickText(`${library} button`, 'Add 1 document')
  await page.waitForFunction(() => document.querySelectorAll('.knowledge-docs li').length === 1, { timeout: 10000 })
  const upload = await page.$(`input[aria-label="Upload files into ${COLLECTION}"]`)
  await upload.uploadFile(guidePath)
  await page.waitForFunction(() => document.querySelectorAll('.knowledge-docs li').length === 2, { timeout: 120000 })
  await page.$eval('.knowledge-search-toggle input', (input) => input.click())
  await page.waitForFunction(() => document.querySelector('.knowledge-collection[aria-pressed="true"]')?.textContent.includes('In this chat'), { timeout: 10000 })
  const collections = (await api('GET', '/api/collections')).json
  const collection = collections.find((item) => item.name === COLLECTION)
  summary.collection_api = collection
  summary.members_indexed = await waitIndexed(collection.doc_ids)
  summary.library_members = await texts('.knowledge-docs li')
  await shot('01-knowledge-library.png', await page.$(`${library}`))
  await closeLibrary()

  // 2. The composer shows what the next message will search.
  await page.waitForFunction(() => /docs$/.test(document.querySelector('.cxcomposer__doc-pill--collection .cxcomposer__doc-chunks')?.textContent || ''), { timeout: 120000, polling: 500 })
  summary.composer_chips = await texts('.cxcomposer__doc-pill--collection .cxcomposer__doc-open')
  await shot('02-composer-collection-chip.png', await page.$('.cxcomposer__box'))

  // 3. A real answer from the collection.
  await setValue('textarea[aria-label="Message Camelid"]:not([disabled])', QUESTION)
  await page.waitForFunction(() => document.querySelector('button[aria-label="Send message"]')?.getAttribute('data-send-ready') === 'true', { timeout: 300000 })
  await page.click('button[aria-label="Send message"]')
  await page.waitForSelector('.cxturn__user-doc--collection', { timeout: 300000 })
  await page.waitForFunction(() => document.querySelectorAll('.cxturn--assistant').length > 0, { timeout: 300000 })
  await page.waitForFunction(() => !document.querySelector('.cxcomposer__stop') && !document.querySelector('.cxturn--assistant.is-streaming'), { timeout: 600000 })
  summary.ui_search_requests = [...uiSearches]
  summary.answer = await page.evaluate(() => [...document.querySelectorAll('.cxturn--assistant')].at(-1).querySelector('.cxturn__body')?.innerText)
  summary.message_chips = await texts('.cxturn__user-doc--collection')
  const citations = await page.evaluate(() => {
    const conversations = JSON.parse(localStorage.getItem('camelid.conversations') || '[]')
    const messages = conversations.flatMap((conversation) => conversation.messages || [])
    return messages.filter((message) => message.role === 'assistant').at(-1)?.citations || []
  })
  summary.chat_citations = citations.map(brief)
  summary.citations_all_members = citations.every((citation) => collection.doc_ids.includes(citation.doc_id))
  const replay = await api('POST', '/api/documents/search', uiSearches.at(-1))
  summary.search_replay = { status: replay.status, retrieval: replay.json.retrieval, results: replay.json.results.map(brief) }
  await page.evaluate(() => document.querySelector('.cxturn--user').scrollIntoView({ block: 'start' }))
  await shot('03-answer-from-collection.png')

  // 4. The citation that holds the incident passage, verified.
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
    console.log('the answer did not cite the incident passage with a pill; no citation screenshot')
  }

  // 5. A project that searches the collection, and a new chat in it.
  await page.evaluate(() => [...document.querySelectorAll('nav[aria-label="Primary"] button')].find((button) => button.textContent.trim() === 'Projects').click())
  await page.waitForSelector('.projects-view', { timeout: 10000 })
  await clickText('.projects-view button', 'New project')
  await page.waitForSelector('input[aria-label="Project name"]', { timeout: 10000 })
  await setValue('input[aria-label="Project name"]', PROJECT)
  await page.waitForSelector('.context-collections label input', { timeout: 10000 })
  await page.$$eval('.context-collections label', (labels, wanted) => labels.find((label) => label.textContent.startsWith(wanted)).querySelector('input').click(), COLLECTION)
  summary.project_editor_collections = await texts('.context-collections label')
  await shot('05-project-editor.png', await page.$('.context-modal'))
  await clickText('.context-modal .cx-modal__footer button', 'Save project')
  await page.waitForFunction((name) => [...document.querySelectorAll('.project-card')].some((card) => card.textContent.includes(name) && card.textContent.includes('1 collection')), { timeout: 10000 }, PROJECT)
  await page.click(`button[aria-label="New chat in ${PROJECT}"]`)
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  await page.waitForFunction(() => /project$/.test(document.querySelector('.cxcomposer__doc-pill--collection .cxcomposer__doc-chunks')?.textContent || ''), { timeout: 30000 })

  // 6. A collection deleted while a chat searches it.
  await openLibrary()
  await setValue('input[aria-label="New collection name"]', 'Old drafts')
  await page.click('button[aria-label="Create collection"]')
  await page.waitForFunction(() => document.querySelector('.knowledge-detail h3')?.textContent.includes('Old drafts'), { timeout: 10000 })
  await page.$eval('.knowledge-search-toggle input', (input) => input.click())
  await page.click('button[aria-label="Delete Old drafts"]')
  await clickText(`${library} .knowledge-confirm button`, 'Delete collection')
  await page.waitForFunction(() => ![...document.querySelectorAll('.knowledge-collection')].some((node) => node.textContent.includes('Old drafts')), { timeout: 10000 })
  await closeLibrary()
  await page.waitForSelector('.cxcomposer__doc-pill--collection.is-unavailable', { timeout: 10000 })
  summary.project_chat_chips = await texts('.cxcomposer__doc-pill--collection .cxcomposer__doc-open')
  await shot('06-project-chat-and-unavailable-chip.png', await page.$('.cxcomposer__box'))

  summary.page_errors = pageErrors
  writeFileSync(join(outDir, 'capture-collections.json'), JSON.stringify(summary, null, 2))
  console.log('CAPTURE_OK', JSON.stringify({ chips: summary.composer_chips, message: summary.message_chips, cited: summary.expected_citation_number }))
} finally {
  await browser.close()
}
