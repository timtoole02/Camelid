#!/usr/bin/env node
// Capture watched-folder evidence from the shipped UI against a live `camelid serve`.
// Usage: node capture_folders.mjs <origin> <out dir> <launch-browser.mjs> <folder parent> <watched folder>
import { appendFileSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

const [origin, outDir, launcherPath, parent, watched] = process.argv.slice(2)
const { launchBrowser } = await import(pathToFileURL(launcherPath).href)
mkdirSync(join(outDir, 'screenshots'), { recursive: true })

const summary = { origin_is_loopback: new URL(origin).hostname === '127.0.0.1', folder_responses: [], steps: [] }
const browser = await launchBrowser({ purpose: 'watched folders evidence capture', headless: 'new' })
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 })
const pageErrors = []
page.on('pageerror', (error) => pageErrors.push(String(error)))
page.on('response', (response) => {
  const url = new URL(response.url())
  if (url.pathname.startsWith('/api/folders') || url.pathname === '/api/agent/workspace/browse') {
    summary.folder_responses.push(`${response.request().method()} ${url.pathname} ${response.status()}`)
  }
})
const pause = (ms) => new Promise((done) => setTimeout(done, ms))
const library = '.knowledge-modal'
const folderItem = `${library} .knowledge-folder`
const shot = async (name) => {
  await pause(600)
  await (await page.$(`${library} .cx-modal`) || await page.$(library)).screenshot({ path: join(outDir, 'screenshots', name), type: 'png' })
  console.log(`captured ${name}`)
}
const setValue = (selector, value) => page.$eval(selector, (node, next) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, next)
  node.dispatchEvent(new Event('input', { bubbles: true }))
}, value)
const clickText = async (selector, text) => {
  const clicked = await page.$$eval(selector, (nodes, wanted) => {
    const node = nodes.find((item) => item.textContent.trim() === wanted)
    if (node) node.click()
    return Boolean(node)
  }, text)
  if (!clicked) throw new Error(`no ${selector} reads ${text}`)
}
const waitText = (selector, text, timeout = 30000) => page.waitForFunction((s, t) => [...document.querySelectorAll(s)].some((node) => node.textContent.includes(t)), { timeout }, selector, text)
const texts = (selector) => page.$$eval(selector, (nodes) => nodes.map((node) => node.textContent.trim()))
const step = async (name) => {
  summary.steps.push({ step: name, folder_status: await texts(`${folderItem} .knowledge-folder__status`).catch(() => []), documents: (await texts(`${library} .knowledge-docs .knowledge-doc__name`)).sort() })
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded', timeout: 60000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 120000 })
  await page.click('button[aria-label="Attach"]')
  await page.waitForSelector('button[aria-label="Knowledge collections"]', { timeout: 5000 })
  await page.click('button[aria-label="Knowledge collections"]')
  await page.waitForSelector(library, { timeout: 5000 })
  await setValue('input[aria-label="New collection name"]', 'Policies')
  await page.click('button[aria-label="Create collection"]')
  await waitText('.knowledge-collection', 'Policies')

  await clickText(`${library} .knowledge-folder-add`, 'Watch a folder')
  await setValue('#knowledge-folder-path', parent)
  await clickText(`${library} .knowledge-folder-form button`, 'Browse')
  await waitText(`${library} .knowledge-browser`, 'watched')
  await shot('1-browse.png')
  await clickText(`${library} .knowledge-browser__entry`, 'watched')
  await page.waitForFunction((want) => document.querySelector('#knowledge-folder-path')?.value === want, { timeout: 10000 }, watched)
  await clickText(`${library} .knowledge-folder-form button`, 'Watch into Policies')
  await waitText(`${folderItem} .knowledge-folder__status`, '3 documents.')
  await waitText(`${library} .knowledge-docs`, 'team/notes.md')
  await page.$eval(`${folderItem} .knowledge-folder__skipped`, (node) => { node.open = true })
  await step('watched')
  await shot('2-watched.png')

  // Change the folder on disk and let the server's timer find it.
  appendFileSync(join(watched, 'support-policy.txt'), '\n7. Refund appeals\n\nA refused refund can be appealed once; appeals are decided within ten business days.\n')
  rmSync(join(watched, 'escalation-guide.txt'))
  writeFileSync(join(watched, 'onboarding.md'), '# Onboarding\n\nNew support engineers shadow the queue for two weeks.\n')
  const changed = Date.now()
  await waitText(`${folderItem} .knowledge-folder__status`, 'Last check: 1 added, 1 updated, 1 removed.', 75000)
  await waitText(`${library} .knowledge-docs`, 'onboarding.md')
  summary.seconds_until_ui_showed_change = Math.round((Date.now() - changed) / 100) / 10
  await step('changed on disk')
  await shot('3-changed-on-disk.png')

  await page.click(`button[aria-label="Stop watching ${watched}"]`)
  await waitText(`${library} .knowledge-confirm`, 'The files on disk are not touched.')
  await shot('4-stop-watching.png')
  await clickText(`${library} .knowledge-confirm button`, 'Stop watching')
  await page.waitForFunction((s) => !document.querySelector(s), { timeout: 10000 }, folderItem)
  await pause(1000)
  await step('stopped')
} finally {
  summary.page_errors = pageErrors
  writeFileSync(join(outDir, 'capture-folders.json'), `${JSON.stringify(summary, null, 1)}\n`)
  await browser.close()
}
console.log(JSON.stringify(summary.steps))
