#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { resolve } from 'node:path'
import { launchBrowser } from './lib/launch-browser.mjs'

const base = process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const output = resolve(process.env.CAMELID_CAPTURE_DIR || '../target/retro-transition')
await mkdir(output, { recursive: true })
const browser = await launchBrowser({ purpose: 'retro transitions', headless: true })
const page = await browser.newPage()
const errors = []
page.on('pageerror', (error) => errors.push(error.message))
await page.setViewport({ width: 1280, height: 800 })
await page.evaluateOnNewDocument(() => {
  window.EventSource = class { addEventListener() {} removeEventListener() {} close() {} }
})
await page.setRequestInterception(true)
let delayCode = 0
page.on('request', async (request) => {
  const url = new URL(request.url())
  if ((/\/assets\/CodingWorkspace-|\/src\/views\/CodingWorkspace\.jsx/.test(url.pathname)) && delayCode) await new Promise((done) => setTimeout(done, delayCode))
  if (!/^\/(api|v1|__camelid)\//.test(url.pathname)) return request.continue()
  const fixtures = {
    '/v1/health': { ok: true, engine: 'camelid', loaded_now: false, generation_ready: false },
    '/v1/models': { data: [] },
    '/api/capabilities': { model_compatibility: [], api_features: [], support_contract: {} },
    '/api/models/local': { models: [] },
    '/api/models/catalog/downloads': [],
    '/api/agent/workspace/models': { models: [] },
    '/api/agent/coding/sessions': { sessions: [], execution_engine: { id: 'retro-test', name: 'Test engine' } },
    '/api/mcp/connections': { connections: [] },
    '/__camelid/backend/status': { available: false, running: false },
  }
  return request.respond({ status: 200, contentType: 'application/json', headers: { 'Access-Control-Allow-Origin': '*' }, body: JSON.stringify(fixtures[url.pathname] || {}) })
})
const nav = async (label) => {
 await page.evaluate((text) => {
  if ((text === 'Chat' || text === 'Code') && document.querySelector('.coding-mode-switch')) {
    const mode = [...document.querySelectorAll('.coding-mode-switch button')].find(node => node.textContent.trim() === text)
    mode.click()
    return
  }
  const button = [...document.querySelectorAll('#camelid-sidebar button')].find((node) => node.getAttribute('aria-label') === text || node.textContent.trim() === text)
  if (!button) throw new Error(`Missing navigation: ${text}`)
  button.click()
 }, label)
 if (label === 'Settings') await page.waitForSelector('[aria-label="16-bit mode transitions"]')
 if (label === 'Chat' || label === 'Code') await page.waitForSelector('.coding-mode-switch')
}
const view = (tab) => page.waitForSelector(tab === 'chat' || tab === 'code' ? `.camelid-main[data-view="chat"][data-chat-mode="${tab}"]` : `.camelid-main[data-view="${tab}"]`)
const phase = (name) => page.waitForSelector(`[data-retro-phase="${name}"]`, { timeout: 4000 })
const settled = () => page.waitForFunction(() => !document.querySelector('[data-retro-phase]') && !document.querySelector('.camelid-view').inert)

try {
  await page.goto(`${base}/#settings`, { waitUntil: 'networkidle0' })
  const toggle = '[aria-label="16-bit mode transitions"]'
  await page.waitForSelector(toggle)
  assert.equal(await page.$eval(toggle, (node) => node.checked), false, 'default is off')
  await nav('Chat')
  await nav('Code')
  await view('code')
  assert.equal(await page.$('[data-retro-phase]'), null, 'default navigation has no effect')
  await nav('Settings')
  await page.click(toggle)
  assert.equal(await page.evaluate(() => localStorage.getItem('camelid.retroTransitions')), 'true')
  await page.evaluate(() => localStorage.setItem('camelid.chatMode', 'chat'))
  await page.reload({ waitUntil: 'networkidle0' })
  assert.equal(await page.$eval(toggle, (node) => node.checked), true, 'setting survives reload')
  await nav('Chat')
  delayCode = 1000
  await nav('Code')
  await phase('dive')
  await page.evaluate(() => {
    const animation = document.querySelector('.camelid-view').getAnimations()[0]
    animation.pause()
    animation.currentTime = 640
  })
  assert.equal(await page.$eval('.camelid-view', node => getComputedStyle(node).filter), 'none', 'zoom-in stays unpixelated')
  await page.screenshot({ path: `${output}/dive.png` })
  await phase('pixel')
  assert.match(await page.$eval('.camelid-view', node => node.style.filter), /-pixel/)
  await phase('waiting')
  await phase('mosaic')
  assert.match(await page.$eval('.camelid-view', (node) => node.style.filter), /url\(/)
  await page.screenshot({ path: `${output}/mosaic.png` })
  await settled()
  await view('code')
  await page.screenshot({ path: `${output}/complete.png` })
  assert.equal(await page.$eval('.camelid-view', (node) => node.style.filter), '', 'filter released')
  assert.equal(await page.$eval('.camelid-view', node => node.style.transformOrigin), '', 'pixel target released')
  assert.equal(await page.$eval('.camelid-view', (node) => node.getAnimations().length), 0, 'animation released')

  await nav('Chat')
  await phase('dive')
  await nav('Chat') // repeated destination does not restart the animation
  await phase('mosaic')
  await settled()
  await view('chat')
  await nav('Code')
  await phase('dive')
  await nav('Settings')
  await settled()
  await new Promise((done) => setTimeout(done, 750))
  await view('settings') // canceled callback cannot pull us back to Code
  await nav('Chat')
  await nav('Code')
  await phase('dive')
  await nav('New chat')
  await settled()
  await new Promise((done) => setTimeout(done, 750))
  await view('chat')
  await nav('Code')
  await phase('dive')
  await page.keyboard.press('Escape')
  await view('code')
  await settled()

  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }])
  await nav('Chat')
  await view('chat')
  assert.equal(await page.$('[data-retro-phase]'), null, 'reduced motion skips effect')
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'no-preference' }])
  await nav('Code')
  await phase('dive')
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }])
  await settled()
  await view('code')

  await page.setViewport({ width: 390, height: 844 })
  await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))))
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'no-preference' }])
  await nav('Chat')
  await phase('mosaic')
  await page.screenshot({ path: `${output}/mobile-mosaic.png` })
  await settled()
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'no horizontal overflow')
  await nav('Settings')
  await page.click(toggle)
  await page.reload({ waitUntil: 'networkidle0' })
  assert.equal(await page.$eval(toggle, (node) => node.checked), false, 'can persist opt-out')
  assert.equal(await page.$('.retro-transition-filters'), null, 'filters unmounted when off')
  assert.deepEqual(errors, [], 'no browser exceptions')
  console.log(`Retro transition smoke passed. Screenshots: ${output}`)
} finally {
  await browser.close()
}
