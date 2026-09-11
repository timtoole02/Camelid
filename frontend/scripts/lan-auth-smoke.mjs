#!/usr/bin/env node
/* Real-browser acceptance for a fresh browser opening Camelid through an
 * authenticated LAN listener. Requires `npm run build` first. */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { launchBrowser } from './lib/launch-browser.mjs'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const distDir = resolve(scriptDir, '../dist')
const ledgerPath = resolve(scriptDir, '../../ledger/camelid-ledger.json')
const captureDir = process.env.CAMELID_CAPTURE_LAN_RESPONSIVE === '1'
  ? resolve(scriptDir, '../design-evidence/lan-chat-responsive')
  : null
const API_KEY = 'lan-browser-key-9f72'
const MODEL_FILENAME = 'Qwen3-0.6B-Q8_0.gguf'
const SECOND_MODEL_FILENAME = 'Qwen3-1.7B-Q8_0.gguf'
const RESPONSIVE_VIEWPORTS = [
  { name: 'compact-portrait', width: 360, height: 800, compact: true },
  { name: 'phone-portrait', width: 390, height: 844, compact: true },
  { name: 'narrow-phone-landscape', width: 480, height: 320, compact: true },
  { name: 'phone-landscape', width: 844, height: 390, compact: true },
  { name: 'phone-landscape-short', width: 740, height: 320, compact: true },
  { name: 'tablet-portrait', width: 768, height: 1024, compact: true },
  { name: 'desktop', width: 1440, height: 900, compact: false },
]
const MIME = {
  '.css': 'text/css',
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
}

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'))
const capabilities = {
  ...ledger.capabilities,
  model_compatibility: ledger.model_rows.map((row) => row.contract),
}
const observed = []
const foreignRequests = []
const pageErrors = []
const chatRequests = []
const modelSwitchRequests = []
let activeModelFilename = MODEL_FILENAME
let releaseSecondChatFrame = null

function sendJson(res, status, body) {
  const payload = JSON.stringify(body)
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload),
  })
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
  if (!chunks.length) return null
  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

async function protectedAnswer(path, req, res) {
  if (path === '/v1/models') {
    return sendJson(res, 200, {
      object: 'list',
      data: [MODEL_FILENAME, SECOND_MODEL_FILENAME].map((id) => ({ id, object: 'model' })),
    })
  }
  if (path === '/api/capabilities') return sendJson(res, 200, capabilities)
  if (path === '/api/models/catalog/downloads') return sendJson(res, 200, [])
  if (path === '/api/models/local') {
    return sendJson(res, 200, {
      models_dir: 'models',
      models: [MODEL_FILENAME, SECOND_MODEL_FILENAME].map((filename, index) => ({
        filename,
        size_bytes: index === 0 ? 639446688 : 1800000000,
        architecture: 'qwen3',
        quantization: 'Q8_0',
        admitted: true,
        oracle_qualified: true,
        chat_capable: true,
        generation_capable: true,
        lane_class: 'supported',
      })),
    })
  }
  if (path === '/api/models/current') return sendJson(res, 200, { path: `models/${activeModelFilename}` })
  if (path === '/api/models/catalog') return sendJson(res, 200, { items: [], next_cursor: null })
  if (path === '/api/models/load' && req.method === 'POST') {
    const body = await readJsonBody(req)
    modelSwitchRequests.push(body)
    if (body?.filename !== SECOND_MODEL_FILENAME || body?.replace !== true) {
      return sendJson(res, 403, { error: { code: 'lan_model_not_local', message: 'unsafe model selection' } })
    }
    activeModelFilename = SECOND_MODEL_FILENAME
    return sendJson(res, 200, {
      id: SECOND_MODEL_FILENAME,
      path: `models/${SECOND_MODEL_FILENAME}`,
      generation_ready: true,
      loaded_now: true,
      quantization: 'Q8_0',
    })
  }
  if (path === '/v1/chat/completions' && req.method === 'POST') {
    const body = await readJsonBody(req)
    chatRequests.push(body)
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
    })
    res.socket?.setNoDelay(true)
    res.flushHeaders()
    res.write('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
    res.write('data: {"choices":[{"delta":{"content":"LAN "}}]}\n\n')
    await new Promise((done) => { releaseSecondChatFrame = done })
    res.write('data: {"choices":[{"delta":{"content":"reply"}}]}\n\n')
    res.write('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
    res.write('data: [DONE]\n\n')
    return res.end()
  }
  return sendJson(res, 404, { error: { code: 'not_found', message: 'not found' } })
}

const appServer = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://127.0.0.1').pathname
  const apiKey = req.headers['x-api-key'] || ''
  const protectedRoute = path.startsWith('/api/') || (path.startsWith('/v1/') && path !== '/v1/health')
  observed.push({ method: req.method, path, apiKey })

  if (path === '/v1/health') {
    return sendJson(res, 200, {
      ok: true,
      engine: 'camelid',
      api_surface: 'lan_chat_only',
      version: 'lan-auth-smoke',
      build: 'lan-auth-smoke',
      backend: 'llama',
      loaded_now: true,
      generation_ready: true,
      active_model_id: activeModelFilename,
    })
  }
  if (protectedRoute) {
    if (apiKey !== API_KEY) {
      res.setHeader('WWW-Authenticate', 'Bearer realm="camelid"')
      return sendJson(res, 401, { error: { code: 'unauthorized', message: 'API key required' } })
    }
    return protectedAnswer(path, req, res)
  }

  const relative = path === '/' ? 'index.html' : path.replace(/^\//, '')
  const filePath = join(distDir, relative)
  if (filePath.startsWith(distDir) && existsSync(filePath)) return sendFile(res, filePath)
  return sendFile(res, join(distDir, 'index.html'))
})

const foreignServer = createServer((req, res) => {
  foreignRequests.push({ path: req.url, apiKey: req.headers['x-api-key'] || '' })
  res.writeHead(204, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'x-api-key',
  })
  res.end()
})

await new Promise((done) => appServer.listen(0, '127.0.0.1', done))
await new Promise((done) => foreignServer.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${appServer.address().port}`
const foreignOrigin = `http://127.0.0.1:${foreignServer.address().port}`

const browser = await launchBrowser({ purpose: 'the LAN authentication smoke', headless: 'new' })
const page = await browser.newPage()
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 })
page.on('pageerror', (error) => pageErrors.push(String(error)))
await page.evaluateOnNewDocument(() => {
  if (window.sessionStorage.getItem('camelid.lanAuthSmokeInitialized')) return
  window.localStorage.clear()
  window.sessionStorage.setItem('camelid.lanAuthSmokeInitialized', 'true')
})

async function saveKey(value) {
  const input = await page.waitForSelector('input[type="password"]')
  await input.click({ count: 3 })
  await page.keyboard.press('Backspace')
  await input.type(value)
  await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((candidate) => candidate.textContent.trim() === 'Save key')
    if (!button) throw new Error('Save key button is missing')
    button.click()
  })
}

async function measureResponsiveLayout(viewport) {
  await page.setViewport({ width: viewport.width, height: viewport.height, deviceScaleFactor: 1 })
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  return page.evaluate(() => {
    const rect = (selector) => {
      const bounds = document.querySelector(selector)?.getBoundingClientRect()
      return bounds && {
        top: Math.round(bounds.top),
        right: Math.round(bounds.right),
        bottom: Math.round(bounds.bottom),
        left: Math.round(bounds.left),
        width: Math.round(bounds.width),
        height: Math.round(bounds.height),
      }
    }
    const noticeText = document.querySelector('.cx-notice__text')
    const noticeTextStyle = noticeText ? window.getComputedStyle(noticeText) : null
    return {
      innerWidth: window.innerWidth,
      innerHeight: window.innerHeight,
      scrollWidth: document.documentElement.scrollWidth,
      topbar: rect('.topbar'),
      notice: rect('.camelid-notice-slot'),
      noticeTextStyle: noticeTextStyle && {
        display: noticeTextStyle.display,
        lineClamp: noticeTextStyle.webkitLineClamp,
        textOverflow: noticeTextStyle.textOverflow,
        whiteSpace: noticeTextStyle.whiteSpace,
      },
      toolbar: rect('.cxcomposer__toolbar'),
      composer: rect('.cxcomposer'),
      composerBox: rect('.cxcomposer__box'),
      composerInput: rect('.cxcomposer__input'),
      composerStatus: rect('.cxcomposer__status'),
      chatScroll: rect('.cxchat__scroll'),
      modelSelect: rect('.cxcomposer__model-select'),
      send: rect('.cxcomposer__send'),
    }
  })
}

async function assertResponsiveLayout() {
  const measurements = []
  for (const viewport of RESPONSIVE_VIEWPORTS) {
    const layout = await measureResponsiveLayout(viewport)
    measurements.push({ name: viewport.name, ...layout })
    assert.ok(layout.scrollWidth <= layout.innerWidth + 1, `${viewport.name} document overflow: ${JSON.stringify(layout)}`)
    assert.ok(layout.toolbar && layout.composer && layout.chatScroll && layout.modelSelect && layout.send, `${viewport.name} is missing a chat control`)
    assert.ok(layout.send.right <= layout.innerWidth && layout.send.left >= 0, `${viewport.name} send control is clipped: ${JSON.stringify(layout)}`)
    assert.ok(layout.modelSelect.right <= layout.innerWidth && layout.modelSelect.left >= 0, `${viewport.name} model control is clipped: ${JSON.stringify(layout)}`)
    assert.ok(layout.composer.bottom <= layout.innerHeight + 1, `${viewport.name} composer is below the viewport: ${JSON.stringify(layout)}`)
    if (viewport.compact) {
      assert.ok(layout.toolbar.height <= 44, `${viewport.name} composer tools wrapped into extra rows: ${JSON.stringify(layout)}`)
      assert.ok(layout.chatScroll.height >= 120, `${viewport.name} leaves too little room for messages: ${JSON.stringify(layout)}`)
      if (layout.notice && viewport.width <= 480 && viewport.height > viewport.width) {
        assert.ok(layout.notice.height <= 72, `${viewport.name} notice consumes too much chat height: ${JSON.stringify(layout)}`)
      }
      if (layout.notice && viewport.height <= 500 && viewport.width > viewport.height) {
        assert.ok(layout.notice.height <= 36, `${viewport.name} landscape notice is not compact: ${JSON.stringify(layout)}`)
      }
      if (viewport.name === 'narrow-phone-landscape') {
        assert.deepEqual(
          layout.noticeTextStyle,
          { display: 'block', lineClamp: 'none', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
          `${viewport.name} notice must use one-line ellipsis instead of the portrait line clamp`,
        )
      }
    }
  }
  return measurements
}

async function captureResponsiveEvidence() {
  if (!captureDir) return
  mkdirSync(captureDir, { recursive: true })
  for (const [viewportName, filename] of [
    ['compact-portrait', '01-mobile-portrait.png'],
    ['phone-landscape-short', '02-mobile-landscape-short.png'],
  ]) {
    const viewport = RESPONSIVE_VIEWPORTS.find((candidate) => candidate.name === viewportName)
    await page.setViewport({ width: viewport.width, height: viewport.height, deviceScaleFactor: 1 })
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
    await page.waitForFunction(() => {
      const app = document.querySelector('.camelid-app')
      const rail = document.querySelector('#camelid-sidebar')?.getBoundingClientRect()
      return !app?.classList.contains('is-mobile-open') && rail && rail.right <= 1
    })
    await page.screenshot({ path: join(captureDir, filename) })
  }
}

async function assertCompactInteractions(viewport) {
  await page.setViewport({ width: viewport.width, height: viewport.height, deviceScaleFactor: 1 })
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  await page.waitForFunction(() => {
    const app = document.querySelector('.camelid-app')
    const rail = document.querySelector('#camelid-sidebar')?.getBoundingClientRect()
    return !app?.classList.contains('is-mobile-open') && rail && rail.right <= 1
  })

  const toggleHitTarget = await page.$eval('button[aria-label="Toggle sidebar"]', (button) => {
    const bounds = button.getBoundingClientRect()
    const centerX = bounds.left + bounds.width / 2
    const centerY = bounds.top + bounds.height / 2
    const hit = document.elementFromPoint(centerX, centerY)
    return {
      left: Math.round(bounds.left),
      top: Math.round(bounds.top),
      right: Math.round(bounds.right),
      bottom: Math.round(bounds.bottom),
      disabled: button.disabled,
      hitLabel: hit?.closest('button')?.getAttribute('aria-label') || null,
    }
  })
  assert.equal(toggleHitTarget.disabled, false, `${viewport.name} drawer toggle is disabled: ${JSON.stringify(toggleHitTarget)}`)
  assert.equal(toggleHitTarget.hitLabel, 'Toggle sidebar', `${viewport.name} drawer toggle is not reachable: ${JSON.stringify(toggleHitTarget)}`)
  await page.click('button[aria-label="Toggle sidebar"]')
  await page.waitForFunction(() => document.querySelector('.camelid-app')?.classList.contains('is-mobile-open'))
  await page.waitForFunction(() => Math.abs(document.querySelector('#camelid-sidebar')?.getBoundingClientRect().left || 0) <= 1)
  const drawer = await page.$eval('#camelid-sidebar', (node) => {
    const bounds = node.getBoundingClientRect()
    return { left: Math.round(bounds.left), right: Math.round(bounds.right), width: Math.round(bounds.width) }
  })
  assert.ok(Math.abs(drawer.left) <= 1, `${viewport.name} drawer should settle against the left edge: ${JSON.stringify(drawer)}`)
  assert.ok(drawer.right <= viewport.width, `${viewport.name} drawer exceeds the viewport: ${JSON.stringify(drawer)}`)
  assert.ok(drawer.right < viewport.width, `${viewport.name} drawer leaves no reachable close scrim: ${JSON.stringify(drawer)}`)
  await page.click('button[aria-label="Close navigation"]', {
    offset: {
      x: Math.round(drawer.right + (viewport.width - drawer.right) / 2),
      y: Math.round(viewport.height / 2),
    },
  })
  await page.waitForFunction(() => !document.querySelector('.camelid-app')?.classList.contains('is-mobile-open'))

  const toolReachability = await page.evaluate(() => {
    const tools = document.querySelector('.cxcomposer__tools')
    const model = tools?.querySelector('.cxcomposer__model-select')
    const control = tools?.querySelector('button[aria-label="Generation controls"]')
    if (!tools || !model || !control) return null
    tools.scrollLeft = tools.scrollWidth
    const toolsRect = tools.getBoundingClientRect()
    const modelRect = model.getBoundingClientRect()
    const controlRect = control.getBoundingClientRect()
    return {
      scrollLeft: tools.scrollLeft,
      toolsLeft: Math.round(toolsRect.left),
      toolsRight: Math.round(toolsRect.right),
      modelLeft: Math.round(modelRect.left),
      modelRight: Math.round(modelRect.right),
      controlLeft: Math.round(controlRect.left),
      controlRight: Math.round(controlRect.right),
    }
  })
  assert.ok(toolReachability, `${viewport.name} compact composer controls are missing`)
  assert.ok(toolReachability.modelLeft >= toolReachability.toolsLeft - 1, `${viewport.name} model selector scrolled out of view: ${JSON.stringify(toolReachability)}`)
  assert.ok(toolReachability.modelRight <= toolReachability.toolsRight + 1, `${viewport.name} model selector is clipped: ${JSON.stringify(toolReachability)}`)
  assert.ok(toolReachability.controlLeft >= toolReachability.toolsLeft, `${viewport.name} last composer control cannot scroll into view: ${JSON.stringify(toolReachability)}`)
  assert.ok(toolReachability.controlRight <= toolReachability.toolsRight + 1, `${viewport.name} last composer control remains clipped: ${JSON.stringify(toolReachability)}`)
  assert.ok(toolReachability.modelRight <= toolReachability.controlLeft, `${viewport.name} compact composer controls overlap: ${JSON.stringify(toolReachability)}`)
  await page.$eval('button[aria-label="Generation controls"]', (button) => button.click())
  await page.waitForSelector('.chat-controls')
  await page.click('.chat-controls__head button')
  await page.waitForFunction(() => !document.querySelector('.chat-controls'))
}

try {
  await page.goto(origin, { waitUntil: 'domcontentloaded' })
  await page.waitForSelector('main[data-view="settings"]', { timeout: 30000 })
  await page.waitForFunction(() => document.body.textContent.includes('API key required'))

  assert.equal(
    await page.evaluate(() => document.activeElement?.matches('input[type="password"]')),
    true,
    'a fresh unauthorized browser should focus the API-key field',
  )
  assert.match(
    await page.$eval('#api-key-required-message', (node) => node.textContent),
    /online and requires the laptop's LAN Chat key/i,
    'reachable-but-protected must not be described as an offline server',
  )
  // Anchored on the card itself, not its wording: a text match on the old title
  // kept passing once the card was renamed, whether or not it leaked.
  assert.equal(
    await page.evaluate(() => Boolean(
      document.querySelector('.settings-cluster-card')
      || [...document.querySelectorAll('main[data-view="settings"] *')]
        .some((node) => node.childElementCount === 0 && /^Cluster( Topology)?$/.test(node.textContent.trim())),
    )),
    false,
    'restricted Settings must not expose the Cluster card before the key is accepted',
  )
  assert.ok(
    observed.some((request) => request.path === '/v1/models' && request.apiKey === ''),
    'the initial protected request must actually be refused without a key',
  )

  await saveKey('wrong-key')
  await page.waitForFunction(() => document.body.textContent.includes('needs the server API key'))
  assert.ok(
    observed.some((request) => request.path === '/v1/models' && request.apiKey === 'wrong-key'),
    'the wrong-key arm must reach the server and remain in credential setup',
  )
  assert.ok(await page.$('main[data-view="settings"]'), 'a wrong key must not unlock Chat')

  await saveKey(API_KEY)
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  await page.waitForSelector('textarea[aria-label="Message Camelid"]:not([disabled])', { timeout: 30000 })
  assert.ok(
    observed.some((request) => request.path === '/v1/models' && request.apiKey === API_KEY),
    'the accepted key must authenticate the protected dashboard request',
  )
  assert.equal(
    await page.evaluate(() => window.localStorage.getItem('camelid.apiKey')),
    API_KEY,
    'the accepted key should persist only in this browser',
  )

  const modelSwitchResponse = page.waitForResponse(
    (response) => response.url() === `${origin}/api/models/load` && response.status() === 200,
    { timeout: 30000 },
  )
  await page.select('select[aria-label="Choose model for chat"]', SECOND_MODEL_FILENAME)
  await modelSwitchResponse
  await page.waitForFunction((filename) => (
    document.querySelector('select[aria-label="Choose model for chat"]')?.value === filename
  ), { timeout: 30000 }, SECOND_MODEL_FILENAME)
  assert.equal(modelSwitchRequests.length, 1, 'one mobile model choice should produce one host switch request')
  assert.equal(modelSwitchRequests[0].filename, SECOND_MODEL_FILENAME)
  assert.equal(modelSwitchRequests[0].replace, true)
  assert.ok(
    observed.some((request) => request.path === '/api/models/load' && request.apiKey === API_KEY),
    'the browser must authenticate the model switch request',
  )

  await page.type('textarea[aria-label="Message Camelid"]:not([disabled])', 'Answer from the laptop')
  await page.click('button[aria-label="Send message"]')
  try {
    await page.waitForFunction(() => {
      const reply = document.querySelector('.cxturn--assistant .cxturn__body')?.textContent || ''
      return reply.includes('LAN')
    }, { timeout: 30000 })
  } catch (error) {
    const browserState = await page.evaluate(() => ({
      activeView: document.querySelector('main')?.getAttribute('data-view') || null,
      assistantTurns: [...document.querySelectorAll('.cxturn--assistant')].map((turn) => ({
        text: turn.textContent,
        streaming: turn.matches('.is-streaming'),
      })),
      sendDisabled: document.querySelector('button[aria-label="Send message"]')?.disabled ?? null,
    }))
    throw new Error(`first LAN stream frame was not rendered: ${JSON.stringify({
      chatRequestCount: chatRequests.length,
      hostWaitingAfterFirstFrame: Boolean(releaseSecondChatFrame),
      observedTail: observed.slice(-8).map(({ method, path, apiKey }) => ({ method, path, authenticated: Boolean(apiKey) })),
      pageErrors,
      browserState,
    })}`, { cause: error })
  }
  assert.equal(
    await page.evaluate(() => document.querySelector('.cxturn--assistant .cxturn__body')?.textContent.includes('LAN reply')),
    false,
    'the first streamed frame should render before the host sends the second frame',
  )
  assert.ok(releaseSecondChatFrame, 'the host should be waiting to send the second frame')
  releaseSecondChatFrame()
  releaseSecondChatFrame = null
  await page.waitForFunction(() => {
    const reply = document.querySelector('.cxturn--assistant .cxturn__body')?.textContent || ''
    return reply.includes('LAN reply')
  }, { timeout: 30000 })
  assert.equal(chatRequests.length, 1, 'one mobile send should produce one host chat request')
  assert.equal(chatRequests[0].stream, true, 'the mobile WebUI should request the streaming path')
  assert.equal(chatRequests[0].model, SECOND_MODEL_FILENAME, 'Chat must use the switched laptop runtime model identity')
  assert.equal(chatRequests[0].messages.at(-1)?.content, 'Answer from the laptop')
  assert.ok(
    observed.some((request) => request.path === '/v1/chat/completions' && request.apiKey === API_KEY),
    'the browser must authenticate the host-side inference request',
  )
  assert.equal(
    observed.some((request) => request.path === '/api/telemetry/stream'),
    false,
    'LAN Chat must not start a credential-less EventSource reconnect loop',
  )
  const navigationLabels = await page.evaluate(() => [...document.querySelectorAll('#camelid-sidebar button')]
    .map((button) => button.getAttribute('aria-label') || button.textContent.trim()))
  for (const forbidden of ['Workspace', 'Models', 'Downloaded models', 'Analytics', 'Telemetry', 'System', 'API', 'Compatibility', 'Cluster']) {
    assert.equal(navigationLabels.includes(forbidden), false, `${forbidden} should not be offered by a LAN Chat-only listener`)
  }
  await page.keyboard.down('Control')
  await page.keyboard.press('KeyK')
  await page.keyboard.up('Control')
  await page.waitForSelector('.palette[role="dialog"]')
  const paletteLabels = await page.$$eval('.palette__label', (nodes) => nodes.map((node) => node.textContent.trim()))
  for (const forbidden of ['Workspace', 'Models', 'Analytics', 'Telemetry', 'System', 'API', 'Compatibility', 'Cluster', 'Observatory']) {
    assert.equal(paletteLabels.includes(forbidden), false, `${forbidden} should not be offered by the LAN Chat command palette`)
  }
  assert.ok(paletteLabels.includes('Chat'), 'the LAN Chat command palette should retain Chat')
  assert.ok(paletteLabels.includes('Settings'), 'the LAN Chat command palette should retain Settings')
  await page.keyboard.press('Escape')

  const responsiveMeasurements = await assertResponsiveLayout()
  await captureResponsiveEvidence()
  await assertCompactInteractions(RESPONSIVE_VIEWPORTS.find((viewport) => viewport.name === 'compact-portrait'))
  await assertCompactInteractions(RESPONSIVE_VIEWPORTS.find((viewport) => viewport.name === 'phone-landscape'))

  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 })
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))))
  await page.evaluate((url) => fetch(url), `${foreignOrigin}/foreign-origin-negative-control`)
  await page.waitForFunction(() => true)
  assert.deepEqual(
    foreignRequests,
    [{ path: '/foreign-origin-negative-control', apiKey: '' }],
    'the Camelid API key must not be attached to an unrelated origin',
  )

  const geometry = await page.evaluate(() => ({
    width: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  assert.ok(geometry.scrollWidth <= geometry.width + 1, `mobile page overflows: ${JSON.stringify(geometry)}`)

  const acceptedCountBeforeReload = observed.filter((request) => request.path === '/v1/models' && request.apiKey === API_KEY).length
  const authenticatedReload = page.waitForResponse(
    (response) => response.url() === `${origin}/v1/models` && response.status() === 200,
    { timeout: 30000 },
  )
  await page.reload({ waitUntil: 'domcontentloaded' })
  await authenticatedReload
  await page.waitForSelector('main[data-view="chat"]', { timeout: 30000 })
  const acceptedCountAfterReload = observed.filter((request) => request.path === '/v1/models' && request.apiKey === API_KEY).length
  assert.ok(acceptedCountAfterReload > acceptedCountBeforeReload, 'reload should reuse the browser-held key')
  assert.equal(pageErrors.length, 0, `browser errors: ${pageErrors.join('\n')}`)

  console.log('LAN_AUTH_SMOKE_PASS')
  console.log(JSON.stringify({ protectedRequests: observed.filter((request) => request.apiKey).length, viewport: geometry, responsiveMeasurements }))
} finally {
  releaseSecondChatFrame?.()
  await page.close().catch(() => {})
  await browser.close().catch(() => {})
  await new Promise((done) => appServer.close(done))
  await new Promise((done) => foreignServer.close(done))
}