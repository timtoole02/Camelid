#!/usr/bin/env node
/* Component-level coverage for the offline banner's two audiences.
 *
 * The banner is the only thing a user sees when the engine dies with the tab
 * still open, and it behaved differently depending on something invisible: the
 * Vite dev-server launch hook. With the hook present it offers a working Start
 * button; in every shipped build the hook does not exist, and the banner used to
 * fall back to a lone "Settings" button that navigated to a page explaining the
 * launcher needs `npm run dev`. That is a dead end for the only population that
 * ever hits it.
 *
 * Rendering the component is not enough to prove this: the branch is chosen from
 * a live probe of `/__camelid/backend/status`, and the packaged answer is the SPA
 * HTML fallback rather than JSON. So this drives the real app in a real browser
 * against both server shapes.
 *
 * Requires `npm run build` first (it serves frontend/dist) and Chrome/Edge.
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, readFileSync } from 'node:fs'
import { extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { launchBrowser } from './lib/launch-browser.mjs'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const distDir = resolve(scriptDir, '../dist')

if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

/* The exact sentence the packaged build used to show. Asserting its ABSENCE is
   what keeps the dead end from coming back. */
const RETIRED_COPY = 'Start it from a terminal, then this clears automatically.'

let stub
function resetStub() {
  stub = {
    // 'packaged' = no launcher hook, so /__camelid/* falls through to the SPA
    // HTML exactly as the real engine serves it. 'dev' = the Vite plugin.
    mode: 'packaged',
    detected: 'cargo run --release -- serve',
    requests: [],
  }
}
resetStub()

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.json': 'application/json', '.png': 'image/png' }

function sendJson(res, status, body) {
  const payload = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) })
  res.end(payload)
}

function sendIndex(res) {
  const body = readFileSync(join(distDir, 'index.html'))
  res.writeHead(200, { 'Content-Type': 'text/html' })
  res.end(body)
}

const server = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://127.0.0.1').pathname
  stub.requests.push({ method: req.method, path })

  // The engine is gone: health fails, which is what drives runtime.status to
  // 'offline' (see hooks/useDashboardData.js).
  if (path === '/v1/health') return sendJson(res, 503, { ok: false })

  if (path.startsWith('/__camelid/backend/')) {
    if (stub.mode !== 'dev') return sendIndex(res) // SPA fallback -> JSON.parse throws -> unavailable
    if (path.endsWith('/status')) {
      return sendJson(res, 200, { available: true, running: false, detected: stub.detected, logTail: '' })
    }
    return sendJson(res, 200, { available: true, running: true, pid: 4242, logTail: '' })
  }

  if (path.startsWith('/api/') || path.startsWith('/v1/')) {
    return sendJson(res, 503, { error: { message: 'engine offline' } })
  }

  const filePath = join(distDir, path === '/' ? 'index.html' : path.replace(/^\//, ''))
  if (existsSync(filePath) && !filePath.includes('..')) {
    const body = readFileSync(filePath)
    res.writeHead(200, { 'Content-Type': MIME[extname(filePath)] || 'application/octet-stream' })
    return res.end(body)
  }
  return sendIndex(res)
})

await new Promise((done) => server.listen(0, '127.0.0.1', done))
const origin = `http://127.0.0.1:${server.address().port}`

const browser = await launchBrowser({ purpose: 'the offline banner smoke', headless: 'new' })
const pageErrors = []

async function openBanner({ clipboard = 'ok', storage = {}, platform = null, desktopShell = null } = {}) {
  const page = await browser.newPage()
  page.on('pageerror', (error) => pageErrors.push(String(error)))
  // Nothing may leave the harness. One scenario points the app at an engine on
  // another host; without this the probe sits in TCP retries (measured: 44s for
  // the suite, versus ~7s with it) and the smoke would depend on the state of
  // whatever network the runner is attached to.
  await page.setRequestInterception(true)
  page.on('request', (request) => {
    if (request.url().startsWith(origin)) request.continue()
    else request.abort()
  })
  // copyText() calls navigator.clipboard.writeText. Drive all three real shapes:
  // present and working, ABSENT (what a non-secure context gives you, e.g. a
  // plain-http LAN address), and present but rejecting (permission denied).
  await page.evaluateOnNewDocument((mode, seed, platformOverride, desktopShell) => {
    window.__copied = []
    /* The banner's wording is platform-aware: only Windows users are told to run
       camelid.exe. Scenarios that assert that phrasing must say which platform
       they are standing on rather than inheriting the runner's. */
    if (platformOverride) Object.defineProperty(navigator, 'platform', { configurable: true, value: platformOverride })
    /* Desktop shell stub: the banner detects Tauri by window.__TAURI__.core.invoke
       and, inside the app, offers to restart the sidecar itself. Record the
       commands so the scenario can prove which one it calls. */
    if (desktopShell) {
      window.__invoked = []
      window.__TAURI__ = {
        core: {
          invoke: (cmd) => {
            window.__invoked.push(cmd)
            return desktopShell === 'failing' ? Promise.reject(new Error('nope')) : Promise.resolve()
          },
        },
      }
    }
    // Pages share the browser profile, so a key seeded by an earlier scenario
    // would survive into the next one and silently test the wrong thing.
    window.localStorage.clear()
    for (const [key, value] of Object.entries(seed)) window.localStorage.setItem(key, value)
    if (mode === 'absent') {
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined })
      return
    }
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: (text) => {
          if (mode === 'denied') return Promise.reject(new Error('denied'))
          window.__copied.push(text)
          return Promise.resolve()
        },
      },
    })
  }, clipboard, storage, platform, desktopShell)
  // NOT networkidle2: one scenario points the app at an engine on another host,
  // whose probe never settles, and the banner does not depend on it.
  await page.goto(origin, { waitUntil: 'domcontentloaded' })
  await page.waitForSelector('.backend-banner', { timeout: 30000 })
  return page
}

const clickBannerButton = async (page, needle) => {
  const handle = await page.evaluateHandle((text) =>
    [...document.querySelectorAll('.backend-banner__actions button')].find((b) => b.textContent.includes(text)), needle)
  const element = handle.asElement()
  assert.ok(element, `no banner button matching ${needle}`)
  await element.click()
}

const readBanner = () => {
  const banner = document.querySelector('.backend-banner')
  const code = banner?.querySelector('.backend-banner__copy code')
  // Geometry, not scrollWidth: an overflowing child does not necessarily make
  // its ancestor scrollable (ancestors clip or hide), so scrollWidth silently
  // reports "fine" for text that visibly escapes its box.
  let escapesBox = false
  if (code && banner) {
    const codeRect = code.getBoundingClientRect()
    const bannerRect = banner.getBoundingClientRect()
    escapesBox = codeRect.right > bannerRect.right + 1 || codeRect.left < bannerRect.left - 1
  }
  return {
    text: banner?.textContent?.trim() || '',
    code: code?.textContent?.trim() || null,
    buttons: [...(banner?.querySelectorAll('.backend-banner__actions button') || [])].map((b) => b.textContent.trim()),
    escapesBox,
    // The failure label is much longer than the idle one; at 390px it is the
    // most likely thing in this banner to burst its container.
    buttonOverflow: [...(banner?.querySelectorAll('.backend-banner__actions button') || [])]
      .some((b) => b.scrollWidth > b.clientWidth + 1),
    documentOverflows: document.documentElement.scrollWidth > window.innerWidth + 1,
  }
}

const results = []
function check(name, fn) { fn(); results.push(name) }

try {
  /* ---- Scenario 1: a downloaded build. No launcher hook exists. ---- */
  resetStub()
  let page = await openBanner({ platform: 'Win32' })
  let banner = await page.evaluate(readBanner)

  check('packaged build offers no Start button it cannot honour', () => {
    assert.equal(banner.buttons.some((label) => label.includes('Start Camelid')), false,
      `expected no Start button, got ${JSON.stringify(banner.buttons)}`)
  })
  check('packaged build offers a real action instead', () => {
    assert.ok(banner.buttons.includes('Copy terminal command'),
      `expected a copy action, got ${JSON.stringify(banner.buttons)}`)
  })
  check('packaged build names the executable, not a PATH-dependent command', () => {
    // `camelid serve` errors with "'camelid' is not recognized" for anyone who
    // unzipped a release, so the instruction must name the file to run.
    assert.equal(banner.code, 'camelid.exe')
    assert.match(banner.text, /unzipped/i)
  })
  check('packaged build explains why the page cannot restart the engine', () => {
    assert.match(banner.text, /served by the engine/i)
  })
  await page.close()

  /* ---- Scenario 1b: the same packaged build on macOS/Linux. camelid.exe does
     not exist there, so naming it was an instruction the reader could not
     follow; the guidance must be platform-correct in both directions. ---- */
  resetStub()
  page = await openBanner({ platform: 'MacIntel' })
  const macBanner = await page.evaluate(readBanner)
  check('a non-Windows packaged build is never told to run camelid.exe', () => {
    assert.equal(macBanner.text.includes('camelid.exe'), false,
      `macOS/Linux was told to run a Windows executable: ${macBanner.text}`)
    assert.equal(macBanner.text.includes('unzipped'), false,
      'unzipped-folder phrasing is Windows-specific packaging')
  })
  check('a non-Windows packaged build still explains how to restart', () => {
    assert.match(macBanner.text, /served by the engine/i)
    assert.match(macBanner.text, /Start the Camelid app again|re-run the command/i)
  })
  await page.close()

  /* ---- Scenario 1c: inside Camelid Desktop. The app supervises the engine as a
     sidecar, so it can restart it in place; telling the reader to quit and
     reopen was handing them a chore the app could do itself. ---- */
  resetStub()
  page = await openBanner({ desktopShell: 'ok' })
  let desktopBanner = await page.evaluate(readBanner)
  check('the desktop app offers to restart the engine itself', () => {
    assert.ok(desktopBanner.buttons.some((label) => label.includes('Restart engine')),
      `expected a restart action, got ${JSON.stringify(desktopBanner.buttons)}`)
    assert.equal(/Quit and reopen/i.test(desktopBanner.text), false,
      'quit-and-reopen must not be the first thing offered when the app can restart the engine')
  })
  check('the desktop restart reassures that nothing is lost', () => {
    assert.match(desktopBanner.text, /models and settings are untouched/i)
  })
  check('the desktop app never shows terminal or executable instructions', () => {
    assert.equal(/camelid\.exe|cargo run|serve\b/i.test(desktopBanner.text), false,
      `desktop guidance leaked a terminal instruction: ${desktopBanner.text}`)
  })
  await clickBannerButton(page, 'Restart engine')
  check('restarting calls the engine-restart command, not a page reload', async () => {
    const invoked = await page.evaluate(() => window.__invoked || [])
    assert.deepEqual(invoked, ['retry_startup'], `unexpected commands: ${JSON.stringify(invoked)}`)
  })
  await page.close()

  /* Only once a restart has actually failed is quit-and-reopen the honest advice. */
  resetStub()
  page = await openBanner({ desktopShell: 'failing' })
  await clickBannerButton(page, 'Restart engine')
  await page.waitForFunction(() => /Quit and reopen/i.test(document.querySelector('.backend-banner')?.textContent || ''), { timeout: 5000 })
  desktopBanner = await page.evaluate(readBanner)
  check('a failed restart falls back to quit-and-reopen', () => {
    assert.match(desktopBanner.text, /would not restart/i)
    assert.match(desktopBanner.text, /Quit and reopen Camelid Desktop/i)
  })
  await page.close()

  /* Restore the packaged Windows context: Scenario 1 continues below. */
  resetStub()
  page = await openBanner({ platform: 'Win32' })
  banner = await page.evaluate(readBanner)
  check('the retired dead-end sentence is gone', () => {
    assert.equal(banner.text.includes(RETIRED_COPY), false)
  })

  const copyButton = await page.evaluateHandle(() =>
    [...document.querySelectorAll('.backend-banner__actions button')].find((b) => b.textContent.includes('Copy')))
  await copyButton.asElement().click()
  await page.waitForFunction(() => window.__copied.length > 0, { timeout: 5000 })
  const copied = await page.evaluate(() => window.__copied)
  check('the copy action puts a runnable command on the clipboard', () => {
    assert.deepEqual(copied, ['camelid serve'])
  })
  const confirmed = await page.evaluate(readBanner)
  check('the button confirms the copy', () => {
    assert.ok(confirmed.buttons.includes('Copied'),
      `expected a Copied confirmation, got ${JSON.stringify(confirmed.buttons)}`)
  })
  await page.close()

  /* ---- Scenario 1b: no clipboard API (a plain-http LAN address is not a
     secure context). Claiming "Copied" there would be the same lie. Run at
     390px: the failure label is the longest string this banner can show. ---- */
  resetStub()
  page = await openBanner({ clipboard: 'absent', platform: 'Win32' })
  await page.setViewport({ width: 390, height: 844 })
  await new Promise((done) => setTimeout(done, 150))
  await clickBannerButton(page, 'Copy')
  await page.waitForFunction(
    () => [...document.querySelectorAll('.backend-banner__actions button')].some((b) => b.textContent.includes('Couldn\u2019t copy')),
    { timeout: 5000 })
  banner = await page.evaluate(readBanner)
  check('an unavailable clipboard is never reported as a successful copy', () => {
    assert.equal(banner.buttons.includes('Copied'), false,
      `claimed a copy with no clipboard API: ${JSON.stringify(banner.buttons)}`)
    assert.ok(banner.buttons.some((label) => label.includes('Couldn’t copy')),
      `expected an honest failure label, got ${JSON.stringify(banner.buttons)}`)
  })
  check('the instruction stays readable when the copy failed', () => {
    assert.equal(banner.code, 'camelid.exe')
  })
  check('the failure label does not burst its button at 390px', () => {
    assert.equal(banner.buttonOverflow, false, 'a banner button overflows its box')
    assert.equal(banner.documentOverflows, false, 'the page scrolls horizontally')
  })
  await page.close()

  /* ---- Scenario 1c: clipboard present but rejecting (permission denied). ---- */
  resetStub()
  page = await openBanner({ clipboard: 'denied' })
  await clickBannerButton(page, 'Copy')
  await page.waitForFunction(
    () => [...document.querySelectorAll('.backend-banner__actions button')].some((b) => b.textContent.includes('Couldn\u2019t copy')),
    { timeout: 5000 })
  banner = await page.evaluate(readBanner)
  check('a rejected clipboard write is not reported as success', () => {
    assert.equal(banner.buttons.includes('Copied'), false)
  })
  await page.close()

  /* ---- Scenario 1d: the remote state renders the host name, which is the only
     variable-length text in this banner and therefore the only overflow risk.
     The token is deliberately unbroken (no hyphens): hyphens are line-break
     opportunities and would make this check pass for free. ---- */
  resetStub()
  const longHost = `${'verylongsubdomainsegment'.repeat(4)}.example`
  page = await openBanner({ storage: { 'camelid.apiBase': `http://${longHost}:8181` } })
  await page.setViewport({ width: 390, height: 844 })
  await new Promise((done) => setTimeout(done, 200))
  banner = await page.evaluate(readBanner)
  check('a long unbreakable host wraps instead of breaking the layout at 390px', () => {
    // Prove the scenario is actually exercising the long token before drawing
    // any conclusion from the geometry.
    assert.equal(banner.code, longHost,
      `the long host never reached the banner: ${JSON.stringify(banner.code)}`)
    assert.equal(banner.escapesBox, false, 'the host text escapes the banner box')
    assert.equal(banner.documentOverflows, false, 'the page scrolls horizontally')
  })
  await page.close()

  /* ---- Scenario 1f: the engine told us where it lives while it was up. The
     copy button must then produce a command that runs as-is, with no PATH
     assumption. A path WITHOUT spaces needs no quoting and runs unchanged in
     both cmd and PowerShell. ---- */
  resetStub()
  page = await openBanner({ storage: { 'camelid.enginePath': 'C:\\camelid\\camelid.exe' } })
  banner = await page.evaluate(readBanner)
  check('a known engine path becomes a paste-and-run command', () => {
    assert.equal(banner.code, 'C:\\camelid\\camelid.exe serve')
  })
  await clickBannerButton(page, 'Copy')
  await page.waitForFunction(() => window.__copied.length > 0, { timeout: 5000 })
  const copiedPath = await page.evaluate(() => window.__copied)
  check('the clipboard gets the real path, not the PATH-dependent command', () => {
    assert.deepEqual(copiedPath, ['C:\\camelid\\camelid.exe serve'])
  })
  await page.close()

  /* ---- Scenario 1g: a path WITH a space. PowerShell is the default Windows
     shell and will not execute a quoted string as a command without the `&`
     call operator, so the naive quoted form fails on the shell most users are
     actually in. ---- */
  resetStub()
  page = await openBanner({ storage: { 'camelid.enginePath': 'C:\\Program Files\\Camelid\\camelid.exe' } })
  banner = await page.evaluate(readBanner)
  check('a spaced Windows path is quoted AND call-operated for PowerShell', () => {
    assert.equal(banner.code, '& "C:\\Program Files\\Camelid\\camelid.exe" serve')
  })
  await page.close()

  /* ---- Scenario 1h: a non-default port MUST come back in the command. A bare
     `serve` re-binds 127.0.0.1:8181, which fails outright when something else
     owns that port and, even on success, comes up somewhere this tab is not
     looking — so the banner would never clear. Observed for real. ---- */
  resetStub()
  page = await openBanner({ storage: {
    'camelid.enginePath': 'C:\\camelid\\camelid.exe',
    'camelid.engineAddr': '127.0.0.1:8188',
  } })
  banner = await page.evaluate(readBanner)
  check('a non-default port is carried into the restart command', () => {
    assert.equal(banner.code, 'C:\\camelid\\camelid.exe serve --addr 127.0.0.1:8188')
  })
  await page.close()

  /* ---- Scenario 1i: the default port stays implicit, so the common case keeps
     the short command it had. ---- */
  resetStub()
  page = await openBanner({ storage: {
    'camelid.enginePath': 'C:\\camelid\\camelid.exe',
    'camelid.engineAddr': '127.0.0.1:8181',
  } })
  banner = await page.evaluate(readBanner)
  check('the default port is not spelled out', () => {
    assert.equal(banner.code, 'C:\\camelid\\camelid.exe serve')
  })
  await page.close()

  /* ---- Scenario 1e: the engine is on another host. `camelid serve` would be     advice for the wrong machine. The host is an RFC 2606 reserved name, so the
     fixture is unmistakably synthetic and cannot resolve. ---- */
  resetStub()
  page = await openBanner({ storage: { 'camelid.apiBase': 'http://engine.example:8181' } })
  banner = await page.evaluate(readBanner)
  check('a remote engine is not told to restart locally', () => {
    assert.equal(banner.buttons.some((label) => label.includes('Copy')), false,
      `offered a local restart for a remote engine: ${JSON.stringify(banner.buttons)}`)
    assert.equal(banner.text.includes('served by the engine'), false)
  })
  check('a remote engine names the host that is not answering', () => {
    assert.equal(banner.code, 'engine.example')
    assert.ok(banner.buttons.includes('Settings'))
  })
  await page.close()

  /* ---- Scenario 2: the dev server. The Start button must still work. ---- */
  resetStub()
  stub.mode = 'dev'
  page = await openBanner()
  await page.waitForFunction(
    () => [...document.querySelectorAll('.backend-banner__actions button')].some((b) => b.textContent.includes('Start Camelid')),
    { timeout: 15000 })
  banner = await page.evaluate(readBanner)
  check('the dev launcher still gets its one-click Start', () => {
    assert.ok(banner.buttons.some((label) => label.includes('Start Camelid')),
      `expected the Start button, got ${JSON.stringify(banner.buttons)}`)
  })

  const startButton = await page.evaluateHandle(() =>
    [...document.querySelectorAll('.backend-banner__actions button')].find((b) => b.textContent.includes('Start Camelid')))
  await startButton.asElement().click()
  const deadline = Date.now() + 10000
  while (Date.now() < deadline && !stub.requests.some((r) => r.path === '/__camelid/backend/launch')) {
    await new Promise((done) => setTimeout(done, 100))
  }
  check('Start still reaches the dev launch hook', () => {
    assert.ok(stub.requests.some((r) => r.method === 'POST' && r.path === '/__camelid/backend/launch'),
      'expected a POST to the dev launch hook')
  })
  await page.close()

  check('no page errors', () => {
    assert.deepEqual(pageErrors, [])
  })

  console.log(`offline banner smoke: ${results.length} checks passed`)
  for (const name of results) console.log(`  ok  ${name}`)
} finally {
  await browser.close()
  server.close()
}
