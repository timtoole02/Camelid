#!/usr/bin/env node
/* Captures the Cluster view's states for frontend/design-evidence/.
 *
 * Mirrors fabric-view-smoke.mjs: the same scripted proxy on its own origin, so
 * the images are of the real component reading a real answer rather than a
 * mock-up. Nodes are named by hostname on purpose -- a committed evidence
 * bundle may contain no IPv4 literal except 127.0.0.1, and the bundle privacy
 * audit checks that in strict mode.
 *
 * Requires `npm run build` first. Writes PNGs plus SHA256SUMS.
 */
import { createHash } from 'node:crypto'
import { createServer } from 'node:http'
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { launchBrowser } from './lib/launch-browser.mjs'

const scriptDir = fileURLToPath(new URL('.', import.meta.url))
const distDir = resolve(scriptDir, '../dist')
const outDir = resolve(scriptDir, '../design-evidence/fabric-cluster-view-p1')
if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)
mkdirSync(outDir, { recursive: true })

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.json': 'application/json', '.png': 'image/png', '.ico': 'image/x-icon' }

const appServer = createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1')
  const filePath = url.pathname === '/' ? '/index.html' : url.pathname
  const onDisk = join(distDir, filePath)
  if (existsSync(onDisk) && !onDisk.endsWith('/')) {
    res.writeHead(200, { 'Content-Type': MIME[extname(onDisk)] || 'application/octet-stream' })
    return res.end(readFileSync(onDisk))
  }
  // A healthy engine on the app's own origin, so the shell's offline banner
  // does not sit on top of the surface being photographed. Unrelated to the
  // fabric, which lives on the other origin below.
  if (filePath === '/v1/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    return res.end(JSON.stringify({
      ok: true,
      engine: 'camelid',
      version: '0.6.1',
      build: 'v0.6.1-267-gabc1234',
      loaded_now: true,
      generation_ready: true,
      active_model_id: 'Llama 3.2 1B Instruct',
    }))
  }
  if (filePath.startsWith('/api/') || filePath.startsWith('/v1/')) {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    return res.end('{}')
  }
  res.writeHead(200, { 'Content-Type': 'text/html' })
  return res.end(readFileSync(join(distDir, 'index.html')))
})

let mode = 'nodes'
const SUMMARY = { ok: true, service: 'camelid-fabric', version: '0.6.1', build: 'v0.6.1-267-gabc1234' }
const NODES = [
  {
    spec: { label: 'windows', host: '127.0.0.1', port: 8181 },
    status: { state: 'ready', active_model_id: 'Llama 3.2 1B Instruct', backend: 'cpu_q8_runtime_repack', version: 'v0.6.1-267', in_flight: 2, waiting: 1 },
    latency_ms: 3,
  },
  {
    spec: { label: 'studio', host: 'studio.local', port: 8181 },
    status: { state: 'not_ready', reason: 'no model loaded' },
    latency_ms: 21,
  },
  {
    spec: { label: 'workshop', host: 'workshop.local', port: 8181 },
    status: { state: 'unreachable', reason: 'cannot resolve host: failed to lookup address information' },
    latency_ms: null,
  },
]

function body() {
  if (mode === 'withheld') return { ...SUMMARY, ready: true }
  if (mode === 'empty') return { ...SUMMARY, ready: false, nodes: { total: 0, ready: 0, not_ready: 0, unreachable: 0 }, models: [], node_detail: [] }
  return {
    ...SUMMARY,
    ready: true,
    nodes: { total: 3, ready: 1, not_ready: 1, unreachable: 1 },
    models: ['Llama 3.2 1B Instruct'],
    node_detail: NODES,
  }
}

const proxyServer = createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*')
  if (new URL(req.url, 'http://127.0.0.1').pathname !== '/v1/health') {
    res.writeHead(404, { 'Content-Type': 'application/json' })
    return res.end('{}')
  }
  const payload = JSON.stringify(body())
  res.writeHead(mode === 'empty' ? 503 : 200, { 'Content-Type': 'application/json' })
  return res.end(payload)
})

const listen = (server) => new Promise((done) => server.listen(0, '127.0.0.1', () => done(server.address().port)))
const appPort = await listen(appServer)
const proxyPort = await listen(proxyServer)
const appOrigin = `http://127.0.0.1:${appPort}`
const proxyEndpoint = `127.0.0.1:${proxyPort}`

const browser = await launchBrowser({ purpose: 'the fabric view evidence capture', headless: 'new' })

async function shoot(name, { endpoint, viewport, before = null }) {
  const page = await browser.newPage()
  await page.setViewport({ ...viewport, deviceScaleFactor: 1 })
  await page.evaluateOnNewDocument((value) => {
    window.localStorage.clear()
    window.localStorage.setItem('camelid.fabricEndpoint', value)
  }, endpoint)
  await page.goto(`${appOrigin}/#cluster`, { waitUntil: 'networkidle0' })
  await page.waitForSelector('[data-testid="fabric-status"][data-phase="settled"]', { timeout: 20000 })
  if (before) await before(page)
  await new Promise((done) => setTimeout(done, 400))
  await page.screenshot({ path: join(outDir, name), fullPage: true })
  await page.close()
  console.log(`  wrote ${name}`)
}

const DESKTOP = { width: 1280, height: 900 }
const PHONE = { width: 390, height: 844 }

try {
  mode = 'nodes'
  await shoot('01-nodes-desktop.png', { endpoint: proxyEndpoint, viewport: DESKTOP })
  await shoot('02-node-detail-desktop.png', {
    endpoint: proxyEndpoint,
    viewport: DESKTOP,
    before: async (page) => {
      await page.click('.fabric-row[data-node-label="studio"]')
      await page.waitForSelector('.fabric-detail', { timeout: 5000 })
    },
  })

  mode = 'withheld'
  await shoot('03-detail-withheld-desktop.png', { endpoint: proxyEndpoint, viewport: DESKTOP })

  mode = 'empty'
  await shoot('04-empty-fabric-desktop.png', { endpoint: proxyEndpoint, viewport: DESKTOP })

  await shoot('05-no-proxy-desktop.png', { endpoint: '127.0.0.1:9', viewport: DESKTOP })

  mode = 'nodes'
  await shoot('06-nodes-mobile-390.png', { endpoint: proxyEndpoint, viewport: PHONE })

  const sums = readdirSync(outDir)
    .filter((name) => name.endsWith('.png'))
    .sort()
    .map((name) => `${createHash('sha256').update(readFileSync(join(outDir, name))).digest('hex')}  ${name}`)
    .join('\n')
  writeFileSync(join(outDir, 'SHA256SUMS'), `${sums}\n`)
  console.log(`\nwrote SHA256SUMS for ${sums.split('\n').length} images`)
} finally {
  await browser.close()
  appServer.close()
  proxyServer.close()
}
