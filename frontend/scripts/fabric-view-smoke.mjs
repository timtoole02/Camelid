#!/usr/bin/env node
/* Browser coverage for the Cluster view against a scripted fabric proxy.
 *
 * The pure rules are proved in fabric-model-smoke.mjs. What can only be proved
 * here is that they survive to the DOM, because the defect this view exists to
 * remove was a rendering one: the page used to draw a fabric from browser
 * storage and colour it "live" from a string.
 *
 * The four states an operator can actually be in are each driven end to end:
 *   - the proxy answered and disclosed its nodes
 *   - the proxy answered but WITHHELD detail (it is not on loopback)
 *   - the proxy answered and the fabric is genuinely empty
 *   - nothing answered
 * plus pointing the view at an engine by mistake, which must say so.
 *
 * The proxy runs on its own origin, so this also exercises the cross-origin
 * read the shipped app really performs.
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

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.json': 'application/json',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
}

/* ---- the app's own origin ---- */

const appServer = createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1')
  const filePath = url.pathname === '/' ? '/index.html' : url.pathname
  const onDisk = join(distDir, filePath)
  if (existsSync(onDisk) && !onDisk.endsWith('/')) {
    res.writeHead(200, { 'Content-Type': MIME[extname(onDisk)] || 'application/octet-stream' })
    return res.end(readFileSync(onDisk))
  }
  // The engine serves an unauthenticated model-less shell for these routes; the
  // Cluster view does not use them, so an empty answer keeps the app quiet.
  if (filePath.startsWith('/api/') || filePath.startsWith('/v1/')) {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    return res.end('{}')
  }
  res.writeHead(200, { 'Content-Type': 'text/html' })
  return res.end(readFileSync(join(distDir, 'index.html')))
})

/* ---- the scripted fabric proxy, on its own origin ---- */

let proxy
function resetProxy() {
  proxy = { mode: 'down', requests: 0 }
}
resetProxy()

const READY_NODE = {
  spec: { label: 'win', host: '127.0.0.1', port: 8181, engine: 'camelid' },
  status: {
    state: 'ready',
    engine: 'camelid',
    active_model_id: 'Llama 3.2 1B Instruct',
    models: ['Llama 3.2 1B Instruct'],
    backend: 'cpu_q8_runtime_repack',
    version: 'v0.6.1-267',
    in_flight: 2,
    waiting: 1,
  },
  latency_ms: 3,
  placeable: true,
}
const NOT_READY_NODE = {
  spec: { label: 'mac', host: '192.0.2.10', port: 8181, engine: 'camelid' },
  status: { state: 'not_ready', reason: 'no model loaded' },
  latency_ms: 21,
  placeable: false,
}
const UNREACHABLE_NODE = {
  spec: { label: 'pi', host: '192.0.2.11', port: 8181, engine: 'camelid' },
  status: { state: 'unreachable', reason: 'connection timed out' },
  latency_ms: null,
  placeable: false,
}
/* A healthy Ollama node: read and reported, never placed on. It publishes no
   queue depth, so its load must render as an explicit unknown. */
const FOREIGN_NODE = {
  spec: { label: 'studio', host: '192.0.2.12', port: 11434, engine: 'ollama' },
  status: {
    state: 'ready',
    engine: 'ollama',
    active_model_id: null,
    models: ['mistral:latest', 'qwen3:8b'],
    backend: null,
    version: '0.33.3',
  },
  latency_ms: 9,
  placeable: false,
  capabilities: {
    load_reporting: { supported: false, provenance: 'declared', detail: 'publishes no queue depth' },
    typed_backpressure: { supported: false, provenance: 'declared', detail: 'no typed refusal to read' },
    tool_calls: { supported: null, provenance: 'not_probed', detail: null },
  },
  placement_blockers: ['publishes no queue depth', 'has no typed refusal to read'],
}
/* A second foreign engine. It exists in this fixture to prove the page learned
   a rule rather than a name: nothing about it is special-cased. Its tool-call
   support was measured on this exact build, so it renders differently from the
   unmeasured one above. */
const LM_STUDIO_NODE = {
  spec: { label: 'desk', host: '192.0.2.13', port: 1234, engine: 'lmstudio' },
  status: {
    state: 'ready',
    engine: 'lmstudio',
    active_model_id: 'meta-llama-3.1-8b-instruct',
    models: ['meta-llama-3.1-8b-instruct'],
    backend: null,
    version: null,
  },
  latency_ms: 6,
  placeable: false,
  capabilities: {
    load_reporting: { supported: false, provenance: 'declared', detail: 'publishes no queue depth' },
    tool_calls: { supported: true, provenance: 'measured', detail: '15 of 15 tool items answered on this build' },
  },
  placement_blockers: ['publishes no queue depth'],
}

const SUMMARY = {
  ok: true,
  service: 'camelid-fabric',
  version: '0.6.1',
  build: 'v0.6.1-267-gabc1234',
}

function proxyBody() {
  if (proxy.mode === 'withheld') return { ...SUMMARY, ready: true }
  if (proxy.mode === 'empty') return { ...SUMMARY, ready: false, nodes: { total: 0, ready: 0, not_ready: 0, unreachable: 0 }, models: [], node_detail: [] }
  if (proxy.mode === 'engine') return { ok: true, engine: 'camelid', generation_ready: true, version: '0.6.1', build: 'v0.6.1' }
  const nodes = [READY_NODE, NOT_READY_NODE, UNREACHABLE_NODE, FOREIGN_NODE, LM_STUDIO_NODE]
  return {
    ...SUMMARY,
    ready: true,
    nodes: { total: 5, ready: 3, not_ready: 1, unreachable: 1 },
    // Only what the fabric would actually place: the Ollama node's models are
    // deliberately absent here while it stays unplaceable.
    models: ['Llama 3.2 1B Instruct'],
    node_detail: nodes,
  }
}

const proxyServer = createServer((req, res) => {
  // The shipped proxy answers cross-origin; without this the browser blocks the
  // read and the smoke would be testing CORS, not the view.
  res.setHeader('Access-Control-Allow-Origin', '*')
  const url = new URL(req.url, 'http://127.0.0.1')
  if (url.pathname !== '/v1/health') {
    res.writeHead(404, { 'Content-Type': 'application/json' })
    return res.end('{}')
  }
  proxy.requests += 1
  if (proxy.mode === 'garbage') {
    res.writeHead(200, { 'Content-Type': 'text/html' })
    return res.end('<html>not json</html>')
  }
  const body = proxyBody()
  const status = proxy.mode === 'empty' ? 503 : 200
  res.writeHead(status, { 'Content-Type': 'application/json' })
  return res.end(JSON.stringify(body))
})

const listen = (server) => new Promise((done) => server.listen(0, '127.0.0.1', () => done(server.address().port)))

const appPort = await listen(appServer)
const proxyPort = await listen(proxyServer)
const appOrigin = `http://127.0.0.1:${appPort}`
const proxyEndpoint = `127.0.0.1:${proxyPort}`
const deadEndpoint = '127.0.0.1:9'

let checks = 0
function check(name) {
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}

const browser = await launchBrowser({ purpose: 'the fabric view smoke', headless: 'new' })

async function openCluster({ endpoint, viewport = { width: 1280, height: 900 } }) {
  const page = await browser.newPage()
  await page.setViewport(viewport)
  // Seed the address the same way a returning operator would have it, and clear
  // everything else so no scenario can inherit another's state.
  await page.evaluateOnNewDocument((value) => {
    window.localStorage.clear()
    window.localStorage.setItem('camelid.fabricEndpoint', value)
  }, endpoint)
  const errors = []
  page.on('pageerror', (error) => errors.push(String(error)))
  await page.goto(`${appOrigin}/#cluster`, { waitUntil: 'networkidle0' })
  await page.waitForSelector('[data-testid="fabric-status"][data-phase="settled"]', { timeout: 20000 })
  return { page, errors }
}

const textOf = (page, selector) => page.$eval(selector, (el) => el.textContent.trim()).catch(() => null)
const present = (page, selector) => page.$(selector).then((el) => Boolean(el))

try {
  /* ---- 1. the proxy disclosed its nodes ---- */
  proxy.mode = 'nodes'
  {
    const { page, errors } = await openCluster({ endpoint: proxyEndpoint })

    assert.equal(await textOf(page, '[data-testid="fabric-posture"]'), 'degraded', 'three ready nodes of five is degraded')
    check('a partly-ready fabric reports degraded, not serving')

    const counts = await page.$$eval('.fabric-count', (els) => els.map((el) => ({
      label: el.querySelector('.fabric-count__label').textContent.trim(),
      value: el.querySelector('.fabric-count__value').textContent.trim(),
    })))
    assert.deepEqual(counts, [
      { label: 'nodes', value: '5' },
      { label: 'ready', value: '3' },
      { label: 'not ready', value: '1' },
      { label: 'unreachable', value: '1' },
    ], 'each count must carry the proxy summary figure it names')
    check('the counts come from the proxy summary')

    const rows = await page.$$eval('.fabric-row', (els) => els.map((el) => ({
      label: el.getAttribute('data-node-label'),
      state: el.getAttribute('data-node-state'),
      engine: el.getAttribute('data-node-engine'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
      unknowns: el.querySelectorAll('[data-unknown="true"]').length,
    })))
    assert.equal(rows.length, 5, 'every disclosed node gets a row')
    check('one row per node the proxy disclosed')

    const win = rows.find((row) => row.label === 'win')
    assert.equal(win.state, 'ready')
    assert.match(win.text, /127\.0\.0\.1:8181/)
    assert.match(win.text, /Llama 3\.2 1B Instruct/)
    assert.match(win.text, /2 in flight/)
    assert.match(win.text, /1 waiting/)
    assert.match(win.text, /3 ms/)
    assert.equal(win.unknowns, 0, 'a fully-reported node has nothing unknown')
    check('a ready node renders every field the proxy sent')

    const mac = rows.find((row) => row.label === 'mac')
    assert.equal(mac.state, 'not_ready')
    assert.match(mac.text, /no model loaded/, 'the proxy reason reaches the operator verbatim')
    assert.ok(mac.unknowns >= 2, 'model and load are explicitly unknown for a node that is not ready')
    assert.doesNotMatch(mac.text, /0 in flight/, 'a node that reports no load must never render as idle')
    check('a not-ready node shows its reason and explicit unknowns, never a zero load')

    const pi = rows.find((row) => row.label === 'pi')
    assert.equal(pi.state, 'unreachable')
    assert.match(pi.text, /connection timed out/)
    assert.doesNotMatch(pi.text, /0 ms/, 'an unrecorded probe must not render as 0 ms')
    check('an unreachable node shows its reason and no invented latency')

    const models = await page.$eval('[data-testid="fabric-models"]', (el) => el.textContent.replace(/\s+/g, ' ').trim())
    assert.match(models, /Llama 3\.2 1B Instruct/)
    assert.match(models, /win/, 'a model is attributed to the node serving it')
    check('models are listed with the node serving them')

    // A foreign engine: visible, named, and explicitly not routed to.
    const studio = rows.find((row) => row.label === 'studio')
    assert.equal(studio.engine, 'ollama')
    assert.equal(studio.state, 'ready', 'the node itself is healthy')
    assert.match(studio.text, /ollama/)
    assert.match(studio.text, /not routed to/, 'the operator is told this fabric does not send work there')
    assert.match(studio.text, /2 models/, 'an engine holding several serves all of them')
    assert.doesNotMatch(studio.text, /0 in flight/, 'an engine that publishes no load must never render as idle')
    assert.ok(studio.unknowns >= 1, 'its load is an explicit unknown')
    check('a foreign engine is shown, named, and marked as not routed to')

    assert.equal(win.engine, 'camelid')
    assert.doesNotMatch(win.text, /not routed to/, 'a placeable node carries no such note')
    check('a placeable node is not marked as excluded')

    assert.doesNotMatch(
      models,
      /mistral|qwen3/,
      'a model only an unplaceable node holds must not be advertised as servable',
    )
    check('models only a foreign node holds are not advertised')

    // A second foreign engine proves the page applies a rule, not a name.
    const desk = rows.find((row) => row.label === 'desk')
    assert.equal(desk.engine, 'lmstudio')
    assert.equal(desk.state, 'ready')
    assert.match(desk.text, /not routed to/)
    assert.doesNotMatch(models, /meta-llama-3\.1-8b-instruct/, 'nor are its models')
    check('a second foreign engine is handled by the same rule, not a special case')

    // The capability matrix: an answer is only as good as how it was obtained.
    await page.click('.fabric-row[data-node-label="desk"]')
    await page.waitForSelector('[data-testid="fabric-capabilities"]', { timeout: 5000 })
    const caps = await page.$$eval('.fabric-cap', (els) => els.map((el) => ({
      name: el.getAttribute('data-capability'),
      provenance: el.getAttribute('data-provenance'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
      unknowns: el.querySelectorAll('[data-unknown="true"]').length,
    })))
    const tools = caps.find((cap) => cap.name === 'tool_calls')
    assert.equal(tools.provenance, 'measured')
    assert.match(tools.text, /yes/)
    assert.match(tools.text, /measured here/, 'a measured answer says so')
    check('a measured capability is rendered with its provenance')

    const load = caps.find((cap) => cap.name === 'load_reporting')
    assert.match(load.text, /no/)
    assert.equal(load.provenance, 'declared')
    check('a declared capability is not passed off as a measurement')

    const blockers = await page.$eval('.fabric-detail', (el) => el.textContent.replace(/\s+/g, ' ').trim())
    assert.match(blockers, /publishes no queue depth/, 'the operator is told why it is not routed to')
    check('the drawer gives the reason placement refuses, not just the verdict')

    // The unmeasured case must not read as a refusal.
    await page.click('.fabric-row[data-node-label="studio"]')
    await page.waitForSelector('[data-testid="fabric-capabilities"]', { timeout: 5000 })
    const unproved = await page.$$eval('.fabric-cap', (els) => els.map((el) => ({
      name: el.getAttribute('data-capability'),
      provenance: el.getAttribute('data-provenance'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
      unknowns: el.querySelectorAll('[data-unknown="true"]').length,
    }))).then((all) => all.find((cap) => cap.name === 'tool_calls'))
    assert.equal(unproved.provenance, 'not_probed')
    assert.equal(unproved.unknowns, 1, 'an unchecked capability is an explicit unknown')
    assert.doesNotMatch(unproved.text, /\bno\b/, 'never checked must not read as does not support')
    check('an unmeasured capability reads as unknown, never as unsupported')

    // The drawer is the replacement for the inspector that used to offer fake
    // worker controls, so it must open and must offer no such control.
    await page.click('.fabric-row[data-node-label="mac"]')
    await page.waitForSelector('.fabric-detail', { timeout: 5000 })
    const drawer = await page.$eval('.fabric-detail', (el) => el.textContent.replace(/\s+/g, ' ').trim())
    assert.match(drawer, /no model loaded/)
    assert.doesNotMatch(drawer, /Start worker|Stop worker|Restart worker/, 'the drawer must not offer a control it cannot perform')
    check('the node drawer reports state and offers no control it cannot perform')

    assert.deepEqual(errors, [], 'no page errors')
    check('the disclosed view raises no page error')
    await page.close()
  }

  /* ---- 2. the proxy withheld detail ---- */
  proxy.mode = 'withheld'
  {
    const { page, errors } = await openCluster({ endpoint: proxyEndpoint })

    assert.ok(await present(page, '[data-testid="fabric-withheld"]'), 'the withheld panel must appear')
    assert.equal(await present(page, '[data-testid="fabric-counts"]'), false, 'counts must not be shown when they were not disclosed')
    assert.equal(await present(page, '.fabric-row'), false, 'no node rows can exist when no node was disclosed')
    check('a proxy that withholds detail renders the reason, not a node list')

    const body = await page.$eval('.fabric-view', (el) => el.textContent.replace(/\s+/g, ' ').trim())
    assert.doesNotMatch(body, /\b0 nodes\b/, 'withheld detail must never be rendered as an empty fabric')
    assert.doesNotMatch(body, /This proxy has no nodes/, 'the empty state belongs to a disclosed empty fabric only')
    assert.match(body, /not bound to loopback/, 'the operator is told why detail is missing')
    assert.match(body, /serving at least one ready node/, 'what the proxy did disclose still reaches the operator')
    check('withheld is never rendered as empty')

    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- 3. a genuinely empty fabric ---- */
  proxy.mode = 'empty'
  {
    const { page, errors } = await openCluster({ endpoint: proxyEndpoint })

    assert.ok(await present(page, '[data-testid="fabric-counts"]'), 'a disclosed empty fabric still has counts')
    assert.equal(await present(page, '[data-testid="fabric-withheld"]'), false, 'an empty fabric is not a withheld one')
    const body = await page.$eval('.fabric-view', (el) => el.textContent.replace(/\s+/g, ' ').trim())
    assert.match(body, /This proxy has no nodes/)
    assert.match(body, /camelid fabric serve --node/, 'the empty state gives the command that fixes it')
    assert.equal(await textOf(page, '[data-testid="fabric-posture"]'), 'no node ready')
    check('a disclosed empty fabric reports zero and says how to fix it')

    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- 4. nothing answered ---- */
  {
    const { page, errors } = await openCluster({ endpoint: deadEndpoint })

    assert.ok(await present(page, '[data-testid="fabric-problem"]'), 'an unreachable proxy must be reported')
    assert.equal(await present(page, '.fabric-row'), false, 'nothing may be listed when nothing answered')
    assert.equal(await present(page, '[data-testid="fabric-counts"]'), false)
    assert.equal(await textOf(page, '[data-testid="fabric-posture"]'), 'unknown', 'an unreachable proxy has unknown posture, not "not ready"')
    const body = await page.$eval('.fabric-view', (el) => el.textContent.replace(/\s+/g, ' ').trim())
    assert.match(body, /No fabric proxy answered/)
    assert.match(body, /camelid fabric serve --node/)
    check('an unreachable proxy renders an honest failure and no fabricated fabric')

    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- 5. pointed at an engine by mistake ---- */
  proxy.mode = 'engine'
  {
    const { page } = await openCluster({ endpoint: proxyEndpoint })
    const body = await page.$eval('.fabric-view', (el) => el.textContent.replace(/\s+/g, ' ').trim())
    assert.match(body, /is a Camelid engine, not a fabric proxy/)
    assert.equal(await present(page, '.fabric-row'), false)
    check('an engine address is named as such, not shown as an empty fabric')
    await page.close()
  }

  /* ---- 6. a malformed answer ---- */
  proxy.mode = 'garbage'
  {
    const { page } = await openCluster({ endpoint: proxyEndpoint })
    const body = await page.$eval('.fabric-view', (el) => el.textContent.replace(/\s+/g, ' ').trim())
    assert.match(body, /not with a health report we could read/)
    check('a non-JSON answer is reported as unreadable, not as empty')
    await page.close()
  }

  /* ---- 7. nothing survives in storage that could assert a status ---- */
  proxy.mode = 'nodes'
  {
    const { page } = await openCluster({ endpoint: proxyEndpoint })
    await page.waitForSelector('.fabric-row', { timeout: 10000 })
    const stored = await page.evaluate(() => {
      const out = {}
      for (let i = 0; i < window.localStorage.length; i += 1) {
        const key = window.localStorage.key(i)
        out[key] = window.localStorage.getItem(key)
      }
      return out
    })
    // The address is an input and may be stored. A node, a state or a count is
    // a claim about a machine and may not be.
    const values = JSON.stringify(stored)
    assert.doesNotMatch(values, /clusterTopology/, 'the drawn-topology store must be gone')
    assert.doesNotMatch(values, /worker_state/, 'no worker state may be persisted')
    assert.doesNotMatch(values, /Llama 3\.2 1B Instruct/, 'no observed model may be persisted')
    assert.doesNotMatch(values, /node_detail|in_flight/, 'no observed node state may be persisted')
    check('no observed fabric state is written to browser storage')
    await page.close()
  }

  /* ---- 8. a phone ---- */
  {
    const { page, errors } = await openCluster({ endpoint: proxyEndpoint, viewport: { width: 390, height: 844 } })
    await page.waitForSelector('.fabric-row', { timeout: 10000 })
    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
    }))
    assert.ok(
      overflow.scrollWidth <= overflow.clientWidth + 1,
      `no horizontal overflow at 390px (scroll ${overflow.scrollWidth} vs client ${overflow.clientWidth})`,
    )
    check('the view fits a 390px phone without horizontal overflow')

    // A dropped column is a fact the operator silently stops being told, so the
    // narrow layout must still carry every cell.
    const cells = await page.$$eval('.fabric-row[data-node-label="win"] [role="cell"]', (els) => els.length)
    assert.equal(cells, 7, 'the narrow layout keeps every column')
    check('the narrow layout hides no column')

    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  console.log(`\nfabric view smoke: ${checks} checks passed`)
} finally {
  await browser.close()
  appServer.close()
  proxyServer.close()
}
