#!/usr/bin/env node
/* Browser coverage for Screen B, the "Find machines" panel.
 *
 * The pure rules are proved in discovery-model-smoke.mjs. This proves the page
 * actually behaves that way, and the three things it proves are the three that
 * would matter most if they were wrong:
 *
 *   - **Nothing is scanned until somebody asks.** A scan sends traffic to
 *     machines nobody named, so a panel that scanned on mount or on a poll
 *     would be doing that on its own.
 *   - **Nothing is added without a second, explicit click**, and what is
 *     written is what was shown.
 *   - **A written node is drawn only once the proxy reports it.** An
 *     optimistic row would assert a machine is in the fabric before anything
 *     re-read the file.
 *
 * Requires `npm run build` first (it serves frontend/dist) and Chrome/Edge.
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { launchBrowser } from './lib/launch-browser.mjs'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const distDir = resolve(scriptDir, '../dist')
if (!existsSync(distDir)) throw new Error(`missing ${distDir} -- run "npm run build" first`)

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.woff2': 'font/woff2', '.ico': 'image/x-icon',
}

const appServer = createServer((req, res) => {
  const filePath = decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/^\/+/, '')
  const onDisk = join(distDir, filePath)
  if (filePath && existsSync(onDisk) && !onDisk.endsWith('/')) {
    res.writeHead(200, { 'content-type': MIME[extname(onDisk)] || 'application/octet-stream' })
    return res.end(readFileSync(onDisk))
  }
  res.writeHead(200, { 'content-type': 'text/html' })
  return res.end(readFileSync(join(distDir, 'index.html')))
})

/* One finding per classification the panel groups differently, so a single
   scan exercises every row shape at once. */
const FINDINGS = [
  {
    id: 'f0',
    address: '100.64.0.37', port: 11434, addresses: ['100.64.0.37'], via_name: null,
    this_machine: false, identity_basis: 'unauthenticated_answer', tls_name_used: null,
    engines: [
      { engine: 'camelid', verdict: 'ruled_out', detail: '/v1/health answered HTTP 404' },
      { engine: 'ollama', verdict: 'matched', version: '0.33.2' },
      { engine: 'lmstudio', verdict: 'ruled_out' },
    ],
    classification: { kind: 'answers_like', engine: 'ollama', version: '0.33.2', withheld_elsewhere: [] },
    evidence: [{ request: 'GET /api/version', status: 200, content_type: 'application/json', fact: 'a JSON object naming a non-empty version', matched: ['ollama'] }],
    /* A name the router supplied, forward-confirmed. Offered, never defaulted. */
    name: { name: 'workstation.lan', source: 'reverse_dns', source_trust: 'device_claimed', proof: 'resolves_to_this_address', resolved: ['100.64.0.37'], why: null, rejected: null },
    in_fabric: null, possibly_same_as: [],
    proposal: {
      label: 'host-100-64-0-37-ollama', engine: 'ollama', host: '100.64.0.37', port: 11434,
      line: 'host-100-64-0-37-ollama=ollama://100.64.0.37:11434',
      comment_preview: '# joined by fabric discover <time of writing>: 100.64.0.37:11434 answered like ollama 0.33.2',
      engine_choices: [],
      host_alternatives: [{ host: 'workstation.lan', label: 'workstation-ollama', source_trust: 'device_claimed', warning: 'this name comes from your router; the device chooses it, and anything on this network can claim it later' }],
      /* No bearer warning: this engine is never shown the fabric's key. */
      warnings: ['cleartext'],
    },
    not_proposed: null,
  },
  {
    id: 'f1',
    address: '100.64.0.40', port: 8181, addresses: ['100.64.0.40'], via_name: null,
    this_machine: false, identity_basis: 'unauthenticated_answer', tls_name_used: null,
    engines: [{ engine: 'camelid', verdict: 'matched', version: 'v0.7.2-551' }],
    classification: { kind: 'answers_like', engine: 'camelid', version: 'v0.7.2-551', withheld_elsewhere: [] },
    evidence: [{ request: 'GET /v1/health', status: 200, content_type: 'application/json', fact: 'a health object naming this engine, whether it can generate, and a version', matched: ['camelid'] }],
    name: { name: null, source: null, source_trust: null, proof: null, resolved: null, why: 'no reverse DNS record', rejected: null },
    in_fabric: null, possibly_same_as: [],
    proposal: {
      label: 'host-100-64-0-40-camelid', engine: 'camelid', host: '100.64.0.40', port: 8181,
      line: 'host-100-64-0-40-camelid=camelid://100.64.0.40:8181',
      comment_preview: '# joined by fabric discover <time of writing>: 100.64.0.40:8181 answered like camelid v0.7.2-551',
      engine_choices: [], host_alternatives: [],
      /* The server decided this, from a fact only it holds. */
      warnings: ['cleartext', 'bearer_will_be_sent'],
    },
    not_proposed: null,
  },
  {
    id: 'f2',
    address: '100.64.0.50', port: 8080, addresses: ['100.64.0.50'], via_name: null,
    this_machine: false, identity_basis: 'unauthenticated_answer', tls_name_used: null,
    engines: [{ engine: 'camelid', verdict: 'ruled_out' }, { engine: 'ollama', verdict: 'ruled_out' }, { engine: 'lmstudio', verdict: 'ruled_out' }],
    classification: { kind: 'other_http', statuses: { '/v1/health': 200 } },
    evidence: [{ request: 'GET /v1/health', status: 200, content_type: 'text/html', fact: 'HTTP 200 with 42 bytes of text/html', matched: [] }],
    name: { name: null, why: 'no reverse DNS record', rejected: null, resolved: null, source: null, source_trust: null, proof: null },
    in_fabric: null, possibly_same_as: [], proposal: null,
    not_proposed: 'it speaks HTTP and matched no engine this build knows; add it by hand if you know what it is',
  },
  {
    id: 'f3',
    address: '100.64.0.60', port: 8282, addresses: ['100.64.0.60'], via_name: null,
    this_machine: false, identity_basis: 'unauthenticated_answer', tls_name_used: null,
    engines: [{ engine: 'camelid', verdict: 'not_a_node', detail: 'this is a fabric proxy, not a node' }],
    classification: { kind: 'fabric_proxy', reason: 'this is a fabric proxy, not a node' },
    evidence: [], name: { name: null, why: 'no reverse DNS record', rejected: null, resolved: null, source: null, source_trust: null, proof: null },
    in_fabric: null, possibly_same_as: [], proposal: null,
    not_proposed: 'a fabric proxy is not a node; point the Cluster view at it instead',
  },
  {
    id: 'f4',
    address: '100.64.0.70', port: 11434, addresses: ['100.64.0.70'], via_name: null,
    this_machine: false, identity_basis: 'unauthenticated_answer', tls_name_used: null,
    engines: [{ engine: 'ollama', verdict: 'undecided' }],
    classification: { kind: 'incomplete', unanswered: ['/api/tags'], matched_so_far: ['camelid'] },
    evidence: [], name: { name: null, why: 'no reverse DNS record', rejected: null, resolved: null, source: null, source_trust: null, proof: null },
    in_fabric: null, possibly_same_as: [], proposal: null,
    not_proposed: 'a check did not finish (/api/tags), so what this is has not been established; scan again',
  },
  {
    id: 'f5',
    address: '100.64.0.80', port: 1234, addresses: ['100.64.0.80'], via_name: null,
    this_machine: false, identity_basis: 'unauthenticated_answer', tls_name_used: null,
    engines: [{ engine: 'lmstudio', verdict: 'withheld' }],
    classification: { kind: 'requires_credentials', paths: ['/api/v0/models'] },
    evidence: [], name: { name: null, why: 'no reverse DNS record', rejected: null, resolved: null, source: null, source_trust: null, proof: null },
    in_fabric: null, possibly_same_as: [], proposal: null,
    not_proposed: 'it asked for a credential on /api/v0/models',
  },
  {
    id: 'f6',
    address: 'fd00::1', port: 11434, addresses: ['fd00::1'], via_name: null,
    this_machine: false, identity_basis: 'unauthenticated_answer', tls_name_used: null,
    engines: [{ engine: 'ollama', verdict: 'matched', version: '0.33.2' }],
    classification: { kind: 'answers_like', engine: 'ollama', version: '0.33.2', withheld_elsewhere: [] },
    evidence: [{ request: 'GET /api/version', status: 200, content_type: 'application/json', fact: 'a JSON object naming a non-empty version', matched: ['ollama'] }],
    name: { name: null, source: null, source_trust: null, proof: null, resolved: null, why: 'no reverse DNS record', rejected: null },
    in_fabric: null, possibly_same_as: [],
    proposal: {
      label: 'host-fd00--1-ollama', engine: 'ollama', host: '[fd00::1]', port: 11434,
      line: 'host-fd00--1-ollama=ollama://[fd00::1]:11434',
      comment_preview: '# joined by fabric discover <time of writing>: [fd00::1]:11434 answered like ollama 0.33.2',
      /* Bracketed, and deliberately not `address:port`: a page that composed
         this itself would send `fd00::1:11434`, which is not an address. */
      scanned_address: '[fd00::1]:11434',
      engine_choices: [], host_alternatives: [], warnings: ['cleartext'],
    },
    not_proposed: null,
  },
]

const DISCOVERY = {
  scope: { ranges: ['100.64.0.0/24'], hosts: [], loopback: true, default_ports: true, addresses: 254, ports: [1234, 8181, 11434], limits: {} },
  transport: 'cleartext (direct remote transport explicitly acknowledged)',
  credentials_presented: 'none',
  user_agent: 'camelid-fabric-discover/0.7.3',
  elapsed_ms: 6100, planned: 762, probes: 762,
  not_listed: { refused: 500, timed_out: 256, unreachable: 0, other: 0 },
  not_scanned: 0,
  nodes_file: { path: '/etc/camelid/nodes', sha256: 'abc123' },
  findings: FINDINGS,
  hint: null,
}

const POLICY = {
  enabled: true,
  nodes_file: { path: '/etc/camelid/nodes', sha256: 'abc123', labels: ['local'] },
  default_ports: [8181, 11434, 1234],
  limits: { max_addresses: 1024, max_ports: 8, concurrency: 32, connects_per_second: 200, connect_timeout_ms: 250, request_timeout_ms: 1500, max_body_bytes: 1048576, wall_clock_ms: 60000 },
  allowed_ranges: ['100.64.0.0/10', '169.254.0.0/16', '127.0.0.0/8'],
  suggestions: [{ cidr: '100.64.0.0/24', interface: 'en0', address: '100.64.0.20', prefix_source: 'interface_netmask' }],
  transport: { description: 'cleartext (direct remote transport explicitly acknowledged)', lan_permitted: true, flag_needed: '--allow-cleartext-node-transport' },
  fabric_bearer_configured: true,
  credentials_presented: 'none',
}

const NODE_DETAIL = [{
  spec: { label: 'local', host: '127.0.0.1', port: 8181, engine: 'camelid' },
  status: { state: 'ready', engine: 'camelid', active_model_id: 'llama-3.2-1b', models: ['llama-3.2-1b'], backend: 'metal', version: 'v0.7.2-551', in_flight: 0, waiting: 0 },
  latency_ms: 4, placeable: true,
}]

const proxy = {}
function resetProxy(overrides = {}) {
  for (const key of Object.keys(proxy)) delete proxy[key]
  Object.assign(proxy, {
    // 'on' | 'disabled' | 'old_build'
    mode: 'on',
    scans: 0,
    joins: [],
    // Labels /v1/health reports. The joined node appears only when told to.
    labels: ['local'],
    lastJoin: null,
  }, overrides)
}
resetProxy()

const proxyServer = createServer((req, res) => {
  res.setHeader('access-control-allow-origin', '*')
  res.setHeader('access-control-allow-headers', 'content-type, authorization')
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end() }
  const reply = (status, value) => {
    res.writeHead(status, { 'content-type': 'application/json' })
    res.end(JSON.stringify(value))
  }

  if (req.url === '/v1/health') {
    return reply(200, {
      ok: true, service: 'camelid-fabric', version: '0.7.3', build: 'v0.7.3-601', ready: true,
      nodes: { total: proxy.labels.length, ready: proxy.labels.length, not_ready: 0, unreachable: 0 },
      models: ['llama-3.2-1b'],
      node_detail: proxy.labels.map((label) => ({
        ...NODE_DETAIL[0],
        spec: { ...NODE_DETAIL[0].spec, label },
      })),
    })
  }

  if (req.url === '/v1/fabric/discover' && req.method === 'GET') {
    if (proxy.mode === 'disabled') {
      return reply(404, { error: { message: 'not started with --discovery', type: 'fabric_error', code: 'discovery_disabled' } })
    }
    if (proxy.mode === 'old_build') {
      return reply(404, { error: { message: 'this fabric proxy does not serve that route', type: 'fabric_error' } })
    }
    return reply(200, POLICY)
  }

  if (req.url === '/v1/fabric/discover' && req.method === 'POST') {
    proxy.scans += 1
    let raw = ''
    req.on('data', (chunk) => { raw += chunk })
    req.on('end', () => reply(200, DISCOVERY))
    return undefined
  }

  if (req.url === '/v1/fabric/discover/join' && req.method === 'POST') {
    let raw = ''
    req.on('data', (chunk) => { raw += chunk })
    req.on('end', () => {
      const request = JSON.parse(raw)
      proxy.joins.push(request)
      proxy.lastJoin = request
      reply(200, {
        written: true,
        path: '/etc/camelid/nodes',
        appended: `# joined by fabric discover 2026-09-12T10:04:11Z: ${request.host}:${request.port} answered like ${request.engine}\n${request.label}=${request.engine}://${request.host}:${request.port}\n`,
        line: `${request.label}=${request.engine}://${request.host}:${request.port}`,
        answered_from: `${request.host}:${request.port}`,
        sha256_before: 'abc123', sha256_after: 'def456',
        note: 'the proxy re-reads this file within 1 s',
      })
    })
    return undefined
  }

  return reply(404, { error: { message: 'unknown', type: 'fabric_error' } })
})

function listen(server) {
  return new Promise((done) => server.listen(0, '127.0.0.1', () => done(server.address().port)))
}

let checks = 0
function check(name) {
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}

const appPort = await listen(appServer)
const proxyPort = await listen(proxyServer)
const appOrigin = `http://127.0.0.1:${appPort}`
const proxyEndpoint = `127.0.0.1:${proxyPort}`
const browser = await launchBrowser({ purpose: 'the discovery view smoke', headless: 'new' })

async function until(predicate, ms = 5000) {
  const start = Date.now()
  while (!predicate()) {
    if (Date.now() - start > ms) return false
    await new Promise((done) => setTimeout(done, 50))
  }
  return true
}

async function openCluster({ expect = 'ready' } = {}) {
  const page = await browser.newPage()
  await page.setViewport({ width: 1280, height: 900 })
  const errors = []
  page.on('pageerror', (error) => errors.push(String(error)))
  await page.evaluateOnNewDocument((value) => {
    window.localStorage.clear()
    window.localStorage.setItem('camelid.fabricEndpoint', value)
  }, proxyEndpoint)
  await page.goto(`${appOrigin}/#cluster`, { waitUntil: 'networkidle0' })
  await page.waitForSelector(`[data-testid="fabric-discovery"][data-state="${expect}"]`, { timeout: 15000 })
  return { page, errors }
}

const textOf = (page, selector) =>
  page.$eval(selector, (el) => el.textContent.replace(/\s+/g, ' ').trim())

async function scan(page) {
  await page.click('[data-testid="discovery-scan"]')
  await page.waitForSelector('[data-testid="discovery-results"]', { timeout: 15000 })
}

console.log('discovery view')

try {
  /* ---- looking is never automatic ---- */
  resetProxy()
  {
    const { page, errors } = await openCluster()
    // The panel reads the policy on mount, which is a GET and changes nothing.
    // A scan is a POST, and must wait for a person.
    await new Promise((done) => setTimeout(done, 6000))
    assert.equal(proxy.scans, 0, 'the panel scanned without being asked')
    check('no scan is sent until Scan is clicked')

    const prefilled = await page.$eval('[data-testid="discovery-range"]', (el) => el.value)
    assert.equal(prefilled, '100.64.0.0/24', 'the suggested range is offered, pre-filled')
    assert.match(await textOf(page, '[data-testid="discovery-suggestion"]'), /en0/)
    assert.match(await textOf(page, '[data-testid="discovery-transport"]'), /No credential is presented to any host/)
    check('the suggested range is pre-filled, and says where its prefix came from')

    await scan(page)
    assert.equal(proxy.scans, 1, 'one click must send exactly one scan')
    check('one click sends exactly one scan')

    const summary = await textOf(page, '[data-testid="discovery-summary"]')
    assert.match(summary, /762/)
    assert.match(summary, /500/)
    check('the summary reports what was planned, probed and never answered')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- only an identified machine can be added ---- */
  resetProxy()
  {
    const { page, errors } = await openCluster()
    await scan(page)

    const addable = await page.$$eval('[data-testid="discovery-row"]', (rows) => rows
      .filter((row) => row.querySelector('[data-testid="discovery-add"]'))
      .map((row) => row.getAttribute('data-kind')))
    assert.deepEqual(addable, ['answers_like', 'answers_like'],
      'only the rows the proxy identified may be added')

    for (const kind of ['other_http', 'fabric_proxy', 'incomplete', 'requires_credentials']) {
      const row = await page.$(`[data-testid="discovery-row"][data-kind="${kind}"]`)
      assert.ok(row, `a ${kind} row should still be shown`)
      assert.equal(await row.$('[data-testid="discovery-add"]'), null,
        `a ${kind} row must offer no way to add it`)
      const explained = await row.$('[data-testid="discovery-not-proposed"]')
      assert.ok(explained, `a ${kind} row must say why it is not offered`)
    }
    check('an unrelated service offers no way to add it')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- two clicks, and what is written is what was shown ---- */
  resetProxy()
  {
    const { page, errors } = await openCluster()
    await scan(page)

    const ollamaRow = '[data-testid="discovery-row"][data-address="100.64.0.37"]'
    await page.click(`${ollamaRow} [data-testid="discovery-add"]`)
    await page.waitForSelector('[data-testid="discovery-confirm"]', { timeout: 5000 })

    // The host field holds the address. The router's name is an alternative.
    const host = await page.$eval('[data-testid="discovery-confirm-host"]', (el) => el.value)
    assert.equal(host, '100.64.0.37', 'a device-claimed name must never be the default host')
    const alternative = await textOf(page, '[data-testid="discovery-confirm-alternatives"]')
    assert.match(alternative, /workstation\.lan/)
    assert.match(alternative, /comes from your router/)
    check('a router name is an alternative, not the default host')

    // Cancelling writes nothing at all.
    await page.click('[data-testid="discovery-confirm-cancel"]')
    await page.waitForFunction(() => !document.querySelector('[data-testid="discovery-confirm"]'), { timeout: 5000 })
    assert.equal(proxy.joins.length, 0, 'cancelling must write nothing')

    await page.click(`${ollamaRow} [data-testid="discovery-add"]`)
    await page.waitForSelector('[data-testid="discovery-confirm"]', { timeout: 5000 })
    const shown = await textOf(page, '[data-testid="discovery-confirm-lines"]')
    assert.match(shown, /host-100-64-0-37-ollama=ollama:\/\/100\.64\.0\.37:11434/)
    await page.click('[data-testid="discovery-confirm-write"]')
    assert.ok(await until(() => proxy.joins.length === 1), 'the write should have been sent')

    assert.equal(proxy.lastJoin.label, 'host-100-64-0-37-ollama')
    assert.equal(proxy.lastJoin.host, '100.64.0.37')
    assert.equal(proxy.lastJoin.engine, 'ollama')
    assert.equal(proxy.lastJoin.base_sha256, 'abc123')
    check('nothing joins without the second click, and what is written is what was shown')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- a written node is the proxy's to report ---- */
  resetProxy()
  {
    const { page, errors } = await openCluster()
    await scan(page)
    await page.click('[data-testid="discovery-row"][data-address="100.64.0.37"] [data-testid="discovery-add"]')
    await page.waitForSelector('[data-testid="discovery-confirm"]', { timeout: 5000 })
    await page.click('[data-testid="discovery-confirm-write"]')
    await page.waitForSelector('[data-testid="discovery-joined-waiting"]', { timeout: 10000 })

    const waiting = await textOf(page, '[data-testid="discovery-joined-waiting"]')
    assert.match(waiting, /host-100-64-0-37-ollama=ollama:\/\/100\.64\.0\.37:11434/,
      'the exact text the proxy said it wrote is shown')
    assert.match(waiting, /waiting for the proxy to report it/)
    const drawn = await page.$$eval('.fabric-row', (rows) => rows.map((row) => row.getAttribute('data-node-label')))
    assert.ok(!drawn.includes('host-100-64-0-37-ollama'),
      'the node table must not draw a node the proxy has not reported')
    check('a joined node is drawn only once the proxy reports it')

    // Now the proxy picks the file up, and the node appears on its own.
    proxy.labels = ['local', 'host-100-64-0-37-ollama']
    await page.waitForFunction(
      () => [...document.querySelectorAll('.fabric-row')]
        .some((row) => row.getAttribute('data-node-label') === 'host-100-64-0-37-ollama'),
      { timeout: 15000 },
    )
    check('the node appears once the proxy itself reports it')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- the warning comes from the server, not the engine name ---- */
  resetProxy()
  {
    const { page } = await openCluster()
    await scan(page)

    // The camelid row carries the server's warning...
    await page.click('[data-testid="discovery-row"][data-address="100.64.0.40"] [data-testid="discovery-add"]')
    await page.waitForSelector('[data-testid="discovery-confirm-warnings"]', { timeout: 5000 })
    const camelidWarnings = await page.$$eval('[data-testid="discovery-confirm-warnings"] li',
      (items) => items.map((item) => item.getAttribute('data-warning')))
    assert.ok(camelidWarnings.includes('bearer_will_be_sent'), `${camelidWarnings}`)
    await page.click('[data-testid="discovery-confirm-cancel"]')

    // ...and the ollama row does not, because the server did not send it.
    await page.click('[data-testid="discovery-row"][data-address="100.64.0.37"] [data-testid="discovery-add"]')
    await page.waitForSelector('[data-testid="discovery-confirm"]', { timeout: 5000 })
    const ollamaWarnings = await page.$$eval('[data-testid="discovery-confirm-warnings"] li',
      (items) => items.map((item) => item.getAttribute('data-warning')))
    assert.ok(!ollamaWarnings.includes('bearer_will_be_sent'), `${ollamaWarnings}`)
    check("the bearer warning follows the server's warnings, not the engine")
    await page.close()
  }

  /* ---- findings describe what answered moments ago ---- */
  resetProxy()
  {
    const { page } = await openCluster()
    const before = await page.evaluate(() => Object.keys(window.localStorage).sort())
    await scan(page)
    await page.reload({ waitUntil: 'networkidle0' })
    await page.waitForSelector('[data-testid="fabric-discovery"]', { timeout: 15000 })

    assert.equal(await page.$('[data-testid="discovery-results"]'), null,
      'findings must not survive a reload')
    const after = await page.evaluate(() => Object.keys(window.localStorage).sort())
    assert.deepEqual(after, before, 'a scan must write nothing to browser storage')
    check('findings do not survive a reload')
    await page.close()
  }

  /* ---- a proxy that will not do this says what to run ---- */
  resetProxy({ mode: 'disabled' })
  {
    const { page } = await openCluster({ expect: 'disabled' })
    assert.equal(await page.$('[data-testid="discovery-scan"]'), null,
      'a disabled proxy must render no control that does nothing')
    const command = await page.$eval('[data-testid="fabric-discovery"] .fabric-cmd code', (el) => el.textContent)
    assert.match(command, /--discovery/)
    check('a disabled proxy shows a command, not a button')
    await page.close()
  }

  resetProxy({ mode: 'old_build' })
  {
    const { page } = await openCluster({ expect: 'old_build' })
    assert.match(await textOf(page, '[data-testid="fabric-discovery"]'), /before discovery existed/)
    check('a build from before discovery is told apart from one with it switched off')
    await page.close()
  }

  /* ---- the socket the scan reached is the server's to spell ---- */
  resetProxy()
  {
    const { page, errors } = await openCluster()
    await scan(page)
    await page.click('[data-testid="discovery-row"][data-address="fd00::1"] [data-testid="discovery-add"]')
    await page.waitForSelector('[data-testid="discovery-confirm"]', { timeout: 5000 })
    await page.click('[data-testid="discovery-confirm-write"]')
    assert.ok(await until(() => proxy.joins.length === 1), 'the write should have been sent')

    assert.equal(proxy.lastJoin.host, '[fd00::1]')
    assert.equal(proxy.lastJoin.scanned_address, '[fd00::1]:11434',
      "the page must send the server's spelling, not one built from address and port")
    check('an IPv6 machine is joined by the socket the server said it reached')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- a write the proxy never reports stops implying progress ---- */
  resetProxy()
  {
    const { page, errors } = await openCluster()
    await scan(page)
    await page.click('[data-testid="discovery-row"][data-address="100.64.0.37"] [data-testid="discovery-add"]')
    await page.waitForSelector('[data-testid="discovery-confirm"]', { timeout: 5000 })
    await page.click('[data-testid="discovery-confirm-write"]')
    await page.waitForSelector('[data-testid="discovery-joined-waiting"]', { timeout: 10000 })

    // proxy.labels never changes: the file was written and the process never
    // picked it up. The row has to say so on its own, because nothing else
    // will ever arrive to make it.
    await page.waitForSelector('[data-testid="discovery-joined-unreported"]', { timeout: 25000 })
    const said = await textOf(page, '[data-testid="discovery-joined-unreported"]')
    assert.match(said, /has not picked it up/)
    check('a write the proxy never reports escalates instead of waiting for ever')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- layout ---- */
  resetProxy()
  {
    const page = await browser.newPage()
    await page.setViewport({ width: 390, height: 844 })
    await page.evaluateOnNewDocument((value) => {
      window.localStorage.clear()
      window.localStorage.setItem('camelid.fabricEndpoint', value)
    }, proxyEndpoint)
    await page.goto(`${appOrigin}/#cluster`, { waitUntil: 'networkidle0' })
    await page.waitForSelector('[data-testid="discovery-scan"]', { timeout: 15000 })
    await scan(page)
    const overflow = await page.evaluate(() =>
      document.documentElement.scrollWidth - document.documentElement.clientWidth)
    assert.ok(overflow <= 1, `horizontal overflow of ${overflow}px at 390px`)
    check('the panel fits a 390px phone without horizontal overflow')
    await page.close()
  }

  console.log(`\ndiscovery view smoke: ${checks} checks passed`)
} finally {
  await browser.close()
  appServer.close()
  proxyServer.close()
}
