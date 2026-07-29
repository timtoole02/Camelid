#!/usr/bin/env node
// Self-test for smoke-release-artifact.mjs (run in CI by the validation-scripts
// job's test-*.mjs glob).
//
// The smoke script's job is to boot a real release binary, which does not exist
// in the validation-scripts job. So this suite covers it two ways:
//
//   1. every assertion is a pure function and is driven directly, including the
//      failure branches that a healthy binary would never produce;
//   2. the polling and probing logic is driven against a REAL loopback HTTP
//      server that impersonates the engine, so the transport path, retry
//      behaviour and response handling are genuinely executed.
//
// What is deliberately NOT covered here: spawning camelid itself. That is what
// the release workflow does with the actual artifact.
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  HEALTH_POLL_INTERVAL_MS,
  HEALTH_TIMEOUT_MS,
  assertCapabilitiesPayload,
  assertHealthPayload,
  assertNoCrashMarkers,
  assertUiAsset,
  assertWebUi,
  bootAndProbe,
  buildServeArgs,
  extractUiAssetUrl,
  httpGet,
  pickFreePort,
  probeServer,
  waitForHealth,
} from './smoke-release-artifact.mjs'

const has = (failures, pattern) => failures.some((f) => pattern.test(f))

// ---------------------------------------------------------------------------
// buildServeArgs — the boot must be model-less and deterministic
// ---------------------------------------------------------------------------
assert.deepEqual(
  buildServeArgs({ port: 8181, modelsDir: '/tmp/empty' }),
  ['serve', '--addr', '127.0.0.1:8181', '--no-open', '--models-dir', '/tmp/empty'],
  'serve must bind loopback, skip the browser, and be pinned to an explicit models dir',
)
assert.throws(() => buildServeArgs({ port: 8181 }), /modelsDir is required/, 'an unpinned models dir would let auto_select_model load a stray GGUF')
assert.throws(() => buildServeArgs({ port: 0, modelsDir: '/tmp' }), RangeError, 'port 0 must be rejected')
assert.throws(() => buildServeArgs({ port: 70000, modelsDir: '/tmp' }), RangeError, 'an out-of-range port must be rejected')

// ---------------------------------------------------------------------------
// /v1/health — HealthResponse (src/api/mod.rs) on a model-less engine
// ---------------------------------------------------------------------------
const healthyBody = JSON.stringify({
  ok: true,
  engine: 'camelid',
  loaded_now: false,
  generation_ready: false,
  active_model_id: null,
  backend: 'none',
  engine_queue_depth: 0,
})
assert.deepEqual(assertHealthPayload(200, healthyBody), [], 'a model-less engine at rest must pass')
assert.ok(has(assertHealthPayload(503, healthyBody), /expected 200, got 503/), 'a non-200 health must fail')
assert.ok(has(assertHealthPayload(200, 'not json'), /not valid JSON/), 'an unparseable health body must fail')
assert.ok(has(assertHealthPayload(200, JSON.stringify({ ...JSON.parse(healthyBody), ok: false })), /ok=false/), 'ok=false must fail')
assert.ok(
  has(assertHealthPayload(200, JSON.stringify({ ...JSON.parse(healthyBody), loaded_now: true })), /loaded_now=true/),
  'a model-less boot that reports a loaded model means the smoke did not test what it claims',
)
assert.ok(
  has(assertHealthPayload(200, JSON.stringify({ ...JSON.parse(healthyBody), generation_ready: true })), /generation_ready=true/),
  'generation_ready must be false without a model',
)
assert.ok(
  has(assertHealthPayload(200, JSON.stringify({ ...JSON.parse(healthyBody), active_model_id: 'llama' })), /active_model_id/),
  'an active model id without a model must fail',
)
assert.ok(
  has(assertHealthPayload(200, JSON.stringify({ ...JSON.parse(healthyBody), engine_queue_depth: 3 })), /engine_queue_depth=3/),
  'a non-zero queue depth at rest must fail',
)
assert.ok(
  has(assertHealthPayload(200, JSON.stringify({ ...JSON.parse(healthyBody), backend: 7 })), /backend must be a string/),
  'a malformed backend field must fail',
)

// ---------------------------------------------------------------------------
// /api/capabilities
// ---------------------------------------------------------------------------
assert.deepEqual(assertCapabilitiesPayload(200, '{"capabilities":[]}'), [], 'a JSON object must pass')
assert.ok(has(assertCapabilitiesPayload(404, '{}'), /expected 200, got 404/), 'a missing capabilities route must fail')
assert.ok(has(assertCapabilitiesPayload(200, '<html>'), /not valid JSON/), 'an HTML fallback served at an API path must fail')
assert.ok(has(assertCapabilitiesPayload(200, '[]'), /expected a JSON object/), 'a bare array must fail')

// ---------------------------------------------------------------------------
// GET / — the embedded web UI must actually be embedded
// ---------------------------------------------------------------------------
const goodHtml = '<!doctype html><html><head><title>Camelid</title><script type="module" src="/assets/index-BwWMwyQo.js"></script></head><body></body></html>'
const UI_ASSET_URL = '/assets/index-BwWMwyQo.js'
const UI_ASSET_BODY = 'export const app = 1;\n'
const htmlHeaders = { 'content-type': 'text/html' }
assert.deepEqual(assertWebUi(200, htmlHeaders, goodHtml), [], 'the real app shell must pass')
assert.ok(has(assertWebUi(404, htmlHeaders, goodHtml), /expected 200, got 404/), 'a missing root route must fail')
assert.ok(has(assertWebUi(200, { 'content-type': 'application/json' }, goodHtml), /expected an HTML content-type/), 'a non-HTML root must fail')
assert.ok(has(assertWebUi(200, htmlHeaders, '<html><head><title>Other</title></head></html>'), /Camelid app shell title/), 'a foreign page must fail')
assert.ok(
  has(assertWebUi(200, htmlHeaders, '<html><head><title>Camelid</title></head><body>placeholder</body></html>'), /no hashed \/assets/),
  'a shell with no bundle means the web UI build never reached the binary',
)

// ---------------------------------------------------------------------------
// The referenced bundle must be fetched, not just mentioned
// ---------------------------------------------------------------------------
assert.equal(extractUiAssetUrl(goodHtml), UI_ASSET_URL, 'the hashed module URL must be extracted from the shell')
assert.equal(
  extractUiAssetUrl('<script type="module" crossorigin src="/assets/index-Cz-PZZB3.js"></script>'),
  '/assets/index-Cz-PZZB3.js',
  'vite emits extra attributes before src; extraction must still work',
)
assert.equal(extractUiAssetUrl('<html><body>nothing</body></html>'), null, 'a shell with no bundle yields no URL')

assert.deepEqual(
  assertUiAsset(UI_ASSET_URL, 200, { 'content-type': 'text/javascript' }, UI_ASSET_BODY),
  [],
  'a served bundle must pass',
)
assert.deepEqual(
  assertUiAsset(UI_ASSET_URL, 200, { 'content-type': 'application/javascript; charset=utf-8' }, UI_ASSET_BODY),
  [],
  'the other common JavaScript content-type must also pass',
)
assert.ok(
  has(assertUiAsset(UI_ASSET_URL, 404, {}, ''), /does not serve/),
  'a referenced-but-missing bundle is a blank app and must fail',
)
assert.ok(
  has(assertUiAsset(UI_ASSET_URL, 200, { 'content-type': 'text/html' }, '<html>'), /expected a JavaScript content-type/),
  'an HTML fallback served at an asset path must fail',
)
assert.ok(has(assertUiAsset(UI_ASSET_URL, 200, { 'content-type': 'text/javascript' }, ''), /bundle is empty/), 'an empty bundle must fail')

// ---------------------------------------------------------------------------
// crash markers
// ---------------------------------------------------------------------------
assert.deepEqual(assertNoCrashMarkers('listening on 127.0.0.1:8181\n'), [], 'ordinary startup logging must pass')
assert.deepEqual(assertNoCrashMarkers(''), [], 'empty stderr must pass')
assert.ok(has(assertNoCrashMarkers("thread 'main' panicked at src/main.rs:1:1"), /rust panic/), 'a panic must fail')
assert.ok(has(assertNoCrashMarkers('\nthread ... has overflowed its stack\nstack overflow'), /stack overflow/), 'a stack overflow must fail')
assert.ok(has(assertNoCrashMarkers('Segmentation fault'), /segfault/), 'a segfault must fail')
assert.ok(has(assertNoCrashMarkers('Aborted (core dumped)'), /abort/), 'an abort must fail — a missing NVRTC must fall back to CPU, not abort')

// ---------------------------------------------------------------------------
// waitForHealth — injected clock/transport so every branch runs instantly
// ---------------------------------------------------------------------------
assert.equal(HEALTH_TIMEOUT_MS, 40_000, 'the health budget must stay aligned with camelid-desktop/src/engine.rs')
assert.equal(HEALTH_POLL_INTERVAL_MS, 350, 'the poll interval must stay aligned with camelid-desktop/src/engine.rs')

{
  const result = await waitForHealth({ port: 1, get: async () => ({ status: 200, headers: {}, body: healthyBody }) })
  assert.equal(result.ok, true, 'an immediately healthy engine must succeed')
}
{
  let attempts = 0
  const result = await waitForHealth({
    port: 1,
    get: async () => {
      attempts += 1
      if (attempts < 3) throw Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' })
      return { status: 200, headers: {}, body: healthyBody }
    },
    wait: async () => {},
  })
  assert.equal(result.ok, true, 'connection-refused while the listener binds is normal and must be retried')
  assert.equal(attempts, 3, 'polling must continue until the engine answers')
}
{
  const result = await waitForHealth({ port: 1, isAlive: () => false, get: async () => ({ status: 200, headers: {}, body: '{}' }) })
  assert.equal(result.ok, false)
  assert.match(result.reason, /exited before \/v1\/health answered/, 'a dead engine must be reported as death, not as a timeout')
}
{
  let clock = 0
  const result = await waitForHealth({
    port: 1,
    timeoutMs: 1000,
    intervalMs: 350,
    get: async () => { throw Object.assign(new Error('refused'), { code: 'ECONNREFUSED' }) },
    now: () => clock,
    wait: async (ms) => { clock += ms },
  })
  assert.equal(result.ok, false)
  assert.match(result.reason, /did not return 200 within 1000ms/, 'the budget must be enforced')
  assert.match(result.reason, /ECONNREFUSED/, 'the last transport state must be reported for diagnosis')
}
{
  let clock = 0
  const result = await waitForHealth({
    port: 1,
    timeoutMs: 1000,
    intervalMs: 350,
    get: async () => ({ status: 500, headers: {}, body: 'boom' }),
    now: () => clock,
    wait: async (ms) => { clock += ms },
  })
  assert.equal(result.ok, false)
  assert.match(result.reason, /HTTP 500/, 'a server that answers but never becomes healthy must report its status')
}

// ---------------------------------------------------------------------------
// transport + probe, against a real loopback server impersonating the engine
// ---------------------------------------------------------------------------
async function withServer(handler, run) {
  const server = createServer(handler)
  const port = await pickFreePort()
  await new Promise((done) => server.listen(port, '127.0.0.1', done))
  try {
    return await run(port)
  } finally {
    await new Promise((done) => server.close(done))
  }
}

const engineLike = (overrides = {}) => (req, res) => {
  const routes = {
    '/v1/health': () => res.writeHead(200, { 'content-type': 'application/json' }).end(healthyBody),
    '/api/capabilities': () => res.writeHead(200, { 'content-type': 'application/json' }).end('{"capabilities":[]}'),
    '/': () => res.writeHead(200, { 'content-type': 'text/html' }).end(goodHtml),
    [UI_ASSET_URL]: () => res.writeHead(200, { 'content-type': 'text/javascript' }).end(UI_ASSET_BODY),
    ...overrides,
  }
  const route = routes[req.url]
  if (route) route(res)
  else res.writeHead(404).end('nope')
}

await withServer(engineLike(), async (port) => {
  const response = await httpGet(port, '/v1/health')
  assert.equal(response.status, 200, 'httpGet must reach a real server')
  assert.deepEqual(JSON.parse(response.body).ok, true, 'httpGet must return the body verbatim')

  const health = await waitForHealth({ port, timeoutMs: 5000 })
  assert.equal(health.ok, true, 'waitForHealth must succeed against a real listener')

  const probe = await probeServer(port)
  assert.deepEqual(probe.failures, [], 'a fully healthy engine must produce zero probe failures')
})

await withServer(
  engineLike({ '/': (res) => res.writeHead(200, { 'content-type': 'text/html' }).end('<html><head><title>Camelid</title></head></html>') }),
  async (port) => {
    const probe = await probeServer(port)
    assert.ok(has(probe.failures, /no hashed \/assets/), 'an engine with an empty embedded UI must be caught end to end')
  },
)

await withServer(engineLike({ '/api/capabilities': (res) => res.writeHead(500).end('boom') }), async (port) => {
  const probe = await probeServer(port)
  assert.ok(has(probe.failures, /expected 200, got 500/), 'a broken capabilities route must be caught end to end')
})

// The reviewer's case: a syntactically valid shell whose bundle 404s. Every
// markup-level assertion passes; only fetching the URL catches it.
await withServer(engineLike({ [UI_ASSET_URL]: (res) => res.writeHead(404).end('missing') }), async (port) => {
  const probe = await probeServer(port)
  assert.deepEqual(
    assertWebUi(200, htmlHeaders, goodHtml),
    [],
    'the markup assertions alone must still pass — that is why the fetch is required',
  )
  assert.ok(has(probe.failures, /does not serve/), 'a referenced-but-missing bundle must be caught end to end')
})

await withServer(engineLike({ [UI_ASSET_URL]: (res) => res.writeHead(200, { 'content-type': 'text/html' }).end('<html>') }), async (port) => {
  const probe = await probeServer(port)
  assert.ok(has(probe.failures, /expected a JavaScript content-type/), 'an HTML body served at the bundle URL must be caught')
})

const closedPort = await pickFreePort()
await assert.rejects(
  () => httpGet(closedPort, '/v1/health', { timeoutMs: 250 }),
  /ECONNREFUSED|timed out/,
  'a transport failure must reject rather than resolve as a false pass',
)

// ---------------------------------------------------------------------------
// pickFreePort
// ---------------------------------------------------------------------------
{
  const port = await pickFreePort()
  assert.ok(Number.isInteger(port) && port > 0 && port <= 65535, 'pickFreePort must return a usable TCP port')
  const another = await pickFreePort()
  assert.ok(Number.isInteger(another), 'pickFreePort must be repeatable')
}

// ---------------------------------------------------------------------------
// bootAndProbe, against a real child process
//
// The release binary is not available in the validation-scripts job, so these
// drive a fixture process that behaves like the engine and then misbehaves in
// ways a healthy binary never would.
// ---------------------------------------------------------------------------
const fixtureDir = await mkdtemp(join(tmpdir(), 'camelid-smoke-fixture-'))
const fixturePath = join(fixtureDir, 'engine-fixture.mjs')
await writeFile(
  fixturePath,
  `import { createServer } from 'node:http'
import { writeSync } from 'node:fs'

const port = Number(process.argv[2])
const mode = process.argv[3] ?? 'stay'
const health = ${JSON.stringify(healthyBody)}
const html = ${JSON.stringify(goodHtml)}
const assetUrl = ${JSON.stringify(UI_ASSET_URL)}
const assetBody = ${JSON.stringify(UI_ASSET_BODY)}

const server = createServer((req, res) => {
  if (req.url === '/v1/health') res.writeHead(200, { 'content-type': 'application/json' }).end(health)
  else if (req.url === '/api/capabilities') res.writeHead(200, { 'content-type': 'application/json' }).end('{}')
  else if (req.url === '/') res.writeHead(200, { 'content-type': 'text/html' }).end(html)
  else if (req.url === assetUrl) res.writeHead(200, { 'content-type': 'text/javascript' }).end(assetBody)
  else res.writeHead(404).end('nope')

  // The bundle is the last probe probeServer issues. Dying right after
  // flushing it is the "served everything, then fell over" case.
  if (mode === 'die-after-probes' && req.url === assetUrl) {
    res.on('finish', () => {
      // writeSync, not process.stderr.write: an async pipe write can be
      // truncated by process.exit, and this message existing at all is the
      // point of the test.
      writeSync(2, "thread 'main' panicked at fixture: dying right after the last probe\\n")
      server.close()
      process.exit(0)
    })
  }
})
server.listen(port, '127.0.0.1')
`,
  'utf8',
)

{
  const port = await pickFreePort()
  const boot = await bootAndProbe({ command: process.execPath, args: [fixturePath, String(port), 'stay'], port, timeoutMs: 10_000 })
  assert.deepEqual(boot.failures, [], 'a fixture that boots, serves and stays up must pass')
}

{
  const port = await pickFreePort()
  const boot = await bootAndProbe({
    command: process.execPath,
    args: [fixturePath, String(port), 'die-after-probes'],
    port,
    timeoutMs: 10_000,
  })
  assert.ok(
    has(boot.failures, /stopped serving after the probes|exited on its own after answering the probes/),
    `an engine that serves every probe and then exits must NOT pass; got ${JSON.stringify(boot.failures)}`,
  )
  // Proves the teardown waits for stdio 'close' rather than 'exit': this line
  // is written while the process is on its way down, and would be missed by a
  // buffer read at exit time.
  assert.match(boot.stderr, /panicked at fixture/, 'stderr written on the way down must still be captured')
  assert.ok(has(assertNoCrashMarkers(boot.stderr), /rust panic/), 'the captured tail must reach the crash-marker check')
}

{
  const port = await pickFreePort()
  const boot = await bootAndProbe({ command: process.execPath, args: ['--eval', 'process.exit(3)'], port, timeoutMs: 3000 })
  assert.ok(has(boot.failures, /exited before \/v1\/health answered/), 'a process that never listens must fail, not hang')
}

{
  const boot = await bootAndProbe({
    command: join(fixtureDir, 'does-not-exist-anywhere'),
    args: [],
    port: await pickFreePort(),
    timeoutMs: 3000,
  })
  assert.ok(boot.failures.length > 0, 'a binary that cannot be spawned must fail')
}

await rm(fixtureDir, { recursive: true, force: true })

console.log('test-smoke-release-artifact: all checks passed')
