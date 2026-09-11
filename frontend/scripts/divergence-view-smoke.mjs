#!/usr/bin/env node
/* Browser coverage for Screen D.
 *
 * The pure rules are proved in divergence-model-smoke.mjs. This proves the page
 * actually renders them: that a refusal to conclude never reaches the screen
 * looking like a finding, that a side which did not repeat itself is called out
 * before the reader compares anything, and that a request the proxy refused is
 * not shown as a comparison that found nothing.
 *
 * The scripted proxy mirrors the real one wherever the page depends on it: it
 * records `model_identity` from the ids it was asked for, applies and reports
 * the token cap, refuses without its client key when it has one, and — like
 * `camelid fabric serve` started without `--cors-origin` — can send no CORS
 * header at all.
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

const CAMELID_TEMPLATE = '{{- bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}'
const OLLAMA_TEMPLATE = '{{ if .System }}Cutting Knowledge Date: December 2023{{ end }}{{ .Prompt }}'

/* A side whose response named exactly the model it was asked for. */
const ECHO_REQUESTED = '<echo the requested id>'

const DIVERGENT = {
  prompt: 'What is 7 plus 5?',
  prompt_sha256: 'f00dcafe',
  plan: { temperature: 0, seed: 0, max_tokens: 64, repetitions: 2 },
  left: {
    label: 'win', engine: 'camelid', engine_version: 'v0.6.1-267', model: 'llama-3.2-1b',
    applied_sampling: { temperature: 'sent', seed: 'sent' },
    samples: [{ text: '12', sha256: 'aaaa', elapsed_ms: 30 }, { text: '12', sha256: 'aaaa', elapsed_ms: 28 }],
    stability: { kind: 'stable' },
    advertised_template: { kind: 'captured', source: 'GET /props', template: CAMELID_TEMPLATE },
    rendered_prompt: { kind: 'captured', source: 'POST /apply-template', text: '<|user|>What is 7 plus 5?' },
  },
  right: {
    label: 'studio', engine: 'ollama', engine_version: '0.33.3', model: 'llama-3.2-1b',
    applied_sampling: { temperature: 'sent', seed: 'sent' },
    samples: [{ text: '7', sha256: 'bbbb', elapsed_ms: 41 }, { text: '7', sha256: 'bbbb', elapsed_ms: 39 }],
    stability: { kind: 'stable' },
    advertised_template: { kind: 'captured', source: 'POST /api/show', template: OLLAMA_TEMPLATE },
    rendered_prompt: { kind: 'unavailable', reason: "Ollama's documented API has no route that renders a chat prompt without generating" },
  },
  verdict: { kind: 'divergent' },
  diff: { kind: 'lines', lines: [{ op: 'removed', text: '12' }, { op: 'added', text: '7' }] },
  uncontrolled: [],
}

const UNSTABLE = {
  ...DIVERGENT,
  right: {
    ...DIVERGENT.right,
    samples: [{ text: '7', sha256: 'bbbb', elapsed_ms: 41 }, { text: 'seven', sha256: 'cccc', elapsed_ms: 39 }],
    stability: { kind: 'unstable', digests: ['bbbb', 'cccc'] },
  },
  verdict: { kind: 'not_attributable', reason: 'studio did not repeat its own answer' },
  diff: { kind: 'declined', reason: 'the two sides are not comparable, so no diff is shown' },
}

const DIFFERENT_MODELS = {
  ...DIVERGENT,
  right: { ...DIVERGENT.right, model: 'qwen3:8b' },
  verdict: { kind: 'different_models', left: 'llama-3.2-1b', right: 'qwen3:8b' },
  diff: { kind: 'declined', reason: 'the two sides are not comparable, so no diff is shown' },
}

const LMSTUDIO_UNSEEDED = {
  ...DIVERGENT,
  right: {
    label: 'desk', engine: 'lmstudio', engine_version: null, model: 'llama-3.2-1b',
    applied_sampling: { temperature: 'sent', seed: 'unsupported' },
    samples: [{ text: '7', sha256: 'bbbb', elapsed_ms: 41 }, { text: '7', sha256: 'bbbb', elapsed_ms: 39 }],
    stability: { kind: 'stable' },
    advertised_template: { kind: 'not_exposed', detail: "LM Studio's documented API exposes no prompt template" },
    rendered_prompt: { kind: 'unavailable', reason: "LM Studio's documented API has no route that renders a chat prompt without generating" },
  },
  uncontrolled: ['seed'],
  uncontrolled_detail: [
    { name: 'seed', reason: 'desk (lmstudio) runs an engine whose documented completion API has no seed parameter, so its runs were sent none' },
  ],
}

/* Two answers that differ only in how their last line ends, from a pair of
   nodes one of which answered under another model's name. */
const LINE_ENDINGS = {
  ...DIVERGENT,
  left: {
    ...DIVERGENT.left,
    reported_model: ECHO_REQUESTED,
    samples: [
      { text: 'The answer is\n12\r\n', sha256: 'dddd', elapsed_ms: 30 },
      { text: 'The answer is\n12\r\n', sha256: 'dddd', elapsed_ms: 29 },
    ],
  },
  right: {
    ...DIVERGENT.right,
    reported_model: 'llama-3.2-3b-instruct',
    samples: [
      { text: 'The answer is\n12', sha256: 'eeee', elapsed_ms: 41 },
      { text: 'The answer is\n12', sha256: 'eeee', elapsed_ms: 40 },
    ],
  },
  diff: {
    kind: 'lines',
    lines: [
      { op: 'same', text: 'The answer is', eol: 'lf' },
      { op: 'removed', text: '12', eol: 'crlf' },
      { op: 'added', text: '12', eol: 'none' },
    ],
  },
}

/* The live receipt: both engines advertise the same template, and still answer
   differently. */
const SAME_TEMPLATE = {
  ...DIVERGENT,
  right: {
    ...DIVERGENT.right,
    advertised_template: { kind: 'captured', source: 'POST /api/show', template: CAMELID_TEMPLATE },
  },
}

const BODIES = {
  divergent: DIVERGENT,
  same_template: SAME_TEMPLATE,
  unstable: UNSTABLE,
  different_models: DIFFERENT_MODELS,
  lmstudio: LMSTUDIO_UNSEEDED,
  eol: LINE_ENDINGS,
}

/* The view populates its pickers from the fabric, so the stub has to be a
   fabric proxy as well as a comparison endpoint. */
const NODE_DETAIL = [
  {
    spec: { label: 'studio', host: '192.0.2.12', port: 11434, engine: 'ollama' },
    status: {
      state: 'ready', engine: 'ollama', active_model_id: null,
      models: ['llama-3.2-1b-instruct:latest', 'qwen3:8b'],
      backend: null, version: '0.33.3',
    },
    latency_ms: 9, placeable: false,
  },
  {
    spec: { label: 'desk', host: '192.0.2.13', port: 1234, engine: 'lmstudio' },
    status: {
      state: 'ready', engine: 'lmstudio', active_model_id: 'llama-3.2-1b-instruct',
      models: ['llama-3.2-1b-instruct'],
      backend: null, version: null,
    },
    latency_ms: 6, placeable: false,
  },
  {
    spec: { label: 'mac', host: '192.0.2.10', port: 8181, engine: 'camelid' },
    status: { state: 'not_ready', reason: 'no model loaded' },
    latency_ms: 21, placeable: false,
  },
]

const proxy = {}
function resetProxy(overrides = {}) {
  for (const key of Object.keys(proxy)) delete proxy[key]
  Object.assign(proxy, {
    mode: 'divergent',
    // 'disclosed' | 'withheld' (an off-loopback proxy) | 'empty'
    detail: 'disclosed',
    // 'allow' plays a proxy started with --cors-origin; 'none' plays the default.
    cors: 'allow',
    requireKey: null,
    // A proxy from before model_identity, uncontrolled and eol existed.
    legacy: false,
    lastRequest: null,
    hung: null,
  }, overrides)
}
resetProxy()

function healthBody() {
  const body = {
    ok: true, service: 'camelid-fabric', version: '0.6.1', build: 'v0.6.1-349', ready: true,
  }
  // Absent node_detail is what an off-loopback proxy sends. It is not an empty
  // fabric, and the picker has to say so rather than offer an empty list.
  if (proxy.detail === 'disclosed') {
    body.nodes = { total: 3, ready: 2, not_ready: 1, unreachable: 0 }
    body.models = []
    body.node_detail = NODE_DETAIL
  }
  if (proxy.detail === 'empty') {
    body.ready = false
    body.nodes = { total: 0, ready: 0, not_ready: 0, unreachable: 0 }
    body.models = []
    body.node_detail = []
  }
  return body
}

/* What the real proxy does with a request, where the page depends on it. */
function comparisonFor(fixture, request) {
  const body = structuredClone(fixture)
  // The proxy clamps rather than rejects, and reports the cap it applied.
  body.plan = { ...body.plan, max_tokens: Math.min(1024, Math.max(1, request.max_tokens ?? 64)) }
  if (proxy.legacy) {
    delete body.uncontrolled
    for (const line of body.diff.lines || []) delete line.eol
    // An older proxy sent the advertised capture as `template`, and no render.
    for (const side of [body.left, body.right]) {
      side.template = side.advertised_template
      delete side.advertised_template
      delete side.rendered_prompt
    }
    return body
  }
  const leftId = request.left_model ?? request.model
  const rightId = request.right_model ?? request.model
  if (proxy.mode !== 'different_models') {
    body.left.model = leftId
    body.right.model = rightId
  }
  for (const side of [body.left, body.right]) {
    if (side.reported_model === ECHO_REQUESTED) side.reported_model = side.model
  }
  body.model_identity = leftId === rightId ? 'same_id' : 'asserted_by_operator'
  body.uncontrolled_detail = body.uncontrolled_detail || []
  if (body.model_identity === 'asserted_by_operator') {
    body.uncontrolled = [...body.uncontrolled, 'model identity']
    body.uncontrolled_detail = [...body.uncontrolled_detail, {
      name: 'model identity',
      reason: `the operator declared \`${leftId}\` and \`${rightId}\` to be the same weights, and nothing here checked it`,
    }]
  }
  return body
}

function answerCompare(req, res, raw) {
  const request = JSON.parse(raw)
  proxy.lastRequest = { body: request, authorization: req.headers.authorization ?? null }
  const reply = (status, value) => {
    res.writeHead(status, { 'content-type': 'application/json' })
    res.end(JSON.stringify(value))
  }
  if (proxy.requireKey && req.headers.authorization !== `Bearer ${proxy.requireKey}`) {
    return reply(401, { error: { message: 'missing or invalid API key', type: 'authentication_error' } })
  }
  if (proxy.mode === 'hang') {
    // Never answers. The only way this connection closes is the page giving up.
    const hung = { closed: false }
    proxy.hung = hung
    res.on('close', () => { hung.closed = true })
    return undefined
  }
  if (proxy.mode === 'refused') {
    return reply(400, { error: { message: 'studio does not hold llama-3.2-1b; it holds qwen3:8b' } })
  }
  return reply(200, comparisonFor(BODIES[proxy.mode], request))
}

const proxyServer = createServer((req, res) => {
  if (proxy.cors === 'allow') {
    res.setHeader('access-control-allow-origin', '*')
    // A JSON POST carrying a bearer token is preflighted, so a proxy has to
    // allow both headers before this page can send either.
    res.setHeader('access-control-allow-headers', 'content-type, authorization')
  }
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end() }

  if (req.url === '/v1/health') {
    res.writeHead(proxy.detail === 'empty' ? 503 : 200, { 'content-type': 'application/json' })
    return res.end(JSON.stringify(healthBody()))
  }

  if (req.url === '/v1/fabric/compare' && req.method === 'POST') {
    let raw = ''
    req.on('data', (chunk) => { raw += chunk })
    req.on('end', () => answerCompare(req, res, raw))
    return undefined
  }
  res.writeHead(404, { 'content-type': 'application/json' })
  return res.end(JSON.stringify({ error: 'unknown' }))
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
const browser = await launchBrowser({ purpose: 'the divergence view smoke', headless: 'new' })

async function until(predicate, ms = 5000) {
  const start = Date.now()
  while (!predicate()) {
    if (Date.now() - start > ms) return false
    await new Promise((done) => setTimeout(done, 50))
  }
  return true
}

/* `expect` is 'listed' when the pickers should fill from a disclosed fabric,
   otherwise the data-state the fabric panel should settle on. */
async function openDivergence({ endpoint = proxyEndpoint, expect = 'listed', viewport = { width: 1280, height: 900 } } = {}) {
  const page = await browser.newPage()
  await page.setViewport(viewport)
  const errors = []
  page.on('pageerror', (error) => errors.push(String(error)))
  await page.evaluateOnNewDocument((value) => {
    window.localStorage.clear()
    window.localStorage.setItem('camelid.fabricEndpoint', value)
  }, endpoint)
  await page.goto(`${appOrigin}/#divergence`, { waitUntil: 'networkidle0' })
  await page.waitForSelector('.divergence__form', { timeout: 10000 })
  if (expect === 'listed') {
    // The pickers are populated from the fabric read, which lands after mount.
    await page.waitForFunction(
      () => document.querySelectorAll('.divergence-pick[data-side="left"] option').length > 1,
      { timeout: 10000 },
    )
  } else {
    await page.waitForSelector(`[data-state="${expect}"]`, { timeout: 10000 })
  }
  return { page, errors }
}

async function pick(page, side, node, model) {
  await page.select(`.divergence-pick[data-side="${side}"] select`, node)
  await page.waitForFunction(
    (s) => document.querySelectorAll(`.divergence-pick[data-side="${s}"] select`).length === 2,
    { timeout: 5000 },
    side,
  )
  const selects = await page.$$(`.divergence-pick[data-side="${side}"] select`)
  await selects[1].select(model)
}

async function submit(page, opts = {}) {
  const {
    left = 'studio', leftModel = 'llama-3.2-1b-instruct:latest',
    right = 'desk', rightModel = 'llama-3.2-1b-instruct',
  } = opts
  await pick(page, 'left', left, leftModel)
  await pick(page, 'right', right, rightModel)
  const assertBox = await page.$('[data-testid="divergence-assert"] input')
  if (assertBox) await assertBox.click()
  await page.click('.divergence__form button[type="submit"]')
}

/* Free-text sides, for a proxy that gave no node list to pick from. */
async function typeSides(page, { left, leftModel, right, rightModel }) {
  await page.type('[data-testid="node-input-left"]', left)
  await page.type('.divergence-pick[data-side="left"] input[placeholder="model id"]', leftModel)
  await page.type('[data-testid="node-input-right"]', right)
  await page.type('.divergence-pick[data-side="right"] input[placeholder="model id"]', rightModel)
}

/* Replace a controlled input's value the way React observes it. Selecting by
   triple-click does not select a number input's text in Chrome, so typing over
   it appended instead ("64" became "6200"), which the field's own max then
   refused to submit. */
async function setField(page, selector, value) {
  await page.$eval(selector, (el, next) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
    setter.call(el, next)
    el.dispatchEvent(new Event('input', { bubbles: true }))
  }, value)
  const now = await page.$eval(selector, (el) => el.value)
  assert.equal(now, value, `could not set ${selector}`)
}

/* Leave through the app's own navigation, so the view unmounts while the page
   stays open: closing the page would abort the request whatever the code did. */
async function leaveTo(page, label) {
  await page.evaluate((wanted) => {
    const entries = [...document.querySelectorAll('#camelid-sidebar button, #camelid-sidebar a')]
    const target = entries.find((el) => (el.getAttribute('aria-label') || el.textContent || '').trim() === wanted)
    if (!target) throw new Error(`no navigation entry named ${wanted}`)
    target.click()
  }, label)
}

const textOf = (page, selector) =>
  page.$eval(selector, (el) => el.textContent.replace(/\s+/g, ' ').trim())

const corsHintOf = (page, scope) => page.$eval(`${scope} [data-testid="fabric-cors-hint"]`, (el) => ({
  diagnosis: el.getAttribute('data-diagnosis'),
  command: el.querySelector('.fabric-cmd code')?.textContent.trim(),
}))

console.log('divergence view')

try {
  /* ---- choosing a model, rather than typing one blind ---- */
  resetProxy()
  {
    const { page } = await openDivergence()

    const nodeOptions = await page.$$eval(
      '.divergence-pick[data-side="left"] select option',
      (els) => els.map((el) => el.value).filter(Boolean),
    )
    assert.deepEqual(nodeOptions, ['studio', 'desk', 'mac'], 'every node the proxy disclosed is offered')
    check('the node picker offers what the fabric actually reported')

    await page.select('.divergence-pick[data-side="left"] select', 'studio')
    await page.waitForFunction(
      () => document.querySelectorAll('.divergence-pick[data-side="left"] select').length === 2,
      { timeout: 5000 },
    )
    const models = await page.$$eval(
      '.divergence-pick[data-side="left"] select',
      (els) => [...els[1].options].map((o) => o.value).filter(Boolean),
    )
    assert.deepEqual(models, ['llama-3.2-1b-instruct:latest', 'qwen3:8b'])
    check('the model picker offers exactly what that node reported holding')

    // A node that is not serving has no list, which must not read as "holds nothing".
    await page.select('.divergence-pick[data-side="right"] select', 'mac')
    const note = await textOf(page, '.divergence-pick[data-side="right"] .divergence-pick__note')
    assert.match(note, /not serving/)
    assert.match(note, /no model loaded/)
    check('a node that is not serving explains itself instead of offering an empty list')
    await page.close()
  }

  /* ---- two names are not evidence of the same weights ---- */
  {
    const { page } = await openDivergence()
    await pick(page, 'left', 'studio', 'llama-3.2-1b-instruct:latest')
    await pick(page, 'right', 'desk', 'llama-3.2-1b-instruct')

    await page.waitForSelector('[data-testid="divergence-assert"]', { timeout: 5000 })
    const disabled = await page.$eval('.divergence__form button[type="submit"]', (el) => el.disabled)
    assert.equal(disabled, true, 'differing ids must not be comparable until a human says so')
    const wording = await textOf(page, '[data-testid="divergence-assert"]')
    assert.match(wording, /nothing here can verify they match/)
    check('two differing ids block the comparison until the operator asserts equivalence')

    await page.click('[data-testid="divergence-assert"] input')
    assert.equal(
      await page.$eval('.divergence__form button[type="submit"]', (el) => el.disabled),
      false,
    )
    await page.click('.divergence__form button[type="submit"]')
    await page.waitForSelector('[data-testid="divergence-identity"]', { timeout: 10000 })
    const identity = await page.$eval('[data-testid="divergence-identity"]', (el) => ({
      kind: el.getAttribute('data-identity'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(identity.kind, 'asserted_by_operator', 'the proxy recorded the assertion and the page read it')
    assert.match(identity.text, /Asserted by the operator, not verified/)
    assert.match(
      identity.text,
      /llama-3\.2-1b-instruct:latest and llama-3\.2-1b-instruct were declared to be the same weights/,
      'the result names both ids the claim was made about',
    )
    assert.match(await textOf(page, '[data-testid="divergence-uncontrolled"]'), /model identity/)
    assert.equal(proxy.lastRequest.body.left_model, 'llama-3.2-1b-instruct:latest')
    assert.equal(proxy.lastRequest.body.right_model, 'llama-3.2-1b-instruct')
    check('ticking the assertion unblocks it, and the result records the claim')
    await page.close()
  }

  /* ---- the headline finding ---- */
  {
    const { page, errors } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-verdict"]', { timeout: 10000 })

    const verdict = await page.$eval('[data-testid="divergence-verdict"]', (el) => ({
      kind: el.getAttribute('data-verdict'),
      attributable: el.getAttribute('data-attributable'),
      tone: el.getAttribute('data-tone'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(verdict.kind, 'divergent')
    assert.equal(verdict.attributable, 'true')
    assert.match(verdict.text, /agreed with themselves, and disagreed/)
    check('a divergence between two self-consistent nodes is stated as established')

    const diff = await page.$$eval('.divergence-diff__line', (els) =>
      els.map((el) => ({ op: el.getAttribute('data-op'), text: el.textContent.trim() })))
    assert.deepEqual(diff.map((line) => line.op), ['removed', 'added'])
    assert.match(diff[0].text, /12/)
    assert.match(diff[1].text, /7/)
    check('the difference is rendered as a diff, marked per side')

    const templates = await page.$$eval('.divergence-template', (els) =>
      els.map((el) => el.textContent.replace(/\s+/g, ' ').trim()))
    assert.ok(templates.some((t) => /GET \/props/.test(t)), 'our own template is shown')
    assert.ok(templates.some((t) => /Cutting Knowledge Date/.test(t)), 'the other template is shown')
    check('both advertised chat templates are shown')

    // C1: an advertised template is what an engine publishes. Only a rendered
    // prompt shows what it applied, so nothing about the templates may say so.
    const templatesHead = await textOf(page, '[data-testid="divergence-templates"] h2')
    assert.equal(templatesHead, 'Advertised chat templates')
    const claims = await page.evaluate(() => [
      document.querySelector('.divergence__lede')?.textContent || '',
      document.querySelector('[data-testid="divergence-templates"]')?.textContent || '',
    ].join(' '))
    assert.doesNotMatch(claims, /\bappl(y|ied)\b/i, 'an advertised template is not the one applied')
    const headings = await page.$$eval('[data-testid="divergence-templates"] .divergence-template h4', (els) =>
      els.map((el) => el.textContent.replace(/\s+/g, ' ').trim()))
    assert.deepEqual(headings, ['win · advertised via GET /props', 'studio · advertised via POST /api/show'])
    check('the templates are labelled advertised, and nothing calls them applied')

    const renderedWin = await page.$eval('[data-testid="rendered-win"]', (el) => ({
      kind: el.getAttribute('data-rendered-kind'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(renderedWin.kind, 'captured')
    assert.match(renderedWin.text, /rendered via POST \/apply-template/)
    assert.match(renderedWin.text, /What is 7 plus 5\?/)
    const renderedStudio = await page.$eval('[data-testid="rendered-studio"]', (el) => ({
      kind: el.getAttribute('data-rendered-kind'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(renderedStudio.kind, 'unavailable')
    assert.match(renderedStudio.text, /no route that renders a chat prompt/)
    check('a rendered prompt is shown under its own name, and an engine that cannot render says so')

    // Everything except the lede, which states the rule and therefore has to
    // use the word. What must never judge a side is the *result*.
    const results = await page.evaluate(() => {
      const root = document.querySelector('.divergence').cloneNode(true)
      root.querySelector('.divergence__lede')?.remove()
      return root.textContent.toLowerCase()
    })
    for (const banned of ['correct', 'incorrect', 'wrong', 'better', 'worse', 'accurate', 'winner']) {
      assert.ok(!results.includes(banned), `the result must not judge a side, found "${banned}"`)
    }
    check('nothing the comparison reports says which side is correct')

    // "Same bytes" under a cap is a claim about a prefix, so the cap is sent
    // explicitly and the result says how much it covered.
    assert.equal(proxy.lastRequest.body.max_tokens, 64, 'the default cap is sent, not left to the proxy')
    assert.match(await textOf(page, '[data-testid="divergence-token-cap"]'), /capped at 64 tokens/)
    await setField(page, '[data-testid="divergence-max-tokens"]', '200')
    await page.click('.divergence__form button[type="submit"]')
    await page.waitForFunction(
      () => /capped at 200 tokens/.test(document.querySelector('[data-testid="divergence-token-cap"]')?.textContent || ''),
      { timeout: 10000 },
    )
    assert.equal(proxy.lastRequest.body.max_tokens, 200)
    check('the token cap is sent, and the result says how much of each answer was compared')

    assert.deepEqual(errors, [], 'no page errors')
    check('the divergence view raises no page error')
    await page.close()
  }

  /* ---- refusals must not read as findings ---- */
  resetProxy({ mode: 'unstable' })
  {
    const { page } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-verdict"]', { timeout: 10000 })

    const verdict = await page.$eval('[data-testid="divergence-verdict"]', (el) => ({
      attributable: el.getAttribute('data-attributable'),
      tone: el.getAttribute('data-tone'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(verdict.attributable, 'false')
    assert.equal(verdict.tone, 'refused', 'a refusal must not be given a result colour')
    assert.match(verdict.text, /Nothing can be concluded/)
    check('a side that did not repeat itself yields no conclusion')

    const diffKind = await page.$eval('[data-testid="divergence-diff"]', (el) => el.getAttribute('data-diff-kind'))
    assert.equal(diffKind, 'declined', 'no diff is rendered for an unattributable comparison')
    assert.equal(
      await page.$$eval('.divergence-diff__line', (els) => els.length),
      0,
      'a diff here would invite exactly the reading the verdict refused',
    )
    check('no diff is drawn when nothing can be attributed')

    const unstable = await page.$eval(
      '.divergence-side__stability[data-stability="unstable"]',
      (el) => el.textContent.replace(/\s+/g, ' ').trim(),
    )
    assert.match(unstable, /did not agree with itself/)
    assert.match(unstable, /2 distinct answers/)
    check('the unstable side is named, with how many answers it gave')
    await page.close()
  }

  /* ---- the same advertised template, and still a divergence ---- */
  resetProxy({ mode: 'same_template' })
  {
    const { page } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-template-note"]', { timeout: 10000 })
    const note = await page.$eval('[data-testid="divergence-template-note"]', (el) => ({
      unexplained: el.getAttribute('data-unexplained'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(note.unexplained, 'true')
    assert.match(note.text, /advertise byte-identical chat templates, so the advertised template does not explain this difference/)
    assert.doesNotMatch(note.text, /usually the explanation/)
    check('identical advertised templates beside a divergence say they do not explain it')
    await page.close()
  }

  resetProxy({ mode: 'different_models' })
  {
    const { page } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-verdict"]', { timeout: 10000 })
    const verdict = await textOf(page, '[data-testid="divergence-verdict"]')
    assert.match(verdict, /Not comparable/)
    assert.match(verdict, /qwen3:8b/)
    assert.doesNotMatch(verdict, /disagree/, 'a model swap is not a disagreement between engines')
    assert.equal(
      await page.$eval('[data-testid="divergence-verdict"]', (el) => el.getAttribute('data-attributable')),
      'false',
    )
    check('two different models are reported as not comparable, never as divergence')
    await page.close()
  }

  /* ---- disclosure on a verdict that WAS reached ---- */
  resetProxy({ mode: 'lmstudio' })
  {
    const { page } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-caveats"]', { timeout: 10000 })

    const caveats = await textOf(page, '[data-testid="divergence-caveats"]')
    assert.match(caveats, /exposes no prompt template/)
    const seed = await textOf(page, '[data-testid="divergence-uncontrolled"] [data-uncontrolled="seed"]')
    assert.match(seed, /^seed — desk \(lmstudio\) runs an engine whose documented completion API has no seed parameter/)
    check('an uncontrolled seed and a missing template are disclosed on the finding itself')

    const items = await page.$$eval('[data-testid="divergence-uncontrolled"] li', (els) =>
      els.map((el) => ({ name: el.getAttribute('data-uncontrolled'), text: el.textContent.replace(/\s+/g, ' ').trim() })))
    assert.deepEqual(items.map((item) => item.name), ['seed', 'model identity'])
    assert.match(items[1].text, /^model identity — the operator declared/)
    assert.doesNotMatch(items[1].text, /parameter/, 'model identity is not a parameter an engine lacks')
    check('each uncontrolled item is listed once, with its own reason')

    assert.equal(
      await page.$eval('[data-testid="divergence-verdict"]', (el) => el.getAttribute('data-attributable')),
      'true',
      'disclosure is not the same as withholding the verdict both sides earned',
    )
    const unknowns = await page.$$eval('[data-unknown="true"]', (els) => els.length)
    assert.ok(unknowns >= 1, 'LM Studio publishes no version, which must render as an explicit unknown')
    check('a missing engine version is an explicit unknown, not a blank')
    await page.close()
  }

  /* ---- a refused request is not an empty comparison ---- */
  resetProxy({ mode: 'refused' })
  {
    const { page } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-problem"]', { timeout: 10000 })
    const problem = await textOf(page, '[data-testid="divergence-problem"]')
    assert.match(problem, /does not hold llama-3\.2-1b/)
    assert.equal(await page.$('[data-testid="divergence-verdict"]'), null,
      'a refused request must never render as a comparison that found nothing')
    check('a refused comparison shows the refusal, not an empty result')
    await page.close()
  }

  /* ---- no node list: each reason is its own fact ---- */
  resetProxy({ detail: 'withheld' })
  {
    const { page, errors } = await openDivergence({ expect: 'withheld' })
    const state = await textOf(page, '[data-testid="divergence-fabric-state"]')
    assert.match(state, /not bound to loopback/, 'the operator is told why there is no list')
    assert.match(state, /type each node's label/)
    assert.equal(await page.$('[data-testid="divergence-no-nodes"]'), null, 'withheld detail is not an empty fabric')
    assert.equal(await page.$('.divergence-pick select'), null, 'there is no list to pick from, so none is drawn')
    check('a proxy that withholds its node detail offers typed labels and says why, never "no nodes"')

    await typeSides(page, { left: 'studio', leftModel: 'llama-3.2-1b', right: 'desk', rightModel: 'llama-3.2-1b' })
    await page.click('.divergence__form button[type="submit"]')
    await page.waitForSelector('[data-testid="divergence-verdict"]', { timeout: 10000 })
    assert.equal(proxy.lastRequest.body.left, 'studio')
    assert.equal(proxy.lastRequest.body.right, 'desk')
    check('typed labels reach the proxy, which resolves them itself')

    assert.equal('left_model' in proxy.lastRequest.body, false)
    assert.equal('right_model' in proxy.lastRequest.body, false)
    assert.equal(proxy.lastRequest.body.model, 'llama-3.2-1b')
    const identity = await page.$eval('[data-testid="divergence-identity"]', (el) => ({
      kind: el.getAttribute('data-identity'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(identity.kind, 'same_id')
    assert.match(identity.text, /same id, llama-3\.2-1b/)
    check("one name on both sides sends no per-side ids, so the proxy's alias table still applies")
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  resetProxy({ detail: 'empty' })
  {
    const { page } = await openDivergence({ expect: 'empty' })
    assert.match(await textOf(page, '[data-testid="divergence-no-nodes"]'), /has no nodes/)
    assert.equal(await page.$('[data-testid="divergence-fabric-state"]'), null, 'an empty fabric is not a failure')
    check('a proxy with genuinely no nodes says so, and only then')
    await page.close()
  }

  resetProxy()
  {
    const { page } = await openDivergence({ endpoint: '127.0.0.1:9', expect: 'problem' })
    const state = await page.$eval('[data-testid="divergence-fabric-state"]', (el) => ({
      code: el.getAttribute('data-code'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(state.code, 'unreachable')
    assert.match(state.text, /No fabric proxy answered/)
    assert.equal(await page.$('[data-testid="divergence-no-nodes"]'), null, 'an unreachable proxy is not an empty one')
    check('an unreachable proxy is named as such, never as an empty node list')
    await page.close()
  }

  /* ---- a proxy that does not allow this page's origin, the real default ---- */
  resetProxy({ cors: 'none' })
  {
    const { page, errors } = await openDivergence({ expect: 'problem' })
    const code = await page.$eval('[data-testid="divergence-fabric-state"]', (el) => el.getAttribute('data-code'))
    assert.equal(code, 'origin_not_allowed')
    const hint = await corsHintOf(page, '[data-testid="divergence-fabric-state"]')
    assert.equal(hint.diagnosis, 'blocked')
    assert.equal(hint.command, `camelid fabric serve --cors-origin ${appOrigin}`)
    assert.equal(await page.$('[data-testid="divergence-no-nodes"]'), null)
    check("a proxy that does not allow this page's origin shows the exact --cors-origin command")

    await typeSides(page, { left: 'studio', leftModel: 'llama-3.2-1b', right: 'desk', rightModel: 'llama-3.2-1b' })
    await page.click('.divergence__form button[type="submit"]')
    await page.waitForSelector('[data-testid="divergence-problem"]', { timeout: 10000 })
    assert.equal(await page.$eval('[data-testid="divergence-problem"]', (el) => el.getAttribute('data-code')), 'unreachable')
    const requestHint = await corsHintOf(page, '[data-testid="divergence-problem"]')
    assert.equal(requestHint.command, `camelid fabric serve --cors-origin ${appOrigin}`)
    // Whether the body itself reached this fake depends on the browser's
    // preflight cache, which an earlier scenario (CORS on) may have filled. It
    // is not asserted: either way the page cannot read the answer, and that is
    // what it must say.
    check('a comparison whose answer the browser withheld offers the same fix')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- a proxy with client keys ---- */
  resetProxy({ requireKey: 'k-9f3a' })
  {
    const { page } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-problem"][data-code="key_required"]', { timeout: 10000 })
    assert.match(await textOf(page, '[data-testid="divergence-problem"]'), /requires a client key/)
    assert.equal(proxy.lastRequest.authorization, null)
    assert.equal(await page.$('[data-testid="divergence-verdict"]'), null)
    check('a keyed proxy that got no key says a client key is needed')

    await page.type('[data-testid="divergence-client-key"]', 'wrong')
    await page.click('.divergence__form button[type="submit"]')
    await page.waitForSelector('[data-testid="divergence-problem"][data-code="key_refused"]', { timeout: 10000 })
    await setField(page, '[data-testid="divergence-client-key"]', 'k-9f3a')
    await page.click('.divergence__form button[type="submit"]')
    await page.waitForSelector('[data-testid="divergence-verdict"]', { timeout: 10000 })
    assert.equal(proxy.lastRequest.authorization, 'Bearer k-9f3a')
    check('a wrong key is named as refused, and the right one is sent as a bearer token')

    const kept = await page.evaluate(() => JSON.stringify({
      local: { ...window.localStorage },
      session: { ...window.sessionStorage },
      href: window.location.href,
    }))
    assert.doesNotMatch(kept, /k-9f3a/, 'a secret the operator typed must not outlive the page')
    check('the client key is never written to browser storage or the address bar')
    await page.close()
  }

  /* ---- a proxy that never answers ---- */
  resetProxy({ mode: 'hang' })
  {
    const { page } = await openDivergence()
    await submit(page)
    assert.ok(await until(() => proxy.hung !== null), 'the comparison reached the proxy')
    assert.equal(await textOf(page, '.divergence__form button[type="submit"]'), 'Asking both nodes…')
    await leaveTo(page, 'Cluster')
    await page.waitForFunction(() => !document.querySelector('.divergence__form'), { timeout: 5000 })
    assert.ok(await until(() => proxy.hung.closed), 'leaving the view must abort the request it started')
    check('leaving the page aborts a comparison still waiting on the proxy')
    await page.close()
  }

  /* ---- differences the text alone would hide ---- */
  resetProxy({ mode: 'eol' })
  {
    const { page, errors } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-diff"]', { timeout: 10000 })
    const lines = await page.$$eval('.divergence-diff__line', (els) => els.map((el) => ({
      op: el.getAttribute('data-op'),
      marker: el.querySelector('[data-eol-marker]')?.textContent.trim() ?? null,
    })))
    assert.deepEqual(lines.map((line) => line.op), ['same', 'removed', 'added'])
    assert.deepEqual(lines.map((line) => line.marker), [null, '␍␊ CRLF', 'no newline at end'])
    check('a difference that is only a line ending is marked, not drawn as two identical lines')

    const warning = await textOf(page, '[data-testid="reported-model-studio"]')
    assert.match(warning, /response named llama-3\.2-3b-instruct, not the requested llama-3\.2-1b-instruct/)
    assert.equal(await page.$('[data-testid="reported-model-win"]'), null,
      'a node that named the model it was asked for carries no warning')
    check('a node that answered under another model name is shown prominently')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- a proxy from before these fields ---- */
  resetProxy({ legacy: true })
  {
    const { page, errors } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-verdict"]', { timeout: 10000 })
    const identity = await page.$eval('[data-testid="divergence-identity"]', (el) => ({
      kind: el.getAttribute('data-identity'),
      text: el.textContent.replace(/\s+/g, ' ').trim(),
    }))
    assert.equal(identity.kind, 'not_reported')
    assert.match(identity.text, /did not record how model identity was established/)
    assert.equal(
      await page.$$eval('[data-testid="divergence-uncontrolled"] [data-unknown="true"]', (els) => els.length),
      1,
      'an unreported list is unknown, never "nothing uncontrolled"',
    )
    assert.equal(await page.$$eval('[data-eol-marker]', (els) => els.length), 0, 'no eol, rendered as before')
    assert.equal(
      await page.$eval('[data-testid="rendered-win"]', (el) => el.getAttribute('data-rendered-kind')),
      'not_reported',
      'a proxy that captured no render is not one whose render failed',
    )
    const legacyHeadings = await page.$$eval('[data-testid="divergence-templates"] .divergence-template h4', (els) =>
      els.map((el) => el.textContent.replace(/\s+/g, ' ').trim()))
    assert.deepEqual(legacyHeadings, ['win · advertised via GET /props', 'studio · advertised via POST /api/show'],
      "an older proxy's template is shown as advertised, which is what it always was")
    check('an older proxy that records no identity says so, rather than implying one')
    assert.deepEqual(errors, [], 'no page errors')
    await page.close()
  }

  /* ---- layout ---- */
  resetProxy()
  {
    const { page } = await openDivergence({ viewport: { width: 390, height: 844 } })
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-verdict"]', { timeout: 10000 })
    const overflow = await page.evaluate(() =>
      document.documentElement.scrollWidth - document.documentElement.clientWidth)
    assert.ok(overflow <= 1, `horizontal overflow of ${overflow}px at 390px`)
    check('the view fits a 390px phone without horizontal overflow')
    await page.close()
  }

  console.log(`\ndivergence view smoke: ${checks} checks passed`)
} finally {
  await browser.close()
  appServer.close()
  proxyServer.close()
}
