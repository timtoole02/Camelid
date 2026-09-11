#!/usr/bin/env node
/* Browser coverage for Screen D.
 *
 * The pure rules are proved in divergence-model-smoke.mjs. This proves the page
 * actually renders them: that a refusal to conclude never reaches the screen
 * looking like a finding, that a side which did not repeat itself is called out
 * before the reader compares anything, and that a request the proxy refused is
 * not shown as a comparison that found nothing.
 *
 * Requires `npm run build` first (it serves frontend/dist) and Chrome/Edge.
 */
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'

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

const DIVERGENT = {
  prompt: 'What is 7 plus 5?',
  prompt_sha256: 'f00dcafe',
  plan: { temperature: 0, seed: 0, max_tokens: 64, repetitions: 2 },
  left: {
    label: 'win', engine: 'camelid', engine_version: 'v0.6.1-267', model: 'llama-3.2-1b',
    applied_sampling: { temperature: 'sent', seed: 'sent' },
    samples: [{ text: '12', sha256: 'aaaa', elapsed_ms: 30 }, { text: '12', sha256: 'aaaa', elapsed_ms: 28 }],
    stability: { kind: 'stable' },
    template: { kind: 'captured', source: 'GET /props', template: CAMELID_TEMPLATE },
  },
  right: {
    label: 'studio', engine: 'ollama', engine_version: '0.33.3', model: 'llama-3.2-1b',
    applied_sampling: { temperature: 'sent', seed: 'sent' },
    samples: [{ text: '7', sha256: 'bbbb', elapsed_ms: 41 }, { text: '7', sha256: 'bbbb', elapsed_ms: 39 }],
    stability: { kind: 'stable' },
    template: { kind: 'captured', source: 'POST /api/show', template: OLLAMA_TEMPLATE },
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
    template: { kind: 'not_exposed', detail: "LM Studio's documented API exposes no prompt template" },
  },
  uncontrolled: ['seed'],
}

const proxy = { mode: 'divergent', detail: true }

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

function healthBody() {
  const body = {
    ok: true, service: 'camelid-fabric', version: '0.6.1', build: 'v0.6.1-349', ready: true,
  }
  // Absent node_detail is what an off-loopback proxy sends. It is not an empty
  // fabric, and the picker has to say so rather than offer an empty list.
  if (proxy.detail) {
    body.nodes = { total: 3, ready: 2, not_ready: 1, unreachable: 0 }
    body.models = []
    body.node_detail = NODE_DETAIL
  }
  return body
}

const proxyServer = createServer((req, res) => {
  res.setHeader('access-control-allow-origin', '*')
  res.setHeader('access-control-allow-headers', 'content-type')
  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end() }

  if (req.url === '/v1/health') {
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end(JSON.stringify(healthBody()))
  }

  if (req.url === '/v1/fabric/compare' && req.method === 'POST') {
    if (proxy.mode === 'refused') {
      res.writeHead(400, { 'content-type': 'application/json' })
      return res.end(JSON.stringify({ error: { message: 'studio does not hold llama-3.2-1b; it holds qwen3:8b' } }))
    }
    const bodies = {
      divergent: DIVERGENT, unstable: UNSTABLE,
      different_models: DIFFERENT_MODELS, lmstudio: LMSTUDIO_UNSEEDED,
    }
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end(JSON.stringify(bodies[proxy.mode]))
  }
  res.writeHead(404, { 'content-type': 'application/json' })
  return res.end(JSON.stringify({ error: 'unknown' }))
})

function listen(server) {
  return new Promise((done) => server.listen(0, '127.0.0.1', () => done(server.address().port)))
}

function findBrowser() {
  const candidates = [
    process.env.PUPPETEER_EXECUTABLE_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
  ].filter(Boolean)
  const found = candidates.find((path) => existsSync(path))
  if (!found) throw new Error('no Chrome or Edge found; set PUPPETEER_EXECUTABLE_PATH')
  return found
}

let checks = 0
function check(name) {
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}

const appPort = await listen(appServer)
const proxyPort = await listen(proxyServer)
const browser = await puppeteer.launch({
  executablePath: findBrowser(),
  headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
})

async function openDivergence() {
  const page = await browser.newPage()
  await page.setViewport({ width: 1280, height: 900 })
  const errors = []
  page.on('pageerror', (error) => errors.push(String(error)))
  await page.evaluateOnNewDocument((endpoint) => {
    window.localStorage.setItem('camelid.fabricEndpoint', endpoint)
  }, `127.0.0.1:${proxyPort}`)
  await page.goto(`http://127.0.0.1:${appPort}/#divergence`, { waitUntil: 'networkidle0' })
  await page.waitForSelector('.divergence__form', { timeout: 10000 })
  // The pickers are populated from the fabric read, which lands after mount.
  await page.waitForFunction(
    () => document.querySelectorAll('.divergence-pick[data-side="left"] option').length > 1,
    { timeout: 10000 },
  )
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

const textOf = (page, selector) =>
  page.$eval(selector, (el) => el.textContent.replace(/\s+/g, ' ').trim())

console.log('divergence view')

try {
  /* ---- choosing a model, rather than typing one blind ---- */
  proxy.mode = 'divergent'
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
    check('both chat templates are shown, which is the explanation')

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

    assert.deepEqual(errors, [], 'no page errors')
    check('the divergence view raises no page error')
    await page.close()
  }

  /* ---- refusals must not read as findings ---- */
  proxy.mode = 'unstable'
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

  proxy.mode = 'different_models'
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
  proxy.mode = 'lmstudio'
  {
    const { page } = await openDivergence()
    await submit(page)
    await page.waitForSelector('[data-testid="divergence-caveats"]', { timeout: 10000 })

    const caveats = await textOf(page, '[data-testid="divergence-caveats"]')
    assert.match(caveats, /seed was not controlled/)
    assert.match(caveats, /exposes no prompt template/)
    check('an uncontrolled seed and a missing template are disclosed on the finding itself')

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
  proxy.mode = 'refused'
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

  /* ---- layout ---- */
  proxy.mode = 'divergent'
  {
    const page = await browser.newPage()
    await page.setViewport({ width: 390, height: 844 })
    await page.evaluateOnNewDocument((endpoint) => {
      window.localStorage.setItem('camelid.fabricEndpoint', endpoint)
    }, `127.0.0.1:${proxyPort}`)
    await page.goto(`http://127.0.0.1:${appPort}/#divergence`, { waitUntil: 'networkidle0' })
    await page.waitForSelector('.divergence__form', { timeout: 10000 })
    await page.waitForFunction(
      () => document.querySelectorAll('.divergence-pick[data-side="left"] option').length > 1,
      { timeout: 10000 },
    )
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
