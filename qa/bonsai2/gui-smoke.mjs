// Run against a dedicated Camelid test server, never the user's active server.
// node qa/bonsai2/gui-smoke.mjs MODEL_PATH OUTPUT_PREFIX [API_URL] [--send] [--default-image]
import assert from 'node:assert/strict'
import { basename, resolve } from 'node:path'
import { writeFile } from 'node:fs/promises'
import { launchBrowser } from '../../frontend/scripts/lib/launch-browser.mjs'

const [modelPath, outputPrefix, api = 'http://127.0.0.1:28182'] = process.argv.slice(2)
assert.ok(modelPath && outputPrefix, 'MODEL_PATH and OUTPUT_PREFIX are required')
const send = process.argv.includes('--send')
const defaultImage = process.argv.includes('--default-image')
const requireCuda = process.argv.includes('--require-cuda')
const requireSupported = process.argv.includes('--require-supported')
const loaded = await fetch(`${api}/api/models/load`, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ path: resolve(modelPath), replace: true, id: 'bonsai2-gui-smoke' }),
})
const loadResult = await loaded.json()
assert.ok(loaded.ok, JSON.stringify(loadResult))
let health
for (let i = 0; i < 120; i++) {
  health = await (await fetch(`${api}/v1/health`)).json()
  if (health.generation_ready && health.vision_ready) break
  await new Promise(done => setTimeout(done, 1000))
}
assert.equal(health.generation_ready, true)
assert.equal(health.vision_ready, true)
if (requireCuda) {
  assert.equal(health.execution_plan?.selected_backend, 'bonsai2_cuda_runtime')
  assert.equal(health.execution_plan?.cuda_resident_active, true)
}
if (requireSupported) assert.equal(health.execution_plan?.support_level, 'supported_windows_runtime')
console.log(`${basename(modelPath)}: generation and vision ready (${health.build})`)
const browser = await launchBrowser({ purpose: 'Bonsai 2 image attachment smoke', headless: true, protocolTimeout: 900000 })
try {
  const page = await browser.newPage()
  await page.setViewport({ width: 1440, height: 1000 })
  // Bound this inference smoke while retaining the UI's real attachment,
  // message construction, streaming renderer and backend request path.
  if (send) {
    await page.setRequestInterception(true)
    page.on('request', req => {
      if (req.url().endsWith('/v1/chat/completions') && req.method() === 'POST') {
        const body = JSON.parse(req.postData())
        body.max_tokens = 16
        if (!defaultImage) {
          body.camelid_image_min_tokens = 4
          body.camelid_image_max_tokens = 4
        }
        if (body.messages.some(m => Array.isArray(m.content))) {
          assert.ok(body.messages.some(m => Array.isArray(m.content) && m.content.some(p => p.type === 'image_url')))
        }
        req.continue({ postData: JSON.stringify(body) })
      } else req.continue()
    })
  }
  await page.goto(`${api}/#chat`, { waitUntil: 'networkidle2', timeout: 60000 })
  const input = await page.waitForSelector('input[accept="image/png,image/jpeg"]', { timeout: 30000 })
  const receipt = {
    model: basename(modelPath), platform: process.platform, backend_build: health.build,
    selected_backend: health.execution_plan?.selected_backend, date: new Date().toISOString(),
    support_level: health.execution_plan?.support_level,
    operating_system: health.execution_plan?.operating_system, architecture: health.execution_plan?.architecture,
    generation_ready: health.generation_ready, vision_ready: health.vision_ready,
    image_input_present: true, image_limits: defaultImage ? 'GUI defaults' : 'four merged tokens',
  }
  if (send && requireCuda) {
    await page.type('textarea[aria-label="Message Camelid"]', 'What is the capital of France? Answer in one sentence.')
    const responsePromise = page.waitForResponse(r => r.url().endsWith('/v1/chat/completions') && r.request().method() === 'POST', { timeout: 900000 })
    await page.click('button[aria-label="Send message"]')
    const textResponse = await responsePromise
    assert.ok(textResponse.ok(), 'CUDA text request failed')
    await page.waitForFunction(() => {
      const last = [...document.querySelectorAll('.cxturn--assistant')].at(-1)
      return last && !last.classList.contains('is-streaming') && last.querySelector('.cxturn__body')?.innerText.trim()
    }, { timeout: 900000 })
    receipt.text_answer = await page.$eval('.cxturn--assistant:last-of-type .cxturn__body', el => el.innerText)
    if (!/Paris/i.test(receipt.text_answer)) {
      // Headers can be 200 even when cold CUDA setup fails inside the SSE body.
      // Keep that diagnostic in the runner log instead of losing it behind the
      // GUI's short error message. Successful public receipts contain no trace.
      console.error('CUDA text SSE:', await textResponse.text().catch(error => error.message))
    }
    assert.match(receipt.text_answer, /Paris/i)
    // CUDA callbacks can arrive as one burst; the browser must label its
    // full-request clock rather than presenting callback spacing as decode rate.
    if (process.argv.includes('--require-request-rate')) {
      assert.match(receipt.text_answer, /tok\/s request/)
      receipt.output_rate_source = 'client_request'
    }
    if (requireSupported) {
      assert.doesNotMatch(receipt.text_answer, /unverified/i)
      assert.match(receipt.text_answer, /\bverified\b/i)
    }
    await page.screenshot({ path: `${outputPrefix}-text.png`, fullPage: true })
    console.log(`${basename(modelPath)}: text answered through CUDA GUI`)
  }
  if (send) {
    await input.uploadFile(resolve('qa/bonsai2/red.png'))
    await page.type('textarea[aria-label="Message Camelid"]', 'What color is the image? Answer with one color.')
    const responsePromise = page.waitForResponse(r => r.url().endsWith('/v1/chat/completions') && r.request().method() === 'POST', { timeout: 900000 })
    await page.click('button[aria-label="Send message"]')
    console.log(`${basename(modelPath)}: image submitted through GUI`)
    const response = await responsePromise
    assert.ok(response.ok(), `image request returned ${response.status()}`)
    // Read the rendered completion: long SSE responses can be evicted from
    // Chromium's inspector cache even though the GUI received them correctly.
    await page.waitForFunction(expected => {
      const turns = [...document.querySelectorAll('.cxturn--assistant')]
      const last = turns.at(-1)
      return turns.length >= expected && last && !last.classList.contains('is-streaming') && last.querySelector('.cxturn__body')?.innerText.trim()
    }, { timeout: 900000 }, requireCuda ? 2 : 1)
    receipt.answer = await page.$eval('.cxturn--assistant:last-of-type .cxturn__body', el => el.innerText)
    // The body includes footer text such as CLIENT-MEASURED, whose suffix
    // matches /red/i even when the generated answer is the wrong color.
    assert.match(receipt.answer.split(/\r?\n/, 1)[0], /^\s*red[.!]?\s*$/i)
    if (requireSupported) {
      assert.doesNotMatch(receipt.answer, /unverified/i)
      assert.match(receipt.answer, /\bverified\b/i)
    }
  }
  await page.screenshot({ path: `${outputPrefix}.png`, fullPage: true })
  await writeFile(`${outputPrefix}.json`, JSON.stringify(receipt, null, 2))
  console.log(JSON.stringify(receipt))
} finally {
  await browser.close()
}
