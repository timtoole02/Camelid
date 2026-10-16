// Dedicated test server only. MODEL_PATH OUTPUT.json [API] [CASE,...]
import assert from 'node:assert/strict'
import http from 'node:http'
import { basename, resolve } from 'node:path'
import { readFile, writeFile } from 'node:fs/promises'

const [model, output, api = 'http://127.0.0.1:28182', selected = 'text,tools,context,sampling,vision'] = process.argv.slice(2)
assert.ok(model && output, 'MODEL_PATH OUTPUT required')
const cases = new Set(selected.split(','))
function request(path, body) {
  return new Promise((done, reject) => {
    const wire = JSON.stringify(body)
    const req = http.request(new URL(path, api), {
      method: 'POST', headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(wire) },
    }, res => {
      let text = ''
      res.on('data', part => { text += part })
      res.on('error', reject)
      res.on('end', () => {
        try { assert.equal(res.statusCode, 200, text); done(JSON.parse(text)) } catch (error) { reject(error) }
      })
    })
    req.setTimeout(1800000, () => req.destroy(new Error('qualification request timed out')))
    req.on('error', reject)
    req.end(wire)
  })
}
await request('/api/models/load', { path: resolve(model), replace: true, id: 'bonsai2-qualification' })
let health
for (let i = 0; i < 180; i++) {
  health = await (await fetch(`${api}/v1/health`)).json()
  if (health.generation_ready && health.vision_ready) break
  await new Promise(done => setTimeout(done, 1000))
}
assert.equal(health.generation_ready, true)
assert.equal(health.vision_ready, true)
const report = { model: basename(model), platform: process.platform, date: new Date().toISOString(),
  backend_build: health.build, selected_backend: health.execution_plan?.selected_backend, cases: [] }
async function check(name, body, validate) {
  const start = performance.now()
  const response = await request('/v1/chat/completions', {
    model: 'bonsai2-qualification', temperature: 0, max_tokens: 64,
    chat_template_kwargs: { enable_thinking: false }, ...body,
  })
  const row = { name, seconds: (performance.now() - start) / 1000, response }
  report.cases.push(row)
  try { validate(response); row.pass = true } catch (error) { row.pass = false; row.error = error.message }
  await writeFile(output, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({ name, pass: row.pass, seconds: row.seconds, message: response.choices?.[0]?.message }))
  assert.ok(row.pass, row.error)
  return response
}
const user = content => ({ role: 'user', content })
const content = response => response.choices[0].message.content ?? ''
if (cases.has('text')) await check('text', { messages: [user('What is the capital of France? Answer in one sentence.')] }, r => assert.match(content(r), /Paris/))
if (cases.has('tools')) {
  const toolRequest = JSON.parse(await readFile(new URL('./tool-request.json', import.meta.url), 'utf8'))
  const call = await check('tool_call', toolRequest, r => {
    const tool = r.choices[0].message.tool_calls?.[0]
    assert.equal(tool?.function.name, 'get_weather')
    assert.equal(JSON.parse(tool.function.arguments).city, 'Berlin')
    assert.equal(r.choices[0].finish_reason, 'tool_calls')
  })
  const message = call.choices[0].message
  await check('tool_result', {
    ...toolRequest, max_tokens: 40,
    messages: [...toolRequest.messages, { ...message, content: null }, {
      role: 'tool', tool_call_id: message.tool_calls[0].id,
      content: '{"city":"Berlin","temperature_c":21,"condition":"sunny"}',
    }],
  }, r => { assert.match(content(r), /21/); assert.match(content(r), /sunny/i) })
}
if (cases.has('context')) await check('context_recall', {
  messages: [user('Remember: the secret word is ORCHID.\n' + 'A stone rests beside the river. '.repeat(144) + '\nWhat is the secret word? Answer with just that word.')], max_tokens: 16,
}, r => { assert.ok(r.usage.prompt_tokens >= 1024); assert.match(content(r), /orchid/i) })
if (cases.has('sampling')) await check('sampled_text', {
  messages: [user('What is 2 plus 2? Answer with one number.')], temperature: 0.7, seed: 42, max_tokens: 16,
}, r => assert.match(content(r), /4|four/i))
if (cases.has('vision')) await check('vision_spatial', {
  messages: [user([
    { type: 'text', text: 'What color is on the left half of the image? Answer with one color.' },
    { type: 'image_url', image_url: { url: `data:image/png;base64,${(await readFile(new URL('./blue-yellow.png', import.meta.url))).toString('base64')}` } },
  ])], max_tokens: 16,
}, r => assert.match(content(r), /blue/i))
// Capture once with ordinary decoding and once with the draft enabled; compare
// generated_token_ids from the two receipts to check seeded request sampling.
if (cases.has('sampling_long')) await check('sampled_continuation', {
  messages: [user('Explain how rainbows form in three sentences.')],
  temperature: 0.7, seed: 42, max_tokens: 32,
}, r => assert.equal(r.camelid?.generated_token_ids?.length, 32))
