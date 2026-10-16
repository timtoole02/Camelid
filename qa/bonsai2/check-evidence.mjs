// Validate the committed Windows qualification bundle without model downloads.
// node qa/bonsai2/check-evidence.mjs
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

const read = async name => JSON.parse(await readFile(new URL(name, import.meta.url), 'utf8'))
const source = await readFile(new URL('../../src/bonsai2.rs', import.meta.url), 'utf8')
const identities = new Map([...source.matchAll(/filename: "Ternary-Bonsai-2-(2|4|9|27)B-PTQ1_0\.gguf",[\s\S]*?sha256: "([a-f0-9]{64})"/g)].map(m => [+m[1], m[2]]))
assert.equal(identities.size, 4)
for (const size of [2, 4, 9, 27]) {
  const filename = `Ternary-Bonsai-2-${size}B-PTQ1_0.gguf`
  const api = await read(`${size}b-api-qualified.json`)
  assert.equal(api.model, filename)
  assert.deepEqual(api.cases.map(c => c.name).sort(), ['context_recall', 'sampled_text', 'text', 'tool_call', 'tool_result', 'vision_spatial'])
  assert.ok(api.cases.every(c => c.pass === true))
  const reference = await read(`${size}b-reference-text.json`)
  assert.equal(reference.model, filename)
  assert.deepEqual(api.cases.find(c => c.name === 'text').response.camelid.generated_token_ids,
    reference.cases.find(c => c.name === 'text').ids)
  const cpu = await read(`${size}b-cpu-vision.json`)
  assert.equal(cpu.model, filename)
  assert.equal(cpu.cases.find(c => c.name === 'vision_spatial')?.pass, true)
  const gui = await read(`${size}b-gui-supported.json`)
  assert.equal(gui.model, filename)
  assert.equal(gui.operating_system, 'windows')
  assert.equal(gui.architecture, 'x86_64')
  assert.equal(gui.support_level, 'supported_windows_runtime')
  assert.ok(gui.generation_ready && gui.vision_ready && gui.image_input_present)
  assert.equal(gui.image_limits, 'GUI defaults')
  assert.match(gui.answer, /red/i)
  assert.match(gui.answer, /\bverified\b/i)
  assert.doesNotMatch(gui.answer, /unverified/i)
  const bench = await read(`${size}b-draft-benchmark.json`)
  assert.equal(bench.target, filename)
  assert.equal(bench.target_sha256, identities.get(size))
  assert.equal(bench.draft_sha256, identities.get(2))
  assert.deepEqual(bench.cases.map(c => c.name), ['count', 'code', 'explain'])
  for (const row of bench.cases) {
    assert.equal(row.parity, true)
    assert.deepEqual(row.speculative.ids, row.baseline.ids)
    assert.equal(row.baseline.ids.length, 32)
    assert.ok(row.speculative.speculative.proposed > 0)
    assert.ok(row.speculative.speculative.accepted <= row.speculative.speculative.proposed)
    assert.ok(Math.abs(row.total_speedup - row.baseline.seconds / row.speculative.seconds) < 1e-9)
    const expected = reference.cases.find(c => c.name === row.name).ids
    assert.deepEqual(row.reference.expected_ids, expected)
    assert.equal(row.reference.prefix_matches, JSON.stringify(row.baseline.ids) === JSON.stringify(expected))
  }
  console.log(`${size}B: API, CPU vision, supported GUI and three draft comparisons verified`)
}
const plain = (await read('9b-cpu-vision.json')).cases.find(c => c.name === 'sampled_continuation')
const draft = (await read('9b-sampled-draft.json')).cases.find(c => c.name === 'sampled_continuation')
assert.ok(plain.pass && draft.pass)
assert.equal(plain.response.camelid.generated_token_ids.length, 32)
assert.deepEqual(draft.response.camelid.generated_token_ids, plain.response.camelid.generated_token_ids)
assert.equal((await read('cache-parity.json')).pass, true)
console.log('Seeded sampling and real-model cache parity verified')
