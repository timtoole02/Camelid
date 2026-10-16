// Validate the Windows CUDA receipts independently of the historical CPU bundle.
// node qa/bonsai2/check-cuda-evidence.mjs
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
const read = async name => JSON.parse(await readFile(new URL(`cuda/${name}`, import.meta.url), 'utf8'))
const source = await readFile(new URL('../../src/bonsai2.rs', import.meta.url), 'utf8')
const identities = new Map([...source.matchAll(/filename: "Ternary-Bonsai-2-(2|4|9|27)B-PTQ1_0\.gguf",[\s\S]*?sha256: "([a-f0-9]{64})"/g)].map(m => [+m[1], m[2]]))
assert.equal(identities.size, 4)
for (const size of [2, 4, 9, 27]) {
  const filename = `Ternary-Bonsai-2-${size}B-PTQ1_0.gguf`
  const api = await read(`${size}b-api-qualified.json`)
  assert.equal(api.model, filename)
  assert.equal(api.selected_backend, 'bonsai2_cuda_runtime')
  for (const name of ['text', 'tool_call', 'tool_result', 'context_recall', 'sampled_text', 'vision_spatial']) {
    assert.equal(api.cases.find(c => c.name === name)?.pass, true, `${size}B ${name}`)
  }
  const gui = await read(`${size}b-gui-supported.json`)
  assert.equal(gui.model, filename)
  assert.equal(gui.selected_backend, 'bonsai2_cuda_runtime')
  assert.equal(gui.operating_system, 'windows')
  assert.equal(gui.architecture, 'x86_64')
  assert.equal(gui.support_level, 'supported_windows_runtime')
  assert.ok(gui.generation_ready && gui.vision_ready && gui.image_input_present)
  assert.equal(gui.image_limits, 'GUI defaults')
  assert.match(gui.text_answer, /Paris/i)
  assert.match(gui.answer, /red/i)
  for (const answer of [gui.text_answer, gui.answer]) {
    assert.match(answer, /\bverified\b/i)
    assert.doesNotMatch(answer, /unverified/i)
  }
  const native = await read(`${size}b-native-benchmark.json`)
  const reference = await read(`${size}b-reference-benchmark.json`)
  assert.equal(native.target, filename)
  assert.equal(reference.model, filename)
  for (const receipt of [native, reference]) {
    assert.equal(receipt.target_sha256, identities.get(size))
    assert.equal(receipt.draft_sha256, identities.get(2))
    assert.equal(receipt.backend, 'CUDA')
    assert.equal(receipt.max_tokens, 32)
    assert.equal(receipt.draft_tokens, 3)
    assert.equal(receipt.threads, 8)
    assert.deepEqual(receipt.cases.map(c => c.name), ['count', 'code', 'explain'])
  }
  for (const row of native.cases) {
    assert.equal(row.parity, true)
    assert.deepEqual(row.baseline.ids, row.speculative.ids)
    for (const result of [row.baseline, row.speculative]) {
      assert.equal(result.backend, 'cuda')
      assert.equal(result.ids.length, 32)
      assert.ok(result.seconds > 0 && result.warmup_seconds.length > 0)
    }
    assert.ok(row.speculative.speculative.proposed > 0)
    assert.ok(row.speculative.speculative.accepted <= row.speculative.speculative.proposed)
    assert.ok(Math.abs(row.total_speedup - row.baseline.seconds / row.speculative.seconds) < 1e-9)
    const expected = reference.cases.find(c => c.name === row.name).baseline.ids
    assert.deepEqual(row.reference.expected_ids, expected)
    assert.equal(row.reference.prefix_matches, JSON.stringify(row.baseline.ids) === JSON.stringify(expected))
  }
  const plain = api.cases.find(c => c.name === 'sampled_continuation')
  const draft = (await read(`${size}b-sampled-draft.json`)).cases.find(c => c.name === 'sampled_continuation')
  assert.ok(plain?.pass && draft?.pass)
  assert.equal(plain.response.camelid.generated_token_ids.length, 32)
  assert.deepEqual(draft.response.camelid.generated_token_ids, plain.response.camelid.generated_token_ids)
  console.log(`${size}B: CUDA API, GUI text/image, seeded drafting and reference benchmark receipts verified`)
}
assert.equal((await read('cache-parity.json')).pass, true)
console.log('CUDA recurrent/KV rollback and batch verification receipt verified')
