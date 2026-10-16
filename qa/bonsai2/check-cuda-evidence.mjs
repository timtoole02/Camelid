// Validate the Windows CUDA receipts independently of the historical CPU bundle.
// node qa/bonsai2/check-cuda-evidence.mjs
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
const read = async name => JSON.parse(await readFile(new URL(`cuda/${name}`, import.meta.url), 'utf8'))
const source = await readFile(new URL('../../src/bonsai2.rs', import.meta.url), 'utf8')
const identities = new Map([...source.matchAll(/filename: "Ternary-Bonsai-2-(2|4|9|27)B-PTQ1_0\.gguf",[\s\S]*?sha256: "([a-f0-9]{64})"/g)].map(m => [+m[1], m[2]]))
assert.equal(identities.size, 4)
function checkTimedResult(result, tokens, label) {
  assert.equal(result.ids.length, tokens, label)
  assert.ok(Number.isFinite(result.seconds) && result.seconds > 0, label)
  assert.ok(Math.abs(result.total_tokens_per_second - tokens / result.seconds) < 1e-9, label)
  assert.equal(result.runs?.length, 3, `${label}: three repetitions required`)
  assert.equal(result.repeated_ids_match, true, `${label}: greedy runs changed IDs`)
  for (const run of result.runs) {
    assert.deepEqual(run.ids, result.ids, `${label}: repeated IDs`)
    assert.ok(Number.isFinite(run.seconds) && run.seconds > 0, label)
    assert.ok(Math.abs(run.total_tokens_per_second - tokens / run.seconds) < 1e-9, label)
  }
  const median = result.runs.map(run => run.seconds).sort((a, b) => a - b)[1]
  assert.equal(result.seconds, median, `${label}: selected run is not the median`)
}
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
    assert.equal(receipt.context, 2048)
    assert.equal(receipt.kv_type, 'f16')
    assert.equal(receipt.warmups_per_case, 1)
    assert.equal(receipt.warmup_tokens, 8)
    assert.equal(receipt.repetitions, 3)
    assert.deepEqual(receipt.cases.map(c => c.name), ['count', 'code', 'explain'])
  }
  assert.equal(reference.completion_status, 'complete')
  assert.match(reference.devices, /CUDA\d/)
  assert.match(reference.version, /e3de71522/)
  for (const mode of ['baseline', 'speculative']) {
    const launch = reference.modes[mode]
    assert.equal(launch.error, undefined, `${size}B reference ${mode}: incomplete launch`)
    assert.match(launch.log_sha256, /^[a-f0-9]{64}$/)
    const offloads = launch.offload.flatMap(line => {
      const match = line.match(/offloaded (\d+)\/(\d+) layers to GPU/)
      return match ? [+match[1]] : []
    })
    assert.equal(offloads.length, mode === 'speculative' ? 2 : 1)
    assert.ok(offloads.every(layers => layers > 0), `${size}B reference ${mode}: CPU fallback`)
  }
  for (const row of native.cases) {
    const referenceRow = reference.cases.find(c => c.name === row.name)
    const promptHash = createHash('sha256').update(await readFile(new URL(`${row.name}.chatml`, import.meta.url))).digest('hex')
    assert.equal(row.prompt_sha256, promptHash)
    assert.equal(referenceRow.prompt_sha256, promptHash)
    assert.equal(row.parity, true)
    assert.deepEqual(row.baseline.ids, row.speculative.ids)
    for (const result of [row.baseline, row.speculative]) {
      assert.equal(result.backend, 'cuda')
      checkTimedResult(result, 32, `${size}B native ${row.name}`)
      for (const run of result.runs) assert.equal(run.backend, 'cuda')
      assert.ok(result.seconds > 0 && result.warmup_seconds.length > 0)
    }
    assert.ok(row.speculative.speculative.proposed > 0)
    assert.ok(row.speculative.speculative.accepted <= row.speculative.speculative.proposed)
    assert.ok(Math.abs(row.total_speedup - row.baseline.seconds / row.speculative.seconds) < 1e-9)
    for (const mode of ['baseline', 'speculative']) {
      checkTimedResult(referenceRow[mode], 32, `${size}B reference ${row.name} ${mode}`)
    }
    assert.ok(referenceRow.speculative.timings.draft_n > 0)
    assert.ok(referenceRow.speculative.timings.draft_n_accepted <= referenceRow.speculative.timings.draft_n)
    // The pinned port changes the greedy prose continuation with drafting on
    // 2B and 9B. Keep that observed limitation, while verifying the recorded
    // parity flag rather than treating a reference mismatch as a native pass.
    assert.equal(referenceRow.parity, JSON.stringify(referenceRow.baseline.ids) === JSON.stringify(referenceRow.speculative.ids))
    assert.ok(Math.abs(referenceRow.total_speedup - referenceRow.baseline.seconds / referenceRow.speculative.seconds) < 1e-9)
    const expected = referenceRow.baseline.ids
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
const cache = await read('cache-parity.json')
assert.equal(cache.schema, 1)
assert.equal(cache.pass, true)
assert.equal(cache.platform, 'win32')
assert.match(cache.test_binary_sha256, /^[a-f0-9]{64}$/)
for (const file of ['src/runnable/model.rs', 'src/runnable/speculative.rs', 'src/runnable/hadamard.rs',
  'src/cuda_resident.rs', 'src/cuda_resident/bonsai_spec.rs', 'src/cuda_resident/bonsai_ptq.rs', 'src/cuda_resident/bonsai_ptq.cu']) {
  assert.match(cache.source_sha256[file], /^[a-f0-9]{64}$/, `cache source hash missing: ${file}`)
}
const requiredCacheTests = [
  { name: 'bonsai_cuda_transform_projection_and_embedding', size: null, context: 'default' },
  { name: 'bonsai_cuda_bf16_projection_matches_warp_reference', size: null, context: 'default' },
  ...[2, 4, 9, 27].map(size => ({ name: 'bonsai2_cuda_batch_cache_and_rollback_parity', size, context: 'default' })),
  ...['group-boundary', 'splitk-boundary'].map(context => ({ name: 'bonsai2_cuda_batch_cache_and_rollback_parity', size: 2, context })),
  ...[2, 4, 9, 27].map(size => ({ name: 'bonsai2_cuda_draft_greedy_sampled_stop_and_cancel_parity', size, context: 'default' })),
]
assert.equal(cache.tests.length, requiredCacheTests.length)
for (const required of requiredCacheTests) {
  const filename = required.size ? `Ternary-Bonsai-2-${required.size}B-PTQ1_0.gguf` : null
  const matches = cache.tests.filter(row => row.name === required.name && row.model === filename && row.context === required.context)
  assert.equal(matches.length, 1, `missing or duplicate cache test: ${JSON.stringify(required)}`)
  const row = matches[0]
  assert.equal(row.pass, true)
  assert.equal(row.model_sha256, required.size ? identities.get(required.size) : null)
  assert.ok(Number.isFinite(row.seconds) && row.seconds > 0)
  assert.match(row.log_sha256, /^[a-f0-9]{64}$/)
  assert.match(row.summary, /test result: ok\. 1 passed; 0 failed/)
  assert.equal(row.environment.RAYON_NUM_THREADS, '8')
  assert.equal(row.environment.CAMELID_QWEN35_CUDA, '1')
  assert.equal(row.environment.CAMELID_QWEN35_METAL, '0')
  assert.equal(row.environment.CAMELID_QWEN35_CUDA_MAXPOS, '2048')
  if (required.context !== 'default') assert.equal(row.environment.CAMELID_CUDA_GRAPHS, '1')
}
console.log('All four CUDA cache/rollback and draft parity tests, kernel fixture, and graph boundary receipts verified')
