import assert from 'node:assert/strict'
import { readFile, readdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'
const root = resolve('qa/bonsai2/cuda-optimized')
const json = async name => JSON.parse(await readFile(resolve(root, name), 'utf8'))
const digest = bytes => createHash('sha256').update(bytes).digest('hex')
const manifest = await json('manifest.json')
assert.equal(manifest.schema, 1)
assert.match(manifest.source_commit, /^[a-f0-9]{40}$/)
for (const [name, hash] of Object.entries(manifest.artifacts)) {
  assert.equal(digest(await readFile(resolve(root, name))), hash, `${name} checksum`)
}
for (const name of await readdir(root)) {
  if (name.endsWith('.json') && name !== 'manifest.json') assert.ok(manifest.artifacts[name], `${name} unbound`)
}
const cache = await json('cache-parity.json')
assert.equal(cache.pass, true)
assert.equal(cache.test_binary_sha256, manifest.binaries.tests.sha256)
assert.equal(cache.host_tests.length, 2)
assert.equal(cache.tests.length, 13)
for (const [path, hash] of Object.entries(cache.source_sha256)) assert.equal(hash, manifest.source_sha256[path], path)
for (const name of ['bonsai_cuda_projection_tiles_match_legacy_bits', 'bonsai_cuda_transform_projection_and_embedding',
  'bonsai_cuda_bf16_projection_matches_warp_reference']) assert.equal(cache.tests.filter(row => row.name === name).length, 1)
for (const context of ['group-boundary', 'splitk-boundary']) {
  assert.equal(cache.tests.filter(row => row.context === context && row.model === 'Ternary-Bonsai-2-2B-PTQ1_0.gguf').length, 1)
}
for (const row of [...cache.host_tests, ...cache.tests]) {
  assert.equal(row.pass, true)
  assert.match(row.summary, /1 passed; 0 failed/)
}
const comparison = await json('comparison.json')
assert.equal(comparison.models.length, 4)
const profile = await json('profile.json')
assert.equal(profile.qualified_profiles.length, 6)
for (const size of [2, 4, 9]) for (const mode of ['baseline', 'tiled']) {
  const rows = profile.qualified_profiles.filter(row => row.size === size && row.projection_policy === mode)
  assert.equal(rows.length, 1)
  assert.equal(rows[0].probe_sha256, manifest.binaries.probe.sha256)
  assert.equal(rows[0].source_commit, manifest.source_commit)
  assert.ok(rows[0].kernel_total_ns > 0)
  assert.ok(rows[0].projection_fraction > 0 && rows[0].projection_fraction <= 1)
}
for (const size of [2, 4, 9, 27]) {
  const native = await json(`${size}b-native-benchmark.json`)
  const reference = await json(`${size}b-reference-benchmark.json`)
  const baseline = await json(`${size}b-baseline-benchmark.json`)
  const modelName = `Ternary-Bonsai-2-${size}B-PTQ1_0.gguf`
  assert.equal(native.target, modelName)
  assert.equal(baseline.target, modelName)
  assert.equal(reference.model, modelName)
  for (const name of ['bonsai2_cuda_batch_cache_and_rollback_parity', 'bonsai2_cuda_draft_greedy_sampled_stop_and_cancel_parity']) {
    const rows = cache.tests.filter(row => row.name === name && row.model_sha256 === native.target_sha256 && row.context === 'default')
    assert.equal(rows.length, 1, `${size}B ${name}`)
    assert.equal(rows[0].environment.CAMELID_QWEN35_CUDA, '1')
  }
  for (const receipt of [native, baseline]) {
    assert.equal(receipt.probe_sha256, manifest.binaries.probe.sha256)
    for (const [path, hash] of Object.entries(receipt.source_sha256)) assert.equal(hash, manifest.source_sha256[path], path)
  }
  assert.equal(native.projection_policy, 'tiled')
  assert.equal(baseline.projection_policy, 'baseline')
  for (const receipt of [native, reference, baseline]) {
    assert.equal(receipt.backend, 'CUDA')
    assert.equal(receipt.repetitions, 3)
    assert.equal(receipt.cases.length, 3)
    assert.equal(receipt.context, 2048)
    assert.equal(receipt.max_tokens, 32)
    assert.equal(receipt.draft_tokens, 3)
    assert.equal(receipt.threads, 8)
    assert.equal(receipt.target_sha256, native.target_sha256)
    assert.equal(receipt.draft_sha256, native.draft_sha256)
    assert.equal(receipt.kv_type, 'f16')
    assert.equal(receipt.warmup_tokens, 8)
    assert.equal(receipt.warmups_per_case, 1)
    for (const row of receipt.cases) for (const mode of ['baseline', 'speculative']) {
      assert.equal(row[mode].runs.length, 3)
      for (const run of row[mode].runs) {
        assert.equal(run.ids.length, 32)
        assert.ok(run.seconds > 0)
        assert.deepEqual(run.ids, row[mode].ids)
        assert.ok(Math.abs(run.total_tokens_per_second - 32 / run.seconds) < 1e-6)
      }
    }
  }
  const summary = comparison.models.find(row => row.size === size)
  for (const row of native.cases) {
    const before = baseline.cases.find(item => item.name === row.name)
    assert.equal(row.parity, true)
    assert.equal(before.parity, true)
    assert.deepEqual(row.baseline.ids, before.baseline.ids, `${size}B ${row.name} optimized target`)
    assert.deepEqual(row.speculative.ids, before.speculative.ids, `${size}B ${row.name} optimized draft`)
    assert.deepEqual(row.baseline.ids, reference.cases.find(item => item.name === row.name).baseline.ids,
      `${size}B ${row.name} reference ordinary`)
    assert.equal(row.prompt_sha256, before.prompt_sha256)
    const entry = summary.cases.find(item => item.name === row.name)
    assert.equal(entry.native_plain_tokens_per_second, row.baseline.total_tokens_per_second)
    assert.equal(entry.native_draft_tokens_per_second, row.speculative.total_tokens_per_second)
    assert.equal(entry.native_speculative_parity, true)
  }
  const gui = await json(`${size}b-gui-supported.json`)
  assert.equal(gui.model, modelName)
  assert.equal(gui.selected_backend, 'bonsai2_cuda_runtime')
  assert.equal(gui.support_level, 'supported_windows_runtime')
  assert.equal(gui.backend_build, manifest.binaries.server.receipt_build_id)
  for (const key of ['generation_ready', 'vision_ready', 'image_input_present']) assert.equal(gui[key], true)
  assert.match(gui.text_answer, /Paris/)
  assert.match(gui.answer, /Red/i)
  assert.match(gui.text_answer, /VERIFIED/)
  assert.match(gui.answer, /VERIFIED/)
}
console.log('Bonsai CUDA optimization evidence: all checks passed')
