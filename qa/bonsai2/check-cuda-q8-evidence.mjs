import assert from 'node:assert/strict'
import { readFile, readdir } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'
const root = resolve('qa/bonsai2/cuda-q8')
const json = async name => JSON.parse(await readFile(resolve(root, name), 'utf8'))
const digest = bytes => createHash('sha256').update(bytes).digest('hex')
const manifest = await json('manifest.json')
assert.equal(manifest.schema, 1)
assert.equal(manifest.default_projection_policy, 'q8')
assert.match(manifest.source_commit, /^[a-f0-9]{40}$/)
assert.deepEqual((await readdir(root)).filter(name => name !== 'manifest.json').sort(), Object.keys(manifest.artifacts).sort())
for (const name of await readdir(root)) if (name !== 'manifest.json') {
  assert.equal(digest(await readFile(resolve(root, name))), manifest.artifacts[name], `${name} checksum`)
}
const cache = await json('cache-parity.json')
assert.equal(cache.pass, true)
assert.equal(cache.test_binary_sha256, manifest.binaries.tests.sha256)
for (const [path, hash] of Object.entries(cache.source_sha256)) assert.equal(hash, manifest.source_sha256[path], path)
assert.equal(cache.host_tests.length, 2)
assert.equal(cache.tests.length, 14)
for (const row of [...cache.host_tests, ...cache.tests]) {
  assert.equal(row.pass, true)
  assert.match(row.summary, /1 passed; 0 failed/)
  assert.equal(row.environment.CAMELID_BONSAI2_CUDA_PROJECTION, 'q8')
}
assert.equal(cache.tests.filter(row => row.name === 'bonsai_cuda_q8_quantization_projection_and_batch_parity').length, 1)
const quality = await json('quality.json')
assert.equal(quality.models.length, 4)
assert.equal(quality.maximum_mean_nll_increase, 0.1)
const comparison = await json('comparison.json')
assert.equal(comparison.models.length, 4)
const optimization = await json('optimization.json')
assert.equal(optimization.models.length, 4)
const profile = await json('profile.json')
assert.equal(profile.profiles.length, 6)
for (const size of [2, 4, 9]) for (const policy of ['tiled', 'q8']) {
  const rows = profile.profiles.filter(row => row.size === size && row.projection_policy === policy)
  assert.equal(rows.length, 1)
  assert.equal(rows[0].probe_sha256, manifest.binaries.probe.sha256)
  assert.equal(rows[0].source_commit, manifest.source_commit)
  assert.ok(rows[0].kernel_total_ns > 0)
}
for (const size of [2, 4, 9, 27]) {
  const native = await json(`${size}b-native-benchmark.json`)
  const float = await json(`${size}b-float-benchmark.json`)
  const port = await json(`${size}b-reference-benchmark.json`)
  assert.equal(native.target, `Ternary-Bonsai-2-${size}B-PTQ1_0.gguf`)
  assert.equal(native.projection_policy, 'q8')
  assert.equal(float.projection_policy, 'tiled')
  for (const receipt of [native, float]) {
    assert.equal(receipt.probe_sha256, manifest.binaries.probe.sha256)
    for (const [path, hash] of Object.entries(receipt.source_sha256)) assert.equal(hash, manifest.source_sha256[path], path)
    assert.equal(receipt.repetition_process_policy, 'one process per prompt and mode, repeated uncached requests after one warmup')
    for (const row of receipt.cases) for (const run of row.speculative.runs) {
      assert.ok(run.speculative.proposed > 0, 'Drafter must actively propose tokens')
      assert.ok(run.speculative.accepted >= 0 && run.speculative.accepted <= run.speculative.proposed)
    }
  }
  for (const receipt of [native, float, port]) {
    for (const [key, value] of Object.entries({ backend: 'CUDA', repetitions: 3, context: 2048,
      max_tokens: 32, draft_tokens: 3, threads: 8, kv_type: 'f16', warmup_tokens: 8, warmups_per_case: 1 })) assert.equal(receipt[key], value)
    assert.equal(receipt.target_sha256, native.target_sha256)
    assert.equal(receipt.draft_sha256, native.draft_sha256)
    assert.equal(receipt.cases.length, 3)
    for (const row of receipt.cases) for (const mode of ['baseline', 'speculative']) {
      assert.equal(row[mode].runs.length, 3)
      for (const run of row[mode].runs) {
        assert.equal(run.ids.length, 32)
        assert.deepEqual(run.ids, row[mode].ids)
        assert.ok(run.seconds > 0)
        assert.ok(Math.abs(run.total_tokens_per_second - 32 / run.seconds) < 1e-6)
      }
      const times = row[mode].runs.map(run => run.seconds).sort((a, b) => a - b)
      assert.equal(row[mode].seconds, times[1])
    }
  }
  for (const row of native.cases) {
    assert.equal(row.parity, true)
    assert.deepEqual(row.baseline.ids, row.speculative.ids)
    const before = float.cases.find(item => item.name === row.name)
    assert.equal(before.parity, true)
    assert.equal(row.prompt_sha256, before.prompt_sha256)
    const reference = port.cases.find(item => item.name === row.name)
    assert.equal(row.prompt_sha256, reference.prompt_sha256)
    const summary = comparison.models.find(item => item.size === size).cases.find(item => item.name === row.name)
    assert.equal(summary.native_plain_tokens_per_second, row.baseline.total_tokens_per_second)
    assert.equal(summary.native_draft_tokens_per_second, row.speculative.total_tokens_per_second)
    assert.equal(summary.reference_plain_tokens_per_second, reference.baseline.total_tokens_per_second)
    assert.equal(summary.reference_draft_tokens_per_second, reference.speculative.total_tokens_per_second)
    assert.equal(summary.native_reference_parity, JSON.stringify(row.baseline.ids) === JSON.stringify(reference.baseline.ids))
    const optimized = optimization.models.find(item => item.size === size).cases.find(item => item.name === row.name)
    assert.equal(optimized.float_plain, before.baseline.total_tokens_per_second)
    assert.equal(optimized.q8_plain, row.baseline.total_tokens_per_second)
    assert.equal(optimized.float_draft, before.speculative.total_tokens_per_second)
    assert.equal(optimized.q8_draft, row.speculative.total_tokens_per_second)
    assert.equal(optimized.plain_speedup, optimized.q8_plain / optimized.float_plain)
    assert.equal(optimized.drafter_speedup, optimized.q8_draft / optimized.q8_plain)
    assert.equal(optimized.q8_float_ids_match, JSON.stringify(before.baseline.ids) === JSON.stringify(row.baseline.ids))
    assert.equal(optimized.q8_port_ids_match, summary.native_reference_parity)
  }
  for (const name of ['bonsai2_cuda_batch_cache_and_rollback_parity', 'bonsai2_cuda_draft_greedy_sampled_stop_and_cancel_parity'])
    assert.equal(cache.tests.filter(row => row.name === name && row.model_sha256 === native.target_sha256 && row.context === 'default').length, 1)
  const scores = {}
  for (const policy of ['tiled', 'q8']) {
    const receipt = await json(`${size}b-${policy}-quality.json`)
    assert.equal(receipt.test_binary_sha256, manifest.binaries.tests.sha256)
    assert.equal(receipt.model_sha256, native.target_sha256)
    assert.equal(receipt.projection_policy, policy)
    scores[policy] = receipt.cases.flatMap(row => row.scores)
    assert.equal(scores[policy].length, 96)
    for (const score of scores[policy]) assert.ok(Number.isFinite(score.log_probability) && score.log_probability <= 0)
  }
  assert.deepEqual(scores.tiled.map(row => row.token), scores.q8.map(row => row.token))
  const mean = rows => -rows.reduce((sum, row) => sum + row.log_probability, 0) / rows.length
  const delta = mean(scores.q8) - mean(scores.tiled)
  const qualityRow = quality.models.find(row => row.size === size)
  assert.equal(qualityRow.tokens, 96)
  assert.equal(qualityRow.float_mean_nll, mean(scores.tiled))
  assert.equal(qualityRow.q8_mean_nll, mean(scores.q8))
  assert.equal(qualityRow.mean_nll_increase, delta)
  assert.equal(qualityRow.top1_agreement, scores.tiled.filter((row, i) => row.argmax === scores.q8[i].argmax).length / 96)
  assert.equal(qualityRow.maximum_log_probability_change, Math.max(...scores.tiled.map((row, i) => Math.abs(row.log_probability - scores.q8[i].log_probability))))
  assert.ok(delta <= quality.maximum_mean_nll_increase)
  const gui = await json(`${size}b-gui-supported.json`)
  assert.equal(gui.model, native.target)
  assert.equal(gui.selected_backend, 'bonsai2_cuda_runtime')
  assert.equal(gui.support_level, 'supported_windows_runtime')
  assert.equal(gui.backend_build, manifest.binaries.server.receipt_build_id)
  assert.equal(gui.image_limits, 'GUI defaults')
  assert.equal(gui.output_rate_source, 'client_request')
  for (const key of ['generation_ready', 'vision_ready', 'image_input_present']) assert.equal(gui[key], true)
  assert.match(gui.text_answer, /Paris/)
  assert.match(gui.answer.split(/\r?\n/, 1)[0], /^\s*red[.!]?\s*$/i)
  assert.match(gui.text_answer, /tok\/s request/)
  for (const answer of [gui.text_answer, gui.answer]) {
    assert.doesNotMatch(answer, /unverified/i)
    assert.match(answer, /\bverified\b/i)
  }
  const api = await json(`${size}b-api-qualified.json`)
  assert.equal(api.model, native.target)
  assert.equal(api.backend_build, manifest.binaries.server.receipt_build_id)
  assert.equal(api.selected_backend, 'bonsai2_cuda_runtime')
  for (const name of ['text', 'tool_call', 'tool_result', 'context_recall', 'sampled_text', 'vision_spatial'])
    assert.equal(api.cases.find(row => row.name === name)?.pass, true, `${size}B ${name}`)
}
console.log('Bonsai Q8 CUDA evidence: all checks passed')
