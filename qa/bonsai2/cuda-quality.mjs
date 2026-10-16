// TEST_BINARY MODEL_DIRECTORY OUTPUT_DIRECTORY [FLOAT_CASES_DIRECTORY]
// Score the same 96 continuation tokens under float and Q8 arithmetic.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'
const [binary, models, directory, casesDirectory = 'qa/bonsai2/cuda-optimized'] = process.argv.slice(2)
assert.ok(binary && models && directory, 'TEST_BINARY MODEL_DIRECTORY OUTPUT_DIRECTORY required')
await mkdir(directory, { recursive: true })
async function digest(path) {
  const hash = createHash('sha256')
  for await (const bytes of createReadStream(path)) hash.update(bytes)
  return hash.digest('hex')
}
const testHash = await digest(binary)
const summary = { schema: 1, date: new Date().toISOString(),
  scope: 'Three fixed float-path continuations, 32 tokens each. A bounded numerical regression check, not a general perplexity or task-quality evaluation.',
  maximum_mean_nll_increase: 0.1, models: [] }
for (const size of [2, 4, 9, 27]) {
  const model = resolve(models, `Ternary-Bonsai-2-${size}B-PTQ1_0.gguf`)
  const casesPath = resolve(casesDirectory, `${size}b-native-benchmark.json`)
  const cases = JSON.parse(await readFile(casesPath, 'utf8'))
  const modelHash = await digest(model)
  assert.equal(cases.target_sha256, modelHash)
  const policies = {}
  for (const policy of ['tiled', 'q8']) {
    const output = resolve(directory, `${size}b-${policy}-quality.json`)
    const env = { ...process.env, RAYON_NUM_THREADS: '8', CAMELID_QWEN35_CUDA: '1', CAMELID_QWEN35_METAL: '0',
      CAMELID_BONSAI2_CUDA_PROJECTION: policy, CAMELID_BONSAI_TARGET: model,
      CAMELID_BONSAI_QUALITY_CASES: casesPath, CAMELID_BONSAI_QUALITY_OUTPUT: output }
    const child = spawn(resolve(binary), ['bonsai2_cuda_teacher_forced_quality', '--ignored', '--nocapture', '--test-threads=1'], { env, windowsHide: true })
    let log = ''
    child.stdout.on('data', bytes => { log += bytes })
    child.stderr.on('data', bytes => { log += bytes })
    const code = await new Promise((done, reject) => { child.once('error', reject); child.once('close', done) })
    await writeFile(resolve(directory, `${size}b-${policy}-quality.log`), log)
    assert.equal(code, 0, log.slice(-4000))
    assert.match(log, /test result: ok\. 1 passed; 0 failed/)
    const receipt = JSON.parse(await readFile(output, 'utf8'))
    assert.equal(receipt.projection_policy, policy)
    Object.assign(receipt, { model: basename(model), model_sha256: modelHash, test_binary_sha256: testHash,
      cases_sha256: await digest(casesPath), log_sha256: createHash('sha256').update(log).digest('hex') })
    await writeFile(output, JSON.stringify(receipt, null, 2) + '\n')
    policies[policy] = receipt.cases.flatMap(row => row.scores)
    console.log(`${size}B ${policy}: scored ${policies[policy].length} fixed continuation tokens`)
  }
  const before = policies.tiled, after = policies.q8
  assert.equal(before.length, 96)
  assert.equal(after.length, 96)
  assert.deepEqual(before.map(row => row.token), after.map(row => row.token))
  const meanNll = rows => -rows.reduce((sum, row) => sum + row.log_probability, 0) / rows.length
  const oldNll = meanNll(before), newNll = meanNll(after)
  const row = { size, tokens: 96, float_mean_nll: oldNll, q8_mean_nll: newNll,
    mean_nll_increase: newNll - oldNll,
    top1_agreement: before.filter((row, i) => row.argmax === after[i].argmax).length / 96,
    maximum_log_probability_change: Math.max(...before.map((row, i) => Math.abs(row.log_probability - after[i].log_probability))) }
  assert.ok(Number.isFinite(oldNll) && Number.isFinite(newNll))
  assert.ok(row.mean_nll_increase <= summary.maximum_mean_nll_increase, `${size}B mean NLL regression`)
  summary.models.push(row)
  await writeFile(resolve(directory, 'quality.json'), JSON.stringify(summary, null, 2) + '\n')
  console.log(JSON.stringify(row))
}
