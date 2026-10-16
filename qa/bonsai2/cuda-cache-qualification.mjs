// TEST_BINARY MODEL_DIRECTORY OUTPUT.json [LOG_DIRECTORY=output/bonsai2-reference/cuda-cache]
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'

const [binary, models, output, logs = 'output/bonsai2-reference/cuda-cache'] = process.argv.slice(2)
assert.ok(binary && models && output, 'TEST_BINARY MODEL_DIRECTORY OUTPUT required')
await mkdir(logs, { recursive: true })
const digest = bytes => createHash('sha256').update(bytes).digest('hex')
async function fileDigest(path) {
  const hash = createHash('sha256')
  for await (const bytes of createReadStream(path)) hash.update(bytes)
  return hash.digest('hex')
}
const source = {}
for (const path of ['src/runnable/model.rs', 'src/runnable/speculative.rs', 'src/runnable/hadamard.rs', 'src/wire_mmap.rs', 'src/cuda_resident.rs',
  'src/cuda_resident/bonsai_spec.rs', 'src/cuda_resident/bonsai_ptq.rs', 'src/cuda_resident/bonsai_ptq.cu']) {
  source[path] = digest(await readFile(path))
}
const report = { schema: 1, date: new Date().toISOString(), platform: process.platform,
  test_binary_sha256: await fileDigest(binary), source_sha256: source, log_directory: logs, pass: false, host_tests: [], tests: [] }
const modelPath = size => resolve(models, `Ternary-Bonsai-2-${size}B-PTQ1_0.gguf`)
async function run(name, size = null, context = 'default', overrides = {}, hostOnly = false) {
  const configuration = { RAYON_NUM_THREADS: '8', CAMELID_QWEN35_CUDA: hostOnly ? '0' : '1', CAMELID_QWEN35_METAL: '0',
    CAMELID_QWEN35_CUDA_MAXPOS: '2048', ...overrides }
  const env = { ...process.env, ...configuration }
  delete env.CAMELID_BONSAI_TEST_CONTEXT
  if (context !== 'default') env.CAMELID_BONSAI_TEST_CONTEXT = context
  if (size) env.CAMELID_BONSAI_TARGET = modelPath(size)
  env.CAMELID_SPEC_DRAFT_MODEL = modelPath(2)
  const start = performance.now()
  const child = spawn(resolve(binary), [name, ...(hostOnly ? [] : ['--ignored']), '--nocapture', '--test-threads=1'], { env, windowsHide: true })
  let log = ''
  child.stdout.on('data', bytes => { log += bytes })
  child.stderr.on('data', bytes => { log += bytes })
  const code = await new Promise((done, reject) => { child.once('error', reject); child.once('exit', done) })
  const logName = hostOnly ? `host-${name}.log` : `${size || 'fixture'}-${context}-${name}.log`
  await writeFile(resolve(logs, logName), log)
  const row = { name, model: size ? basename(modelPath(size)) : null,
    model_sha256: size ? await fileDigest(modelPath(size)) : null, context, environment: configuration,
    pass: code === 0 && /test result: ok\. 1 passed; 0 failed/.test(log),
    seconds: (performance.now() - start) / 1000, log_sha256: digest(log), log: logName,
    summary: log.split(/\r?\n/).filter(line => line.startsWith('test result:')).at(-1) || `exit ${code}` }
  const rows = hostOnly ? report.host_tests : report.tests
  rows.push(row)
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`)
  console.log(JSON.stringify(row))
  assert.ok(row.pass, `CUDA qualification failed: ${logName}\n${log.slice(-4000)}`)
}
await run('bonsai_ptq_mapped_bytes_preserve_bounds_and_shared_lifetime', null, 'default', {}, true)
await run('bonsai_upload_borrows_unchanged_packed_bytes', null, 'default', {}, true)
await run('bonsai_cuda_transform_projection_and_embedding')
await run('bonsai_cuda_bf16_projection_matches_warp_reference')
for (const size of [2, 4, 9, 27]) await run('bonsai2_cuda_batch_cache_and_rollback_parity', size)
for (const context of ['group-boundary', 'splitk-boundary']) {
  await run('bonsai2_cuda_batch_cache_and_rollback_parity', 2, context, { CAMELID_CUDA_GRAPHS: '1' })
}
for (const size of [2, 4, 9, 27]) await run('bonsai2_cuda_draft_greedy_sampled_stop_and_cancel_parity', size)
report.pass = true
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`)
