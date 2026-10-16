// CUDA_SERVER TARGET DRAFT OUTPUT.json [TOKENS=32]
// Pinned Prism server; each mode owns one process, warms each exact prompt once,
// then measures an uncached request. Probability captures are separate from timings.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { createServer } from 'node:net'
import { cpus, totalmem } from 'node:os'
import { basename, dirname, resolve } from 'node:path'

const [server, target, draft, output, tokenArgument = '32'] = process.argv.slice(2)
assert.ok(server && target && draft && output, 'CUDA_SERVER TARGET DRAFT OUTPUT.json required')
const maxTokens = Number(tokenArgument)
const port = Number(process.env.CAMELID_REFERENCE_PORT || 18091)
const root = `http://127.0.0.1:${port}`
const cases = (process.env.CAMELID_BENCH_CASES || 'count,code,explain').split(',')
const modes = (process.env.CAMELID_REFERENCE_MODES || 'baseline,speculative').split(',')
const draftTokens = Number(process.env.CAMELID_SPEC_DRAFT_TOKENS || 3)
const threads = 8
const context = Number(process.env.CAMELID_REFERENCE_CONTEXT || 2048)
const kvType = process.env.CAMELID_REFERENCE_KV_TYPE || 'f16'
const gpuLayers = process.env.CAMELID_REFERENCE_GPU_LAYERS || 'auto'
const draftGpuLayers = process.env.CAMELID_REFERENCE_DRAFT_GPU_LAYERS || 'all'
const fitMargin = process.env.CAMELID_REFERENCE_FIT_MARGIN || '768'
const repetitions = Number(process.env.CAMELID_BENCH_REPETITIONS || 1)
// Full logprob capture can disable backend sampling and should be requested only
// when investigating an output divergence, never added to a timing request.
const scores = process.env.CAMELID_REFERENCE_SCORES === '1'
const logDirectory = process.env.CAMELID_REFERENCE_LOG_DIR || dirname(output)
await mkdir(logDirectory, { recursive: true })
const env = { ...process.env }
// Reuse the installed CUDA 12 runtime without copying hundreds of MB of DLLs.
if (process.env.CAMELID_REFERENCE_CUDA_BIN) {
  const pathKey = Object.keys(env).find(key => key.toLowerCase() === 'path') || 'PATH'
  env[pathKey] = `${process.env.CAMELID_REFERENCE_CUDA_BIN};${env[pathKey] || ''}`
}
const sleep = ms => new Promise(done => setTimeout(done, ms))
async function hash(path) {
  const digest = createHash('sha256')
  for await (const bytes of createReadStream(path)) digest.update(bytes)
  return digest.digest('hex')
}
function command(file, args) {
  const result = spawnSync(file, args, { env, encoding: 'utf8', windowsHide: true })
  assert.equal(result.error, undefined, String(result.error))
  assert.equal(result.status, 0, result.stderr)
  return `${result.stdout}${result.stderr}`.trim()
}
async function ensurePortFree() {
  const socket = createServer()
  await new Promise((done, reject) => {
    socket.once('error', reject)
    socket.listen(port, '127.0.0.1', done)
  })
  await new Promise(done => socket.close(done))
}
const version = command(resolve(server), ['--version'])
const devices = command(resolve(server), ['--list-devices'])
assert.match(devices, /CUDA\d/, `The reference binary exposes no CUDA device: ${devices}`)
let report = {
  schema: 1, model: basename(target), draft: basename(draft), date: new Date().toISOString(),
  reference: 'PrismML-Eng/llama.cpp prism-b10777-e3de715', version, devices,
  release_package: 'llama-prism-b10777-e3de715-bin-win-cuda-12.4-x64.zip',
  release_package_sha256: 'fa5874a1412e251a6efb13f289e97bacbbb4e7fc6714c1836b4fc6c1dc863462',
  runtime_note: process.env.CAMELID_REFERENCE_RUNTIME_NOTE || null,
  backend: 'CUDA', platform: process.platform, architecture: process.arch,
  cpu: cpus()[0]?.model, physical_memory_bytes: totalmem(),
  gpu: command('nvidia-smi', ['--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
  target_sha256: await hash(target), draft_sha256: await hash(draft),
  binary_sha256: {}, threads, context, kv_type: kvType, max_tokens: maxTokens,
  draft_tokens: draftTokens, warmups_per_case: 1, warmup_tokens: 8, repetitions,
  timing_definition: 'Request wall time includes full uncached prefill and generation after a warmup with both models resident. Process startup and model loading are reported separately. Engine prompt/decode timings are retained separately. Probability capture is untimed.',
  source_urls: [
    'https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10777-e3de715',
    'https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10777-e3de715/tools/server/README.md',
  ],
  modes: {}, cases: [],
}
for (const name of ['llama-server.exe', 'llama-server-impl.dll', 'llama.dll', 'ggml-cuda.dll']) {
  report.binary_sha256[name] = await hash(resolve(dirname(server), name))
}
if (process.env.CAMELID_REFERENCE_RESUME === '1') {
  const prior = JSON.parse(await readFile(output, 'utf8'))
  for (const field of ['model', 'draft', 'target_sha256', 'draft_sha256', 'threads', 'context', 'kv_type', 'max_tokens', 'draft_tokens', 'repetitions']) {
    assert.equal(report[field], prior[field], `Cannot resume a different ${field}`)
  }
  assert.deepEqual(report.binary_sha256, prior.binary_sha256, 'Cannot resume with different binaries')
  report = prior
  report.resumed_at = new Date().toISOString()
}
let activeChild
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => {
  activeChild?.kill()
  process.exitCode = 130
})
const save = () => writeFile(output, `${JSON.stringify(report, null, 2)}\n`)

async function request(prompt, probabilities = false, tokenLimit = maxTokens) {
  const start = performance.now()
  const response = await fetch(`${root}/completion`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ prompt, n_predict: tokenLimit, temperature: 0, top_k: 1,
      repeat_penalty: 1, seed: 42, return_tokens: true, cache_prompt: false,
      ...(probabilities ? { n_probs: 20, post_sampling_probs: false } : {}),
    }), signal: AbortSignal.timeout(30 * 60 * 1000),
  })
  const result = await response.json()
  const seconds = (performance.now() - start) / 1000
  assert.ok(response.ok, JSON.stringify(result))
  const ids = result.tokens.at(-1) === 248046 ? result.tokens.slice(0, -1) : result.tokens
  return { ids, text: result.content, seconds, total_tokens_per_second: ids.length / seconds,
    prompt_tokens: result.tokens_evaluated, timings: result.timings,
    ...(probabilities ? { completion_probabilities: result.completion_probabilities } : {}),
  }
}

for (const mode of modes) {
  assert.ok(['baseline', 'speculative'].includes(mode), mode)
  const pendingCases = cases.filter(name => !report.cases.find(row => row.name === name)?.[mode]?.runs?.length)
  if (!pendingCases.length) continue
  await ensurePortFree()
  const args = ['-m', resolve(target), '--host', '127.0.0.1', '--port', String(port),
    '-c', String(context), '-t', String(threads), '-tb', String(threads), '-np', '1',
    '-ngl', gpuLayers, '--fit', 'on', '--fit-target', fitMargin, '--fit-ctx', String(context),
    '-ctk', kvType, '-ctv', kvType, '-fa', 'on', '--no-ui', '--perf', '-lv', '4']
  if (mode === 'speculative') args.push('-md', resolve(draft), '--spec-type', 'draft-simple',
    '--spec-draft-n-max', String(draftTokens), '--spec-draft-n-min', '0',
    '--spec-draft-p-min', '0', '-ngld', draftGpuLayers, '-td', String(threads), '-tbd', String(threads),
    '-ctkd', kvType, '-ctvd', kvType)
  let log = ''
  const start = performance.now()
  const child = activeChild = spawn(resolve(server), args, { env, windowsHide: true })
  const completion = new Promise(done => child.once('close', done))
  child.stdout.on('data', data => { log += data })
  child.stderr.on('data', data => { log += data })
  let launchError
  child.on('error', error => { launchError = error })
  const previous = report.modes[mode]
  const { previous_attempts = [], ...previousAttempt } = previous || {}
  const logFile = `${basename(output)}.${mode}${previous ? `.attempt-${previous_attempts.length + 2}` : ''}.log`
  const record = report.modes[mode] = {
    arguments: args.map(value => value === resolve(target) || value === resolve(draft) ? basename(value) : value),
    log_file: logFile,
    ...(previous ? { previous_attempts: [...previous_attempts, previousAttempt] } : {}),
  }
  try {
    let ready = false
    for (let i = 0; i < 300; i++) {
      assert.equal(launchError, undefined, String(launchError))
      assert.equal(child.exitCode, null, log.slice(-16000))
      try { ready = (await fetch(`${root}/health`, { signal: AbortSignal.timeout(1000) })).ok } catch {}
      if (ready) break
      await sleep(1000)
    }
    assert.ok(ready, log.slice(-16000))
    record.startup_seconds = (performance.now() - start) / 1000
    record.offload = log.split(/\r?\n/).filter(line => /offload|CUDA\d.*(buffer|memory)|using device CUDA|n_ctx|flash_attn|draft.*model/i.test(line))
      .map(line => line.replaceAll(resolve(target), basename(target)).replaceAll(resolve(draft), basename(draft)))
    assert.match(log, /offloaded [1-9]\d*\/\d+ layers to GPU/, 'No target GPU layer offload in reference log')
    await save()
    for (const name of pendingCases) {
      const prompt = await readFile(new URL(`./${name}.chatml`, import.meta.url), 'utf8')
      const promptSha = createHash('sha256').update(prompt).digest('hex')
      let row = report.cases.find(row => row.name === name)
      if (!row) { row = { name, prompt_sha256: promptSha }; report.cases.push(row) }
      const warmup = await request(prompt, false, 8)
      const runs = []
      for (let n = 0; n < repetitions; n++) {
        const result = await request(prompt)
        assert.equal(result.ids.length, maxTokens, `${name} ${mode}: expected ${maxTokens} generated tokens`)
        runs.push(result)
      }
      const median = [...runs].sort((a, b) => a.seconds - b.seconds)[Math.floor(runs.length / 2)]
      row[mode] = { ...median, warmup_seconds: warmup.seconds, runs }
      row[mode].repeated_ids_match = runs.every(run => JSON.stringify(run.ids) === JSON.stringify(median.ids))
      if (mode === 'speculative') {
        assert.ok(median.timings?.draft_n > 0, 'Reference speculation did not propose tokens')
        row.parity = row.baseline ? JSON.stringify(row.baseline.ids) === JSON.stringify(median.ids) : null
        if (row.baseline) row.total_speedup = row.baseline.seconds / median.seconds
      }
      await save()
      console.log(JSON.stringify({ model: basename(target), mode, name, seconds: median.seconds,
        tokens_per_second: median.total_tokens_per_second, parity: row.parity, timings: median.timings }))
      if (scores) {
        row[mode].scored_continuation = await request(prompt, true)
        row[mode].scored_ids_match = JSON.stringify(row[mode].scored_continuation.ids) === JSON.stringify(median.ids)
        await save()
      }
    }
  } catch (error) {
    record.error = String(error)
    await save()
    throw error
  } finally {
    child.kill()
    await completion
    activeChild = undefined
    record.log_sha256 = createHash('sha256').update(log).digest('hex')
    await writeFile(resolve(logDirectory, logFile), log)
    await save()
  }
}
report.completion_status = cases.every(name => modes.every(mode =>
  report.cases.find(row => row.name === name)?.[mode]?.runs?.length === repetitions)) ? 'complete' : 'partial'
report.completed_at = new Date().toISOString()
await save()
