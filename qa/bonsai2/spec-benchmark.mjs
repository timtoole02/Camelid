// node qa/bonsai2/spec-benchmark.mjs PROBE TARGET DRAFT OUTPUT [TOKENS=48]
// Each run is a fresh process; identical greedy prompts must produce identical IDs.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { createReadStream } from 'node:fs'
import { cpus, totalmem } from 'node:os'
import { basename, resolve } from 'node:path'
import { createHash } from 'node:crypto'

const [probe, target, draft, output, tokens = '48'] = process.argv.slice(2)
assert.ok(probe && target && draft && output, 'PROBE TARGET DRAFT OUTPUT required')
const cases = (process.env.CAMELID_BENCH_CASES || 'count,code,explain').split(',')
const window = process.env.CAMELID_SPEC_DRAFT_TOKENS || '3'
const referencePath = process.env.CAMELID_BENCH_REFERENCE
const reference = referencePath ? JSON.parse(await readFile(referencePath, 'utf8')) : null
if (reference) assert.equal(reference.model, basename(target))
async function run(prompt, speculative) {
  const env = { ...process.env, RAYON_NUM_THREADS: '8', CAMELID_QWEN35_CUDA: '0', CAMELID_QWEN35_METAL: '0' }
  delete env.CAMELID_SPEC_DECODE
  delete env.CAMELID_SPEC_DRAFT_MODEL
  if (speculative) Object.assign(env, {
    CAMELID_SPEC_DECODE: 'draft', CAMELID_SPEC_DRAFT_MODEL: resolve(draft), CAMELID_SPEC_DRAFT_TOKENS: window,
  })
  const child = spawn(resolve(probe), [resolve(target), prompt, tokens], { env, windowsHide: true })
  let stdout = '', stderr = ''
  child.stdout.on('data', data => { stdout += data })
  child.stderr.on('data', data => { stderr += data })
  const code = await new Promise((done, reject) => { child.once('error', reject); child.once('exit', done) })
  assert.equal(code, 0, stderr)
  const result = JSON.parse(stdout.trim().split('\n').at(-1))
  const match = stderr.match(/accepted (\d+)\/(\d+) drafts; (\d+) tokens in ([\d.]+)s decode/)
  const times = result.token_seconds
  return { ...result,
    first_token_seconds: times[0] ?? null,
    decode_tokens_per_second: times.length > 1 ? (times.length - 1) / (times.at(-1) - times[0]) : null,
    total_tokens_per_second: result.ids.length / result.seconds,
    speculative: match ? { accepted: +match[1], proposed: +match[2], decode_seconds: +match[4] } : null,
  }
}
const digest = bytes => createHash('sha256').update(bytes).digest('hex')
async function fileDigest(path) {
  const hash = createHash('sha256')
  for await (const bytes of createReadStream(path)) hash.update(bytes)
  return hash.digest('hex')
}
const source = {}
for (const path of ['src/runnable/model.rs', 'src/runnable/speculative.rs', 'src/runnable/batched_dot.rs']) {
  source[path] = digest(await readFile(path))
}
const report = { target: basename(target), draft: basename(draft), date: new Date().toISOString(),
  platform: process.platform, architecture: process.arch, cpu: cpus()[0]?.model, physical_memory_bytes: totalmem(),
  build_notes: process.env.CAMELID_BENCH_BUILD || null,
  target_sha256: await fileDigest(target), draft_sha256: await fileDigest(draft),
  probe_sha256: digest(await readFile(probe)), source_sha256: source,
  threads: 8, draft_tokens: Number(window), max_tokens: Number(tokens), cases: [] }
for (const name of cases) {
  const path = resolve(`qa/bonsai2/${name}.chatml`)
  await readFile(path) // fail before spawning if a prompt was omitted
  const baseline = await run(path, false)
  const row = { name, baseline }
  if (reference) {
    const expected = reference.cases.find(row => row.name === name)
    assert.ok(expected, `missing reference case ${name}`)
    const length = Math.min(baseline.ids.length, expected.ids.length)
    row.reference = { compared_tokens: length,
      prefix_matches: length > 0 && JSON.stringify(baseline.ids.slice(0, length)) === JSON.stringify(expected.ids.slice(0, length)),
      expected_ids: expected.ids, expected_text: expected.text }
  }
  report.cases.push(row)
  await writeFile(output, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({ name, phase: 'baseline', seconds: baseline.seconds, tokens: baseline.ids.length }))
  const speculative = await run(path, true)
  const parity = JSON.stringify(baseline.ids) === JSON.stringify(speculative.ids)
  Object.assign(row, { parity, speculative,
    decode_speedup: speculative.decode_tokens_per_second / baseline.decode_tokens_per_second,
    total_speedup: baseline.seconds / speculative.seconds,
  })
  await writeFile(output, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({ name, parity, decode_speedup: row.decode_speedup, total_speedup: row.total_speedup }))
  assert.ok(parity, `${name}: draft output differs from target`)
}
