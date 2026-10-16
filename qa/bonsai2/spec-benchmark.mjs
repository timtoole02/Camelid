// node qa/bonsai2/spec-benchmark.mjs PROBE TARGET DRAFT OUTPUT [TOKENS=48]
// Each run is a fresh process; identical greedy prompts must produce identical IDs.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const [probe, target, draft, output, tokens = '48'] = process.argv.slice(2)
assert.ok(probe && target && draft && output, 'PROBE TARGET DRAFT OUTPUT required')
const cases = ['count', 'code', 'explain']
async function run(prompt, speculative) {
  const env = { ...process.env, RAYON_NUM_THREADS: '8', CAMELID_QWEN35_CUDA: '0', CAMELID_QWEN35_METAL: '0' }
  delete env.CAMELID_SPEC_DECODE
  delete env.CAMELID_SPEC_DRAFT_MODEL
  if (speculative) Object.assign(env, {
    CAMELID_SPEC_DECODE: 'draft', CAMELID_SPEC_DRAFT_MODEL: resolve(draft), CAMELID_SPEC_DRAFT_TOKENS: '3',
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
const report = { target: resolve(target), draft: resolve(draft), threads: 8, draft_tokens: 3, cases: [] }
for (const name of cases) {
  const path = resolve(`qa/bonsai2/${name}.chatml`)
  await readFile(path) // fail before spawning if a prompt was omitted
  const baseline = await run(path, false)
  const speculative = await run(path, true)
  const parity = JSON.stringify(baseline.ids) === JSON.stringify(speculative.ids)
  const row = { name, parity, baseline, speculative,
    decode_speedup: speculative.decode_tokens_per_second / baseline.decode_tokens_per_second,
    total_speedup: baseline.seconds / speculative.seconds,
  }
  report.cases.push(row)
  await writeFile(output, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({ name, parity, decode_speedup: row.decode_speedup, total_speedup: row.total_speedup }))
  assert.ok(parity, `${name}: draft output differs from target`)
}
