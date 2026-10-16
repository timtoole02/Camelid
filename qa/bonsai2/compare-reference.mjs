// PROBE MODEL REFERENCE.json OUTPUT.json [text,count,code,explain]
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'

const [probe, model, referencePath, output, selected = 'text,count,code,explain'] = process.argv.slice(2)
assert.ok(probe && model && referencePath && output)
const reference = JSON.parse(await readFile(referencePath, 'utf8'))
assert.equal(reference.model, basename(model))
const report = {
  model: basename(model), date: new Date().toISOString(), platform: process.platform,
  reference: reference.reference,
  probe_sha256: createHash('sha256').update(await readFile(probe)).digest('hex'), cases: [],
}
for (const name of selected.split(',')) {
  const expected = reference.cases.find(row => row.name === name)
  assert.ok(expected, `missing reference case ${name}`)
  const prompt = resolve(`qa/bonsai2/${name === 'text' ? 'text-prompt.txt' : `${name}.chatml`}`)
  const env = { ...process.env, RAYON_NUM_THREADS: '8', CAMELID_QWEN35_CUDA: '0', CAMELID_QWEN35_METAL: '0' }
  delete env.CAMELID_SPEC_DECODE
  delete env.CAMELID_SPEC_DRAFT_MODEL
  const child = spawn(resolve(probe), [resolve(model), prompt, '32'], { env, windowsHide: true })
  let stdout = '', stderr = ''
  child.stdout.on('data', data => { stdout += data })
  child.stderr.on('data', data => { stderr += data })
  const code = await new Promise((done, reject) => { child.once('error', reject); child.once('exit', done) })
  assert.equal(code, 0, stderr)
  const actual = JSON.parse(stdout.trim().split('\n').at(-1))
  const pass = JSON.stringify(actual.ids) === JSON.stringify(expected.ids)
  report.cases.push({ name, pass, actual, expected: { ids: expected.ids, text: expected.text } })
  report.pass = report.cases.every(row => row.pass)
  await writeFile(output, JSON.stringify(report, null, 2))
  console.log(JSON.stringify({ name, pass, text: actual.text, seconds: actual.seconds }))
}
assert.ok(report.pass, 'reference token disagreement; inspect the saved comparisons')
