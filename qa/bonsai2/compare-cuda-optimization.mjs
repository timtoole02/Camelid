import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
const root = process.argv[2] || 'qa/bonsai2/cuda-optimized'
const report = { date: new Date().toISOString(), metric: 'median warmed uncached request, 32 output tokens', models: [] }
for (const size of [2, 4, 9, 27]) {
  const read = async suffix => JSON.parse(await readFile(`${root}/${size}b-${suffix}-benchmark.json`, 'utf8'))
  const before = await read('baseline'), after = await read('native')
  assert.equal(before.probe_sha256, after.probe_sha256)
  const model = { size, cases: [] }
  for (const row of after.cases) {
    const old = before.cases.find(item => item.name === row.name)
    assert.deepEqual(old.baseline.ids, row.baseline.ids)
    assert.deepEqual(old.speculative.ids, row.speculative.ids)
    model.cases.push({ name: row.name,
      before_plain: old.baseline.total_tokens_per_second,
      after_plain: row.baseline.total_tokens_per_second,
      plain_speedup: row.baseline.total_tokens_per_second / old.baseline.total_tokens_per_second,
      before_draft: old.speculative.total_tokens_per_second,
      after_draft: row.speculative.total_tokens_per_second,
      draft_speedup: row.speculative.total_tokens_per_second / old.speculative.total_tokens_per_second,
      drafter_over_plain: row.speculative.total_tokens_per_second / row.baseline.total_tokens_per_second,
      proposed: row.speculative.speculative.proposed, accepted: row.speculative.speculative.accepted,
      parity: true })
  }
  report.models.push(model)
}
await writeFile(`${root}/optimization.json`, JSON.stringify(report, null, 2) + '\n')
const comparison = JSON.parse(await readFile(`${root}/comparison.json`, 'utf8'))
const lines = [
  '# CUDA optimization measurements', '',
  'Median request tokens/second across three warmed repetitions, including uncached prefill. Same executable for the before/after control; 32 generated tokens, 2048 context slots, 8 threads, and a three-token Bonsai 2 2B drafter. See [README](README.md) for hardware and timing limitations.', '',
  '| Model | Prompt | Before ordinary | After ordinary | Ordinary gain | After draft | Draft / ordinary | Port ordinary | Port draft |',
  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for (const model of report.models) for (const row of model.cases) {
  const port = comparison.models.find(item => item.size === model.size).cases.find(item => item.name === row.name)
  lines.push(`| ${model.size}B | ${row.name} | ${row.before_plain.toFixed(2)} | ${row.after_plain.toFixed(2)} | ${row.plain_speedup.toFixed(2)}x | ${row.after_draft.toFixed(2)} | ${row.drafter_over_plain.toFixed(2)}x | ${port.reference_plain_tokens_per_second.toFixed(2)} | ${port.reference_draft_tokens_per_second.toFixed(2)} |`)
}
lines.push('', 'All ordinary and drafted native token sequences match the original kernel for these cases. Raw repetitions, acceptance counts, placement, and cross-engine parity are retained in the JSON receipts. These bounded checks do not establish identity for arbitrary prompts.', '')
await writeFile(`${root}/RESULTS.md`, lines.join('\n'))
console.log(JSON.stringify(report.models))
