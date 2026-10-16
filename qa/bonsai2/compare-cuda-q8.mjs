import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
const root = process.argv[2] || 'qa/bonsai2/cuda-q8'
const comparison = JSON.parse(await readFile(`${root}/comparison.json`, 'utf8'))
const report = { schema: 1, date: new Date().toISOString(), models: [] }
const lines = ['# Transposed Q8 CUDA measurements', '',
  'Request tokens/second, median of three warmed uncached 32-token requests. Each prompt/mode reuses one process after an eight-token warmup. Both Camelid policies use the same executable. Loading, NVRTC compilation and warmup are excluded; prefill and generation are included. See [README](README.md) for the hardware, entry-point differences, and 27B memory strategies.', '',
  '| Model | Prompt | Float ordinary | Q8 ordinary | Q8 / float | Float draft | Q8 draft | Q8 draft / ordinary | Port ordinary | Port draft |',
  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for (const size of [2, 4, 9, 27]) {
  const read = suffix => readFile(`${root}/${size}b-${suffix}-benchmark.json`, 'utf8').then(JSON.parse)
  const before = await read('float'), after = await read('native')
  assert.equal(before.probe_sha256, after.probe_sha256)
  assert.equal(before.repetition_process_policy, after.repetition_process_policy)
  const model = { size, cases: [] }
  for (const row of after.cases) {
    const old = before.cases.find(item => item.name === row.name)
    const port = comparison.models.find(item => item.size === size).cases.find(item => item.name === row.name)
    assert.equal(row.parity, true)
    assert.equal(old.parity, true)
    const result = { name: row.name,
      float_plain: old.baseline.total_tokens_per_second, q8_plain: row.baseline.total_tokens_per_second,
      plain_speedup: row.baseline.total_tokens_per_second / old.baseline.total_tokens_per_second,
      float_draft: old.speculative.total_tokens_per_second, q8_draft: row.speculative.total_tokens_per_second,
      drafter_speedup: row.speculative.total_tokens_per_second / row.baseline.total_tokens_per_second,
      q8_float_ids_match: JSON.stringify(old.baseline.ids) === JSON.stringify(row.baseline.ids),
      q8_port_ids_match: port.native_reference_parity,
      port_plain: port.reference_plain_tokens_per_second, port_draft: port.reference_draft_tokens_per_second }
    model.cases.push(result)
    lines.push(`| ${size}B | ${row.name} | ${result.float_plain.toFixed(2)} | ${result.q8_plain.toFixed(2)} | ${result.plain_speedup.toFixed(2)}x | ${result.float_draft.toFixed(2)} | ${result.q8_draft.toFixed(2)} | ${result.drafter_speedup.toFixed(2)}x | ${result.port_plain.toFixed(2)} | ${result.port_draft.toFixed(2)} |`)
  }
  report.models.push(model)
}
lines.push('', 'Q8 changes activation arithmetic. Ordinary/draft equality is required within each Camelid policy; cross-policy and cross-engine differences are recorded separately in `optimization.json` and `comparison.json`.', '')
const quality = JSON.parse(await readFile(`${root}/quality.json`, 'utf8'))
lines.push('## Fixed-continuation numerical check', '',
  'Each model scores the same 96 continuation tokens under both arithmetic policies. This is a bounded regression check, not a general task-quality or corpus-perplexity evaluation.', '',
  '| Model | Float mean NLL | Q8 mean NLL | Increase (nats/token) | Top-1 agreement |',
  '| --- | ---: | ---: | ---: | ---: |')
for (const row of quality.models) lines.push(`| ${row.size}B | ${row.float_mean_nll.toFixed(6)} | ${row.q8_mean_nll.toFixed(6)} | ${row.mean_nll_increase.toFixed(6)} | ${(row.top1_agreement * 100).toFixed(2)}% |`)
await writeFile(`${root}/optimization.json`, JSON.stringify(report, null, 2) + '\n')
await writeFile(`${root}/RESULTS.md`, lines.join('\n') + '\n')
console.log(JSON.stringify(report.models))
