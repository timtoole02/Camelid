// [EVIDENCE_DIRECTORY=qa/bonsai2/cuda] [OUTPUT=EVIDENCE_DIRECTORY/comparison.json]
// Both engines report complete uncached request wall time after an 8-token warmup.
import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const directory = process.argv[2] || 'qa/bonsai2/cuda'
const output = process.argv[3] || resolve(directory, 'comparison.json')
const report = {
  date: new Date().toISOString(), metric: '32 generated tokens / uncached warmed request seconds',
  notes: [
    'Request time includes full prompt prefill and decoding; model loading and warmup are excluded for both engines.',
    'Native streaming callback timing can be bursty during speculation; complete request wall time is the comparison metric.',
    'The llama.cpp port records its actual layer offload for each mode; a 27B target may use CPU layers on a 6 GiB GPU.',
    'Automatic fitting may assign fewer target layers to the GPU when draft state and the draft model are resident. Speedups therefore describe complete operating configurations on this hardware.',
    'Greedy speculative parity is checked independently within each engine. Differences from the reference are recorded rather than hidden.',
  ], models: [],
}
for (const size of [2, 4, 9, 27]) {
  const read = suffix => readFile(resolve(directory, `${size}b-${suffix}-benchmark.json`), 'utf8').then(JSON.parse)
  const native = await read('native')
  const reference = await read('reference')
  assert.equal(native.backend, 'CUDA')
  assert.equal(reference.backend, 'CUDA')
  for (const field of ['target_sha256', 'draft_sha256', 'threads', 'max_tokens', 'draft_tokens', 'context', 'kv_type', 'warmup_tokens']) {
    assert.equal(native[field], reference[field], `${size}B ${field} mismatch`)
  }
  const model = { size, native: `${size}b-native-benchmark.json`, reference: `${size}b-reference-benchmark.json`,
    native_repetitions: native.repetitions, reference_repetitions: reference.repetitions,
    reference_gpu_layers: Object.fromEntries(['baseline', 'speculative'].map(mode => [mode,
      reference.modes[mode].offload.flatMap(line => {
        const match = line.match(/offloaded (\d+)\/(\d+) layers to GPU/)
        return match ? [{ gpu_layers: Number(match[1]), total_layers: Number(match[2]) }] : []
      }),
    ])), cases: [] }
  for (const name of ['count', 'code', 'explain']) {
    const ours = native.cases.find(row => row.name === name)
    const theirs = reference.cases.find(row => row.name === name)
    assert.ok(ours && theirs, `${size}B missing ${name}`)
    assert.equal(ours.prompt_sha256, theirs.prompt_sha256, `${size}B ${name} prompt differs`)
    assert.ok(ours.parity, `${size}B ${name}: native draft changed target IDs`)
    for (const mode of ['baseline', 'speculative']) {
      assert.equal(ours[mode].ids.length, 32)
      assert.equal(theirs[mode].ids.length, 32)
      assert.match(ours[mode].backend, /cuda/i)
    }
    const nativePlain = ours.baseline.total_tokens_per_second
    const nativeDraft = ours.speculative.total_tokens_per_second
    const referencePlain = theirs.baseline.total_tokens_per_second
    const referenceDraft = theirs.speculative.total_tokens_per_second
    const divergence = ours.baseline.ids.findIndex((id, i) => id !== theirs.baseline.ids[i])
    model.cases.push({ name, native_plain_tokens_per_second: nativePlain, native_draft_tokens_per_second: nativeDraft,
      native_cuda_capacity_plan: { baseline: ours.baseline.cuda_capacity_plan, speculative: ours.speculative.cuda_capacity_plan },
      reference_plain_tokens_per_second: referencePlain, reference_draft_tokens_per_second: referenceDraft,
      native_draft_speedup: nativeDraft / nativePlain, reference_draft_speedup: referenceDraft / referencePlain,
      native_plain_over_reference_plain: nativePlain / referencePlain,
      native_draft_over_reference_draft: nativeDraft / referenceDraft,
      native_draft_over_reference_plain: nativeDraft / referencePlain,
      native_speculative_parity: ours.parity, reference_speculative_parity: theirs.parity,
      native_reference_parity: divergence < 0, first_native_reference_divergence: divergence < 0 ? null : divergence,
    })
  }
  report.models.push(model)
}
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`)
console.log(JSON.stringify(report.models))
