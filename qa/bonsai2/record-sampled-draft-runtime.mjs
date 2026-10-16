// Preserve the actual dedicated-server log and associate its four runs with
// completed API receipts in the operator's explicitly recorded request order.
// This helper reads files only; it never starts a server or model request.
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const [logPath, receiptDirectory = 'qa/bonsai2/cuda', outputPath = 'qa/bonsai2/cuda/sampled-draft-runtime.json', sequenceArgument = '2,4,9,27'] = process.argv.slice(2)
if (!logPath || logPath === '--help') {
  console.log(`Usage: node qa/bonsai2/record-sampled-draft-runtime.mjs LOG_PATH [RECEIPT_DIRECTORY] [OUTPUT_JSON] [MODEL_SEQUENCE]

Paths are relative to the repository root. Defaults:
  RECEIPT_DIRECTORY = qa/bonsai2/cuda
  OUTPUT_JSON = qa/bonsai2/cuda/sampled-draft-runtime.json
  MODEL_SEQUENCE = 2,4,9,27

Supply the actual sequential model order used in one dedicated draft-server
process. The default records the documented 2B, 4B, 9B, 27B qualification order.
The log must be UTF-8 and contain exactly four CUDA drafting summaries, each
after its projector-load event. Model names and timestamps are NOT inferred
from unnamed log lines: the supplied order is an explicit operator record,
checked against model identities and increasing timestamps in the completed
SIZEb-sampled-draft.json receipts. Original log bytes and receipts are unchanged.`)
  process.exit(logPath ? 0 : 1)
}

const sequence = sequenceArgument.split(',').map(Number)
assert.deepEqual([...sequence].sort((a, b) => a - b), [2, 4, 9, 27], 'MODEL_SEQUENCE must contain 2,4,9,27 exactly once')
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const recordedPath = path => relative(root, resolve(root, path)).split(sep).join('/')
const logBytes = await readFile(resolve(root, logPath))
// Preserve a UTF-8 BOM, if present, as part of the recorded text and byte hash.
// Fatal decoding rejects PowerShell UTF-16 output instead of corrupting it.
const logText = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(logBytes)
assert.deepEqual(Buffer.from(logText, 'utf8'), logBytes, 'log must round-trip as UTF-8 without changing any bytes')
const lines = logText.split(/\r?\n/)
const loadEvent = '[qwen3vl] streaming CUDA projector active (bounded VRAM)'
const summaryPattern = /^\[bonsai2-spec-cuda\] accepted (\d+)\/(\d+) drafts; (\d+) tokens in ([\d.]+)s decode$/
const summaries = lines.flatMap((text, index) => {
  const match = text.match(summaryPattern)
  return match ? [{ line: index + 1, text, match }] : []
})
assert.equal(summaries.length, sequence.length, 'expected exactly four draft summaries in the dedicated server log')
assert.equal(lines.filter(line => line === loadEvent).length, sequence.length, 'expected exactly four model/projector load events')

const runs = []
let backendBuild
let previousDate = -Infinity
for (const [index, size] of sequence.entries()) {
  const apiPath = resolve(root, receiptDirectory, `${size}b-sampled-draft.json`)
  const apiBytes = await readFile(apiPath)
  const api = JSON.parse(apiBytes)
  assert.equal(api.model, `Ternary-Bonsai-2-${size}B-PTQ1_0.gguf`)
  assert.equal(api.platform, 'win32')
  assert.equal(api.selected_backend, 'bonsai2_cuda_runtime')
  assert.ok(api.backend_build)
  backendBuild ??= api.backend_build
  assert.equal(api.backend_build, backendBuild, 'all receipts must use the same server build')
  assert.ok(Date.parse(api.date) > previousDate, 'receipt timestamps do not match the recorded model sequence')
  previousDate = Date.parse(api.date)
  assert.equal(api.cases.length, 1, 'draft receipt must contain only the sampled continuation request')
  assert.equal(api.cases[0].name, 'sampled_continuation')
  assert.equal(api.cases[0].pass, true)
  assert.equal(api.cases[0].response.camelid.generated_token_ids.length, 32)
  const { line, text, match } = summaries[index]
  const loadLine = lines.slice(0, line - 1).lastIndexOf(loadEvent) + 1
  assert.ok(loadLine > (summaries[index - 1]?.line || 0), 'each summary must follow a new model/projector load')
  const [, accepted, proposed, outputTokens, decodeSeconds] = match.map(Number)
  assert.ok(accepted <= proposed && proposed > 0)
  assert.equal(outputTokens, 32)
  assert.ok(Number.isFinite(decodeSeconds) && decodeSeconds > 0)
  runs.push({
    ordinal: index + 1,
    model: api.model,
    api_receipt: { path: recordedPath(apiPath), sha256: hash(apiBytes), started_at: api.date },
    preceding_load_event: { line: loadLine, text: lines[loadLine - 1] },
    summary: { line, text, accepted, proposed, output_tokens: outputTokens, decode_seconds_reported: decodeSeconds },
  })
}

const record = {
  schema_version: 1,
  platform: 'win32',
  backend_build: backendBuild,
  selected_backend: 'bonsai2_cuda_runtime',
  association_basis: `Operator-recorded sequential ${sequence.map(size => `${size}B`).join(', ')} qualification requests in one dedicated server process. The API receipt timestamps increase in the same order. The server log records preceding projector-load events but contains no model names or timestamps; model association is not independently encoded in those log lines.`,
  source_log: { path: recordedPath(logPath), sha256: hash(logBytes), size_bytes: logBytes.length, encoding: 'utf-8', text: logText },
  runs,
}
const destination = resolve(root, outputPath)
await mkdir(dirname(destination), { recursive: true })
// Repository JSON uses LF on every platform. The embedded log text still keeps
// its original line endings, so its byte hash remains independent of formatting.
await writeFile(destination, `${JSON.stringify(record, null, 2)}\n`, 'utf8')
console.log(`Wrote ${recordedPath(destination)} with ${logBytes.length} unchanged log bytes and four API receipt associations.`)
