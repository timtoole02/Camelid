// Generate the CUDA manifest only after the complete evidence checker passes.
// BUILD_RECORD.json [MODEL_DIRECTORY=output/bonsai2-models]
// [OUTPUT=qa/bonsai2/cuda/manifest.json]
// Use --help for the build-record schema. No inference or compilation is run.
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { createReadStream } from 'node:fs'
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { dirname, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const [recordPath, modelDirectory = 'output/bonsai2-models', destination = 'qa/bonsai2/cuda/manifest.json'] = process.argv.slice(2)
if (recordPath === '--help' || !recordPath) {
  console.log(`Usage: node qa/bonsai2/cuda-manifest.mjs BUILD_RECORD.json [MODEL_DIRECTORY] [OUTPUT]

Paths are relative to the repository root. All receipts must already pass
check-cuda-evidence.mjs, and comparison.json must exist. The generator hashes
the actual language packs, projectors and retained binaries. It does not run
models, rebuild binaries, alter receipts, or use current source as build proof.

Build record (supply the actual revisions, build options and retained paths):
{
  "schema_version": 1,
  "build_configuration": {
    "profile": "release", "features": ["cuda"], "target_cpu": "x86-64-v3",
    "lto": false, "codegen_units": 16, "rustc": "record rustc --version",
    "cuda_runtime": "record the runtime used"
  },
  "binaries": {
    "probe": {"role": "probe", "path": "output/bonsai2-reference/bonsai2-probe-cuda.exe", "source_commit": "COMMIT"},
    "tests": {"role": "tests", "path": "output/bonsai2-reference/bonsai2-tests-cuda.exe", "source_commit": "COMMIT"},
    "qualified_server": {"role": "server", "path": "output/bonsai2-reference/bonsai2-server-cuda-qualified.exe", "source_commit": "FINAL_COMMIT", "receipt_build_id": "EXACT_FINAL_BACKEND_BUILD", "additional_source_files": ["frontend/src/PATH_TO_CHANGED_FILE"]}
  }
}

Each binary may override build_configuration and include build_command,
build_log, built_at, and notes. Server receipt_build_id must exactly match its
API/GUI receipt. Keep historical server binaries when receipts use an older
build by adding another named server entry. One server entry is sufficient
when all API/GUI receipts were regenerated with the same final build.
Source commits and build options are explicit operator records; binary
hashes and receipt associations are independently checked by this generator.
`)
  process.exit(recordPath ? 0 : 1)
}

const slash = path => path.split(sep).join('/')
const relativePath = path => slash(relative(root, resolve(root, path)))
const digest = bytes => createHash('sha256').update(bytes).digest('hex')
const git = (...args) => execFileSync('git', args, { cwd: root, windowsHide: true, maxBuffer: 32 * 1024 * 1024 })
const json = async path => JSON.parse(await readFile(resolve(root, path), 'utf8'))
async function fileIdentity(path) {
  const absolute = resolve(root, path)
  const hash = createHash('sha256')
  for await (const bytes of createReadStream(absolute)) hash.update(bytes)
  return { path: relativePath(absolute), size_bytes: (await stat(absolute)).size, sha256: hash.digest('hex') }
}
async function receipt(name) {
  const path = `qa/bonsai2/cuda/${name}`
  return { data: await json(path), identity: await fileIdentity(path) }
}

// This checks case-level success, all 12 device tests, three timing repetitions,
// sampling parity, CUDA routing, reference placement, and prompt/model hashes.
execFileSync(process.execPath, [resolve(root, 'qa/bonsai2/check-cuda-evidence.mjs')], {
  cwd: root, windowsHide: true, stdio: 'inherit',
})
const record = await json(recordPath)
assert.equal(record.schema_version, 1, 'build record schema_version must be 1')
assert.ok(record.build_configuration && record.binaries, 'build configuration and binaries are required')
const sourceFiles = [
  'Cargo.toml', 'Cargo.lock', '.cargo/config.toml', 'build.rs', 'src/bonsai2.rs', 'src/wire_mmap.rs',
  'src/runnable/model.rs', 'src/runnable/speculative.rs', 'src/runnable/hadamard.rs',
  'src/runnable/batched_dot.rs', 'src/runnable/ptq_dot.rs', 'src/cuda_resident.rs',
  'src/cuda_resident/bonsai_spec.rs', 'src/cuda_resident/bonsai_ptq.rs',
  'src/cuda_resident/bonsai_ptq.cu',
]
const binaries = {}
for (const [name, entry] of Object.entries(record.binaries)) {
  assert.ok(['probe', 'tests', 'server'].includes(entry.role), `${name}: unknown binary role`)
  assert.match(entry.source_commit || '', /^[a-fA-F0-9]{7,40}$/, `${name}: supply an explicit commit hash`)
  const commit = git('rev-parse', '--verify', `${entry.source_commit}^{commit}`).toString().trim()
  const tree = git('rev-parse', `${commit}^{tree}`).toString().trim()
  const selectedSource = {}
  const example = entry.role === 'probe' ? ['examples/bonsai2_probe.rs']
    : entry.role === 'server' ? ['examples/bonsai2_server.rs'] : []
  for (const path of new Set([...sourceFiles, ...example, ...(entry.additional_source_files || [])])) {
    assert.ok(!path.includes('..') && !path.includes(':') && !path.startsWith('/'), `${name}: invalid source path`)
    selectedSource[path] = digest(git('show', `${commit}:${path}`))
  }
  if (entry.role === 'server') {
    assert.ok(entry.receipt_build_id, `${name}: server receipt_build_id is required`)
    const suffix = entry.receipt_build_id.match(/-g([a-f0-9]{7,40})(?:-dirty)?$/)?.[1]
    assert.ok(suffix && commit.startsWith(suffix), `${name}: backend build ID does not identify source_commit`)
  }
  binaries[name] = {
    ...await fileIdentity(entry.path), role: entry.role, source_commit: commit, source_tree: tree,
    source_identity: 'Git blob bytes at the explicitly recorded build commit; independent of the current checkout.',
    selected_source_blobs_sha256: selectedSource,
    build_configuration: { ...record.build_configuration, ...entry.build_configuration },
    ...(entry.receipt_build_id ? { receipt_build_id: entry.receipt_build_id } : {}),
    ...(entry.build_command ? { build_command: entry.build_command } : {}),
    ...(entry.build_log ? { build_log: await fileIdentity(entry.build_log) } : {}),
    ...(entry.built_at ? { built_at: entry.built_at } : {}),
    ...(entry.notes ? { notes: entry.notes } : {}),
  }
}
function onlyRole(role) {
  const entries = Object.entries(binaries).filter(([, binary]) => binary.role === role)
  assert.equal(entries.length, 1, `exactly one ${role} binary is required`)
  return entries[0]
}
const [probeName, probe] = onlyRole('probe')
const [testsName, tests] = onlyRole('tests')
function serverReceipt(value) {
  const matches = Object.entries(binaries).filter(([, binary]) => binary.role === 'server'
    && binary.receipt_build_id === value.data.backend_build)
  assert.equal(matches.length, 1, `record exactly one retained server for receipt build ${value.data.backend_build}`)
  return { ...value.identity, backend_build: value.data.backend_build, binary: matches[0][0],
    binary_association: 'Receipt build ID and explicit retained-build record; historical receipts do not contain a server SHA-256.' }
}

const catalog = await readFile(resolve(root, 'src/bonsai2.rs'), 'utf8')
const artifactPattern = /repo_id: "(prism-ml\/[^\"]+)",\s*filename: "(Ternary-Bonsai-2-(2|4|9|27)B-PTQ1_0\.gguf)",\s*size_bytes: (\d+),[\s\S]*?\n\s*sha256: "([a-f0-9]{64})",[\s\S]*?filename: "([^\"]+-mmproj-Q8_0\.gguf)",\s*size_bytes: (\d+),[\s\S]*?projector_sha256: "([a-f0-9]{64})"/g
const artifacts = [...catalog.matchAll(artifactPattern)]
assert.equal(artifacts.length, 4, 'publisher catalog must identify all four language/projector pairs')
const cache = await receipt('cache-parity.json')
assert.equal(cache.data.test_binary_sha256, tests.sha256, 'retained test binary differs from cache receipt')
const comparison = await receipt('comparison.json')
assert.deepEqual(comparison.data.models.map(model => model.size), [2, 4, 9, 27])
const models = {}
let host, referenceIdentity
for (const [, repository, language, size, languageBytes, languageHash, projector, projectorBytes, projectorHash] of artifacts) {
  const artifact = async (filename, bytes, expected) => {
    const measured = await fileIdentity(resolve(root, modelDirectory, filename))
    assert.equal(measured.size_bytes, Number(bytes), `${filename}: size differs from publisher pin`)
    assert.equal(measured.sha256, expected, `${filename}: hash differs from publisher pin`)
    return { ...measured, filename, publisher: `https://huggingface.co/${repository}/blob/main/${filename}` }
  }
  const api = await receipt(`${size}b-api-qualified.json`)
  const gui = await receipt(`${size}b-gui-supported.json`)
  const sampled = await receipt(`${size}b-sampled-draft.json`)
  const native = await receipt(`${size}b-native-benchmark.json`)
  const reference = await receipt(`${size}b-reference-benchmark.json`)
  for (const value of [api, gui, sampled]) {
    assert.equal(value.data.model, language, `${size}B: server receipt model differs`)
    assert.equal(value.data.selected_backend, 'bonsai2_cuda_runtime', `${size}B: server receipt is not CUDA`)
  }
  assert.equal(native.data.probe_sha256, probe.sha256, `${size}B: retained probe differs from benchmark receipt`)
  assert.equal(native.data.target_sha256, languageHash)
  assert.equal(reference.data.target_sha256, languageHash)
  const observedHost = Object.fromEntries(['platform', 'architecture', 'cpu', 'physical_memory_bytes', 'gpu']
    .map(key => [key, reference.data[key]]))
  if (host) assert.deepEqual(observedHost, host, 'reference receipts came from different hosts')
  else host = observedHost
  for (const key of ['platform', 'architecture', 'cpu', 'physical_memory_bytes']) {
    assert.equal(native.data[key], host[key], `${size}B: native/reference host ${key} differs`)
  }
  const observedReference = Object.fromEntries(['reference', 'version', 'binary_sha256', 'release_package',
    'release_package_sha256', 'runtime_note'].map(key => [key, reference.data[key]]))
  if (referenceIdentity) assert.deepEqual(observedReference, referenceIdentity, 'reference binaries/runtime differ')
  else referenceIdentity = observedReference
  const compared = comparison.data.models.find(model => model.size === Number(size))
  for (const row of native.data.cases) {
    const referenceRow = reference.data.cases.find(value => value.name === row.name)
    const summary = compared?.cases.find(value => value.name === row.name)
    assert.ok(summary, `${size}B ${row.name}: comparison row missing`)
    for (const [field, value] of [
      ['native_plain_tokens_per_second', row.baseline.total_tokens_per_second],
      ['native_draft_tokens_per_second', row.speculative.total_tokens_per_second],
      ['reference_plain_tokens_per_second', referenceRow.baseline.total_tokens_per_second],
      ['reference_draft_tokens_per_second', referenceRow.speculative.total_tokens_per_second],
      ['native_speculative_parity', row.parity], ['reference_speculative_parity', referenceRow.parity],
    ]) assert.equal(summary[field], value, `${size}B ${row.name}: stale comparison ${field}`)
    assert.deepEqual(summary.native_cuda_capacity_plan,
      { baseline: row.baseline.cuda_capacity_plan, speculative: row.speculative.cuda_capacity_plan },
      `${size}B ${row.name}: stale comparison placement`)
  }
  const context = api.data.cases.find(row => row.name === 'context_recall')
  assert.equal(context?.response?.usage?.prompt_tokens, 1044, `${size}B: context qualification length changed`)
  models[`${size}B`] = {
    language: await artifact(language, languageBytes, languageHash),
    vision_projector: await artifact(projector, projectorBytes, projectorHash),
    result: 'PASS Windows CUDA text, single-image GUI/API, function tools, context recall, seeded sampling and exact native speculative parity.',
    qualified_context_prompt_tokens: 1044,
    receipts: {
      api: serverReceipt(api), gui: serverReceipt(gui), sampled_draft: serverReceipt(sampled),
      gui_image_screenshot: await fileIdentity(`qa/bonsai2/cuda/${size}b-gui-supported.png`),
      gui_text_screenshot: await fileIdentity(`qa/bonsai2/cuda/${size}b-gui-supported-text.png`),
      native_benchmark: { ...native.identity, binary: probeName }, reference_benchmark: reference.identity,
    },
    native_weight_placement: Object.fromEntries(native.data.cases.map(row => [row.name,
      Object.fromEntries(['baseline', 'speculative'].map(mode => [mode, row[mode].cuda_capacity_plan]))])),
    reference_layer_placement: Object.fromEntries(['baseline', 'speculative'].map(mode => [mode,
      reference.data.modes[mode].offload.filter(line => /offloaded \d+\/\d+ layers to GPU/.test(line))])),
  }
}

const manifest = {
  schema_version: 2, generated_at: new Date().toISOString(), status: 'pass',
  scope: 'Exact official Bonsai 2 2B/4B/9B/27B artifacts on Windows x86_64 CUDA, including the packaged GUI.',
  host: { ...host, identity_source: 'Consistent native/reference benchmark receipts; GPU and driver reported by reference harness.' },
  native: {
    language_backend: 'bonsai2_cuda_runtime; packed PTQ1_0 and native BF16 language computations on CUDA',
    image_backend: 'Accurate f32 CUDA image projector with the matching pinned Q8_0 companion',
    constrained_memory: '27B streams packed weights from host memory and executes language layer computations on CUDA. This is distinct from reference hybrid CPU/GPU layer execution.',
    draft: 'Same official Bonsai 2 2B model, three-token proposal window, batched target verification and recurrent/KV rollback.',
    tested_context: '1044 prompt tokens with context recall; benchmark and device-test allocation is 2048 slots, not a full 2048-token semantic qualification.',
  },
  binaries,
  provenance: {
    build_record: await fileIdentity(recordPath),
    association_policy: 'Source commits/build options are explicit operator records. Probe and test binary digests must match existing receipts. Server records are matched to original receipt build IDs. Existing receipt source_sha256 fields remain unmodified snapshots of source at receipt creation, not proof of binary build inputs.',
    selected_source_hash_format: 'SHA-256 of exact Git blob bytes at each recorded commit, independent of checkout newline conversion or later edits.',
    generator: await fileIdentity('qa/bonsai2/cuda-manifest.mjs'),
    evidence_checker: await fileIdentity('qa/bonsai2/check-cuda-evidence.mjs'),
  },
  models,
  device_tests: { ...cache.identity, binary: testsName, passed: cache.data.tests.length,
    coverage: 'PTQ/FWHT/embedding and BF16 fixtures; all four sizes: every logit, KV, convolution/SSM state, rollback, greedy/seeded generation, stop and cancellation; 2B attention-group and split-K boundaries.' },
  reference: { repository: 'https://github.com/PrismML-Eng/llama.cpp', ...referenceIdentity,
    configuration_note: '2B/4B/9B fit all target layers on CUDA. 27B uses automatic hybrid CPU/GPU placement; actual per-mode placement is retained above. Draft state can change placement.' },
  performance: { ...comparison.identity,
    metric: 'Median of three warmed requests, 32 generated tokens divided by complete uncached request wall seconds.',
    common_work: 'Exact prompt bytes, eight threads, f16 KV, 2048 context slots, one eight-token warmup, greedy sampling, same 2B drafter and three-token window.',
    timing_boundary: 'Native measures the Rust generation call after tokenization. Reference measures a localhost HTTP completion including tokenization, transport, serialization and JSON parsing. Loading and warmup are excluded; prefill and decoding are included.',
    parity_note: 'Native ordinary/draft token IDs must match exactly. Cross-engine differences and reference ordinary/draft divergences remain in comparison.json and the original receipts.',
  },
  not_qualified: ['Apple Silicon and other operating systems or GPU families',
    'Model-native 262K or extended context', 'Multiple images, audio and video',
    'Previous-generation DSpark assistants as Bonsai 2 drafters'],
  reproduction: 'qa/bonsai2/cuda/README.md', historical_cpu_manifest: 'qa/bonsai2/manifest.json',
  evidence_check: 'node qa/bonsai2/check-cuda-evidence.mjs',
}
await mkdir(dirname(resolve(root, destination)), { recursive: true })
await writeFile(resolve(root, destination), `${JSON.stringify(manifest, null, 2)}\n`)
console.log(`Wrote ${relativePath(destination)}: four model/projector pairs, ${Object.keys(binaries).length} binaries, ${cache.data.tests.length} CUDA tests.`)
