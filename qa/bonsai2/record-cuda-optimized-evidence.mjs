// COMMIT PROBE TEST_BINARY SERVER SERVER_BUILD_ID [DIRECTORY]
// Run only after qualification has finished using these retained executables.
import assert from 'node:assert/strict'
import { readFile, readdir, writeFile } from 'node:fs/promises'
import { createReadStream } from 'node:fs'
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { resolve } from 'node:path'
const [commit, probe, tests, server, buildId, directory = 'qa/bonsai2/cuda-optimized'] = process.argv.slice(2)
assert.ok(commit && probe && tests && server && buildId, 'COMMIT PROBE TEST_BINARY SERVER SERVER_BUILD_ID required')
assert.match(commit, /^[a-f0-9]{40}$/)
const digest = bytes => createHash('sha256').update(bytes).digest('hex')
const lf = bytes => Buffer.from(bytes.toString('utf8').replaceAll('\r\n', '\n'))
async function hash(path) { const h = createHash('sha256'); for await (const chunk of createReadStream(path)) h.update(chunk); return h.digest('hex') }
const native = JSON.parse(await readFile(resolve(directory, '2b-native-benchmark.json'), 'utf8'))
const cache = JSON.parse(await readFile(resolve(directory, 'cache-parity.json'), 'utf8'))
const paths = [...new Set([...Object.keys(native.source_sha256), ...Object.keys(cache.source_sha256),
  'Cargo.toml', 'Cargo.lock', '.cargo/config.toml', 'build.rs', 'src/api/mod.rs', 'src/bonsai2.rs',
  'src/execution_plan.rs', 'examples/bonsai2_server.rs'])]
const source_sha256 = {}, git_source_sha256 = {}
for (const path of paths) {
  const bytes = await readFile(path)
  const blob = execFileSync('git', ['show', `${commit}:${path}`], { windowsHide: true, maxBuffer: 10 * 1024 * 1024 })
  assert.equal(digest(lf(bytes)), digest(lf(blob)), `${path} differs from build commit`)
  source_sha256[path] = digest(bytes)
  git_source_sha256[path] = digest(blob)
}
const binaries = {}
for (const [role, path] of Object.entries({ probe, tests, server })) binaries[role] = { path, sha256: await hash(path) }
binaries.server.receipt_build_id = buildId
assert.equal(binaries.probe.sha256, native.probe_sha256)
assert.equal(binaries.tests.sha256, cache.test_binary_sha256)
const artifacts = {}
for (const name of await readdir(directory)) if (name !== 'manifest.json') artifacts[name] = await hash(resolve(directory, name))
const build_logs = {}
for (const name of ['build-final.log', 'build-tests-retry.log', 'build-tests-retry.jsonl', 'clippy.log']) {
  const path = `output/bonsai2-profile/${name}`
  build_logs[path] = await hash(path)
}
const manifest = { schema: 1, date: new Date().toISOString(), source_commit: commit,
  source_tree: execFileSync('git', ['rev-parse', `${commit}^{tree}`], { encoding: 'utf8', windowsHide: true }).trim(),
  build_configuration: { profile: 'release', features: ['cuda'], target_cpu: 'x86-64-v3', lto: false, codegen_units: 16,
    benchmark_rust_opt_level: 3, test_crate_rust_opt_level: 1,
    rustc: execFileSync('rustc', ['--version'], { encoding: 'utf8', windowsHide: true }).trim(),
    notes: 'The opt-level=3 test build exhausted host memory. The test-only retry used jobs=1 and profile.release.package.camelid.opt-level=1. Probe/server retained opt-level=3 and jobs=2. Both compile the identical CUDA source through the same NVRTC settings. Runtime source is committed; documentation/evidence edits during compilation may add a dirty build suffix.' },
  scope: 'Windows CUDA projection optimization; original CPU/API qualification is retained in ../cuda and the parent directory. New GUI text/image runs use the production frontend in the native library-server harness, not an installed Tauri shell.',
  host: { cpu: native.cpu, physical_memory_bytes: native.physical_memory_bytes, gpu: 'RTX 3060 Laptop, 6144 MiB, driver 576.83', cuda_runtime: '12.9' },
  binaries, source_sha256, git_source_sha256, build_logs, artifacts }
await writeFile(resolve(directory, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
console.log(`Recorded ${Object.keys(artifacts).length} optimization artifacts`)
