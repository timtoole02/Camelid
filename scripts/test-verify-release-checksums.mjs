#!/usr/bin/env node
// Self-test for verify-release-checksums.mjs (run in CI by the
// validation-scripts job's test-*.mjs glob).
//
// Covers the two cases the previous inline `awk '{print $1; exit}'` publish
// gate accepted and a user's own `shasum -c` would not: a correct hash under
// the wrong embedded filename, and a valid first record followed by garbage.
import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { CHECKSUM_SUFFIX, checkChecksumRecord, checkStagedSet } from './verify-release-checksums.mjs'

const has = (failures, pattern) => failures.some((f) => pattern.test(f))
const HASH = 'a'.repeat(64)
const OTHER = 'b'.repeat(64)
const EXPECTED = ['camelid-macos-arm64.tar.gz', 'camelid-linux-x86_64.tar.gz', 'camelid-windows-x64.zip']
const staged = (...extra) => [...EXPECTED, ...EXPECTED.map((a) => `${a}${CHECKSUM_SUFFIX}`), ...extra]

// ---------------------------------------------------------------------------
// staged set — only the three server archives and their sidecars
// ---------------------------------------------------------------------------
assert.deepEqual(checkStagedSet(staged(), EXPECTED), [], 'exactly the expected archives and sidecars must pass')
assert.ok(
  has(checkStagedSet(staged('camelid-linux-x86_64-check-receipt.json'), EXPECTED), /unexpected file: camelid-linux-x86_64-check-receipt\.json/),
  'a diagnostic receipt must never be published',
)
assert.ok(
  has(checkStagedSet(staged('camelid-desktop-windows-x64.zip'), EXPECTED), /unexpected file/),
  'the independently published desktop artifact must not be swept in',
)
assert.ok(
  has(checkStagedSet(staged('camelid_0.4.4_x64-setup.exe'), EXPECTED), /unexpected file/),
  'a stray installer must not be published by the server release job',
)
assert.ok(
  has(checkStagedSet(staged().filter((n) => n !== 'camelid-windows-x64.zip'), EXPECTED), /missing expected archive: camelid-windows-x64\.zip/),
  'a missing archive must fail closed',
)
assert.ok(
  has(
    checkStagedSet(staged().filter((n) => n !== `camelid-windows-x64.zip${CHECKSUM_SUFFIX}`), EXPECTED),
    /missing checksum sidecar: camelid-windows-x64\.zip\.sha256/,
  ),
  'an archive with no sidecar must fail closed',
)
assert.ok(has(checkStagedSet([], EXPECTED), /missing expected archive/), 'an empty staging directory must fail')

// ---------------------------------------------------------------------------
// checksum records
// ---------------------------------------------------------------------------
assert.deepEqual(
  checkChecksumRecord('camelid-windows-x64.zip', `${HASH}  camelid-windows-x64.zip`, HASH),
  [],
  'a correct single record must pass (PowerShell writes no trailing newline)',
)
assert.deepEqual(
  checkChecksumRecord('camelid-linux-x86_64.tar.gz', `${HASH}  camelid-linux-x86_64.tar.gz\n`, HASH),
  [],
  'the shasum convention must also pass',
)
assert.ok(
  has(checkChecksumRecord('camelid-windows-x64.zip', `${HASH}  some-other-file.zip`, HASH), /names "some-other-file\.zip"/),
  'a correct hash under the wrong embedded filename must fail',
)
assert.ok(
  has(checkChecksumRecord('camelid-windows-x64.zip', `${HASH}  camelid-windows-x64.zip`, OTHER), /sha256 mismatch/),
  'a hash that does not describe the archive must fail',
)
assert.ok(
  has(checkChecksumRecord('camelid-windows-x64.zip', `${HASH}  camelid-windows-x64.zip\ngarbage`, HASH), /exactly one record/),
  'a valid first record followed by garbage must fail',
)
assert.ok(
  has(checkChecksumRecord('camelid-windows-x64.zip', `${HASH}  camelid-windows-x64.zip\n${OTHER}  extra.zip`, HASH), /exactly one record/),
  'a multi-record sidecar must fail',
)
assert.ok(has(checkChecksumRecord('camelid-windows-x64.zip', '', HASH), /empty/), 'an empty sidecar must fail')
assert.ok(has(checkChecksumRecord('camelid-windows-x64.zip', 'nonsense', HASH), /64-hex/), 'a malformed sidecar must fail')

// ---------------------------------------------------------------------------
// end to end against real files
// ---------------------------------------------------------------------------
const { createHash } = await import('node:crypto')
const workspace = await mkdtemp(join(tmpdir(), 'camelid-checksums-test-'))
const payload = 'archive bytes'
const realHash = createHash('sha256').update(payload).digest('hex')
await writeFile(join(workspace, 'camelid-windows-x64.zip'), payload)
await writeFile(join(workspace, `camelid-windows-x64.zip${CHECKSUM_SUFFIX}`), `${realHash}  camelid-windows-x64.zip`)

const { sha256File } = await import('./check-release-artifact.mjs')
assert.equal(await sha256File(join(workspace, 'camelid-windows-x64.zip')), realHash, 'sha256File must agree with node:crypto')
assert.deepEqual(
  checkChecksumRecord('camelid-windows-x64.zip', `${realHash}  camelid-windows-x64.zip`, realHash),
  [],
  'a real archive and its real sidecar must verify',
)

await rm(workspace, { recursive: true, force: true })
console.log('test-verify-release-checksums: all checks passed')
