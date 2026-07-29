#!/usr/bin/env node
// Self-test for check-release-artifact.mjs (run in CI by the validation-scripts
// job's test-*.mjs glob, so it gates every PR without touching ci.yml).
//
// Strategy: the pure functions are exercised directly, and the manifest
// contract is exercised against REAL directory trees written to a temp dir —
// including real symlinks — so readTree's lstat/readlink handling is covered
// rather than mocked.
import assert from 'node:assert/strict'
import { chmod, mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import {
  ARTIFACT_SPECS,
  checkManifest,
  compareVersionToTag,
  normalizeVersion,
  parseChecksumFile,
  parseVersionOutput,
  readTree,
  resolvePayloadRoot,
  resolveSymlinkChain,
} from './check-release-artifact.mjs'

const workspace = await mkdtemp(join(tmpdir(), 'camelid-artifact-test-'))
let symlinksSupported = true

/**
 * Materialize a fixture tree.
 * Values: string -> file content; {symlink} -> symbolic link; {dir:true} -> directory.
 */
async function makeTree(name, layout) {
  const root = join(workspace, name)
  await mkdir(root, { recursive: true })
  // Two passes so a symlink may point at a file declared after it.
  for (const [relative, value] of Object.entries(layout)) {
    if (typeof value === 'object' && value !== null && value.symlink) continue
    const target = join(root, relative)
    await mkdir(dirname(target), { recursive: true })
    if (typeof value === 'object' && value !== null && value.dir) await mkdir(target, { recursive: true })
    else await writeFile(target, String(value))
  }
  for (const [relative, value] of Object.entries(layout)) {
    if (!(typeof value === 'object' && value !== null && value.symlink)) continue
    const target = join(root, relative)
    await mkdir(dirname(target), { recursive: true })
    try {
      await symlink(value.symlink, target)
    } catch (error) {
      // Windows needs Developer Mode or elevation to create symlinks. Degrade
      // to a regular file and let the caller skip link-specific assertions,
      // rather than failing the whole suite for an unrelated OS restriction.
      if (error.code === 'EPERM' || error.code === 'ENOSYS') {
        symlinksSupported = false
        await writeFile(target, `link->${value.symlink}`)
      } else {
        throw error
      }
    }
  }
  return root
}

const LINUX = ARTIFACT_SPECS['linux-x86_64']
const WINDOWS = ARTIFACT_SPECS['windows-x64']
const DESKTOP = ARTIFACT_SPECS['desktop-windows-x64']
const MACOS = ARTIFACT_SPECS['macos-arm64']

const goodLinux = {
  camelid: 'ELF-ish payload',
  'README.md': '# Camelid',
  LICENSE: 'MIT',
  'THIRD_PARTY_NOTICES.md': 'notices',
  'libnvrtc.so.12.9.86': 'nvrtc real payload',
  'libnvrtc.so.12': { symlink: 'libnvrtc.so.12.9.86' },
  'libnvrtc-builtins.so.12.9': 'builtins payload',
}
const goodWindows = {
  'camelid.exe': 'PE payload',
  'README.md': '# Camelid',
  LICENSE: 'MIT',
  'THIRD_PARTY_NOTICES.md': 'notices',
  'nvrtc64_120_0.dll': 'nvrtc payload',
  'nvrtc-builtins64_129.dll': 'builtins payload',
}

const failuresFor = async (name, layout, spec, options = {}) =>
  checkManifest(await readTree(await makeTree(name, layout)), spec, options)

const has = (failures, pattern) => failures.some((f) => pattern.test(f))
const omit = (layout, ...keys) => Object.fromEntries(Object.entries(layout).filter(([k]) => !keys.includes(k)))

// ---------------------------------------------------------------------------
// parseChecksumFile — two producers, two trailing-byte conventions
// ---------------------------------------------------------------------------
const HASH = 'a'.repeat(64)
assert.deepEqual(
  parseChecksumFile(`${HASH}  camelid-linux-x86_64.tar.gz\n`),
  { hash: HASH, name: 'camelid-linux-x86_64.tar.gz' },
  'shasum output (two spaces, trailing newline) must parse',
)
assert.deepEqual(
  parseChecksumFile(`${HASH}  camelid-windows-x64.zip`),
  { hash: HASH, name: 'camelid-windows-x64.zip' },
  'PowerShell Out-File -NoNewline output must parse',
)
assert.deepEqual(parseChecksumFile(`${HASH}  a.zip\r\n`).name, 'a.zip', 'CRLF must parse')
assert.deepEqual(parseChecksumFile(`\uFEFF${HASH}  a.zip`).name, 'a.zip', 'a UTF-8 BOM must not break the parse')
assert.deepEqual(parseChecksumFile(`${HASH} *a.tar.gz`).name, 'a.tar.gz', "coreutils' binary-mode * marker must parse")
assert.equal(parseChecksumFile(`${'A'.repeat(64)}  a.zip`).hash, HASH, 'hashes must normalize to lower case')
assert.throws(() => parseChecksumFile(''), /empty/, 'an empty sidecar must throw')
assert.throws(() => parseChecksumFile('not-a-hash  a.zip'), /64-hex/, 'a malformed sidecar must throw')
assert.throws(() => parseChecksumFile(`${'a'.repeat(63)}  a.zip`), /64-hex/, 'a short hash must throw')
// `shasum -c` consumes every record, so honouring only the first would verify
// something different from what a user's own check does.
assert.throws(
  () => parseChecksumFile(`${HASH}  a.zip\ngarbage garbage`),
  /exactly one record/,
  'a valid first record followed by garbage must throw',
)
assert.throws(
  () => parseChecksumFile(`${HASH}  a.zip\n${'b'.repeat(64)}  b.zip`),
  /exactly one record/,
  'a multi-record sidecar must throw',
)

// ---------------------------------------------------------------------------
// version comparison
// ---------------------------------------------------------------------------
assert.equal(normalizeVersion('v0.4.4'), '0.4.4')
assert.equal(normalizeVersion('  0.4.4 '), '0.4.4')
assert.equal(compareVersionToTag('0.4.4', 'v0.4.4'), null, 'a bare version must match a v-prefixed tag')
assert.equal(compareVersionToTag('v0.4.4', 'v0.4.4'), null, 'identical strings must match')
assert.match(
  compareVersionToTag('v0.4.4-3-gabcdef', 'v0.4.4'),
  /ahead of tag/,
  'a git-describe suffix means the tag is not at this commit and must fail',
)
assert.match(compareVersionToTag('0.4.3', 'v0.4.4'), /does not match/, 'a stale Cargo.toml version must fail')
assert.match(compareVersionToTag('', 'v0.4.4'), /empty version/, 'an empty version must fail')
assert.equal(parseVersionOutput('camelid 0.4.4\n'), '0.4.4')
assert.equal(parseVersionOutput('camelid v0.4.4-3-gabcdef\n'), 'v0.4.4-3-gabcdef')
assert.equal(parseVersionOutput('\n\ncamelid 0.4.4\n'), '0.4.4', 'leading blank lines must be skipped')
assert.equal(parseVersionOutput(''), null)

// ---------------------------------------------------------------------------
// layout — the tar/zip asymmetry
// ---------------------------------------------------------------------------
assert.deepEqual(
  resolvePayloadRoot(['camelid-linux-x86_64'], LINUX),
  { root: 'camelid-linux-x86_64', failures: [] },
  'a tarball extracts to exactly one wrapper directory',
)
assert.deepEqual(resolvePayloadRoot(['camelid.exe', 'README.md'], WINDOWS).failures, [], 'a zip extracts flat')
{
  const { root, failures } = resolvePayloadRoot(['camelid-linux-x86_64', 'stray.txt'], LINUX)
  assert.equal(root, 'camelid-linux-x86_64', 'the payload root is still resolved so the manifest can be reported too')
  assert.ok(has(failures, /expected exactly one top-level directory/), 'a stray top-level entry must be reported')
}
assert.ok(
  has(resolvePayloadRoot(['nope'], LINUX).failures, /expected exactly one top-level directory/),
  'a missing wrapper directory must be reported',
)
assert.ok(
  has(resolvePayloadRoot(['camelid-windows-x64'], WINDOWS).failures, /found a single wrapper directory/),
  'a zip that gained a wrapper directory must be reported',
)

// ---------------------------------------------------------------------------
// manifest — happy paths
// ---------------------------------------------------------------------------
assert.deepEqual(
  await failuresFor('linux-good', goodLinux, LINUX, { requireNvrtc: true }),
  [],
  'a correctly packaged linux payload must pass with NVRTC required',
)
assert.deepEqual(
  await failuresFor('windows-good', goodWindows, WINDOWS, { requireNvrtc: true }),
  [],
  'a correctly packaged windows payload must pass',
)
assert.deepEqual(
  await failuresFor(
    'macos-good',
    { camelid: 'macho', 'README.md': '#', LICENSE: 'MIT', 'THIRD_PARTY_NOTICES.md': 'n' },
    MACOS,
    { requireNvrtc: false },
  ),
  [],
  'macOS ships no NVRTC and must pass without it',
)
assert.deepEqual(
  await failuresFor('desktop-good', { ...goodWindows, 'camelid-desktop.exe': 'PE', 'DESKTOP_README.md': 'd' }, DESKTOP, {
    requireNvrtc: true,
  }),
  [],
  'a correctly packaged desktop portable payload must pass',
)
assert.deepEqual(
  await failuresFor('linux-no-nvrtc-not-required', omit(goodLinux, 'libnvrtc.so.12.9.86', 'libnvrtc.so.12', 'libnvrtc-builtins.so.12.9'), LINUX, {
    requireNvrtc: false,
  }),
  [],
  'without --require-nvrtc a CPU-only payload is legitimate (local dry runs have no CUDA toolkit)',
)

// ---------------------------------------------------------------------------
// manifest — required payload
// ---------------------------------------------------------------------------
assert.ok(
  has(await failuresFor('linux-no-binary', omit(goodLinux, 'camelid'), LINUX, { requireNvrtc: true }), /missing required entry: camelid/),
  'a missing binary must fail',
)
assert.ok(
  has(await failuresFor('linux-no-license', omit(goodLinux, 'LICENSE'), LINUX, {}), /missing required entry: LICENSE/),
  'a missing LICENSE must fail',
)
assert.ok(
  has(await failuresFor('linux-empty-binary', { ...goodLinux, camelid: '' }, LINUX, {}), /camelid is zero bytes/),
  'a zero-byte binary must fail',
)
assert.ok(
  has(await failuresFor('linux-binary-is-dir', { ...omit(goodLinux, 'camelid'), camelid: { dir: true } }, LINUX, {}), /is a dir, expected a file/),
  'a directory where the binary should be must fail',
)

// ---------------------------------------------------------------------------
// manifest — NVRTC
// ---------------------------------------------------------------------------
assert.ok(
  has(
    await failuresFor('linux-no-compiler', omit(goodLinux, 'libnvrtc.so.12.9.86', 'libnvrtc.so.12'), LINUX, { requireNvrtc: true }),
    /no NVRTC compiler library/,
  ),
  'a linux tarball with no NVRTC compiler library at all would silently fall back to CPU and must fail',
)
// The soname is the whole point: cudarc's dlopen candidates are libnvrtc.so and
// libnvrtc.so.<major>, never the three-component real name. A tarball that kept
// libnvrtc.so.12.9.86 but lost the libnvrtc.so.12 link is a CPU-only artifact
// that looks complete, so the compiler role must be searched for by soname.
assert.ok(
  has(
    await failuresFor('linux-soname-only-missing', omit(goodLinux, 'libnvrtc.so.12'), LINUX, { requireNvrtc: true }),
    /no NVRTC compiler library/,
  ),
  'the versioned real file WITHOUT its soname link cannot be dlopen’d and must fail',
)
assert.ok(
  has(await failuresFor('linux-no-builtins', omit(goodLinux, 'libnvrtc-builtins.so.12.9'), LINUX, { requireNvrtc: true }), /no NVRTC builtins/),
  'an incomplete NVRTC pair must fail',
)
assert.ok(
  has(
    await failuresFor('windows-no-builtins', omit(goodWindows, 'nvrtc-builtins64_129.dll'), WINDOWS, { requireNvrtc: true }),
    /no NVRTC builtins/,
  ),
  'package-windows-cuda.ps1 only warns about an incomplete pair; this gate must fail',
)
assert.ok(
  has(await failuresFor('linux-alt', { ...goodLinux, 'libnvrtc.alt.so.12.9.86': 'alt backend' }, LINUX, { requireNvrtc: true }), /alternate JIT backend/),
  'the .alt backend is ~90 MB cudarc never loads and must not ship',
)

// ---------------------------------------------------------------------------
// manifest — duplicated payload (the v0.4.3 regression, post-archive)
// ---------------------------------------------------------------------------
assert.ok(
  has(
    await failuresFor(
      'linux-duplicated',
      { ...omit(goodLinux, 'libnvrtc.so.12'), 'libnvrtc.so.12': 'nvrtc real payload' },
      LINUX,
      { requireNvrtc: true },
    ),
    /duplicate payload/,
  ),
  'a dereferenced symlink ships the same library twice and must fail',
)
// 'nvrtc real payload' and 'builtins payload!!' are both 18 bytes and differ in
// content — upstream's size comparison would flag this pair, hashing must not.
assert.equal('nvrtc real payload'.length, 'builtins payload!!'.length, 'the fixture pair must be the same size')
assert.deepEqual(
  (await failuresFor('linux-same-size-different-bytes', { ...goodLinux, 'libnvrtc-builtins.so.12.9': 'builtins payload!!' }, LINUX, {
    requireNvrtc: true,
  })).filter((f) => /duplicate payload/.test(f)),
  [],
  'hashing content (not comparing sizes) means same-size different libraries must NOT false-positive',
)
assert.deepEqual(
  (await failuresFor('linux-identical-docs', { ...goodLinux, 'README.md': 'same', LICENSE: 'same' }, LINUX, { requireNvrtc: true })).filter((f) =>
    /duplicate payload/.test(f),
  ),
  [],
  'duplicate detection is scoped to the NVRTC family; two identical docs must not block a release',
)

// ---------------------------------------------------------------------------
// manifest — symlink integrity and mode, as PURE entry lists
//
// checkManifest is a pure function of an entry list, so these run identically
// on every host. The filesystem-backed variants below additionally cover
// readTree's lstat/readlink handling wherever the OS permits it.
// ---------------------------------------------------------------------------
const entry = (path, over = {}) => ({ path, kind: 'file', size: 16, mode: 0o755, sha256: `hash-${path}`, ...over })
const link = (path, linkTarget) => ({ path, kind: 'symlink', size: 0, mode: 0o777, linkTarget })
/**
 * A VALID linux payload. The `libnvrtc.so.<major>` soname belongs in the base
 * fixture, not only in the tests that are about symlinks: it is the name cudarc
 * dlopen's, so a payload without it is a CPU-only artifact rather than a
 * release. Any `extra` entry REPLACES the base entry with the same path, which
 * is how the variants below swap in a dangling or escaping link.
 */
const linuxEntries = (...extra) => {
  const base = [
    entry('camelid'),
    entry('README.md'),
    entry('LICENSE'),
    entry('THIRD_PARTY_NOTICES.md'),
    entry('libnvrtc.so.12.9.86'),
    entry('libnvrtc-builtins.so.12.9'),
    link('libnvrtc.so.12', 'libnvrtc.so.12.9.86'),
  ]
  const replaced = new Set(extra.map((e) => e.path))
  return [...base.filter((e) => !replaced.has(e.path)), ...extra]
}

assert.deepEqual(
  checkManifest(linuxEntries(link('libnvrtc.so.12', 'libnvrtc.so.12.9.86')), LINUX, { requireNvrtc: true }),
  [],
  'a symlink resolving to a staged file is the correct packaging and must pass',
)
assert.ok(
  has(checkManifest(linuxEntries(link('libnvrtc.so.12', 'libnvrtc.so.nope')), LINUX, { requireNvrtc: true }), /is dangling inside the archive/),
  'a link that resolves nowhere means dlopen fails and the GPU never engages',
)
assert.ok(
  has(checkManifest(linuxEntries(link('libnvrtc.so.12', '../../etc/passwd')), LINUX, { requireNvrtc: true }), /escapes the payload/),
  'a relative link pointing outside the payload must fail',
)
assert.ok(
  has(checkManifest(linuxEntries(link('libnvrtc.so.12', '/usr/lib/libnvrtc.so.12')), LINUX, { requireNvrtc: true }), /escapes the payload/),
  'an absolute link target must fail',
)
assert.ok(
  has(
    checkManifest(linuxEntries(entry('libnvrtc.so.12', { sha256: 'hash-libnvrtc.so.12.9.86' })), LINUX, { requireNvrtc: true }),
    /duplicate payload/,
  ),
  'two byte-identical NVRTC real files mean a symlink was dereferenced',
)
assert.ok(
  has(checkManifest(linuxEntries().map((e) => (e.path === 'camelid' ? { ...e, mode: 0o644 } : e)), LINUX, { requireNvrtc: true, checkExecutableBit: true }), /is not executable/),
  'a binary that lost +x extracts fine and then cannot be run',
)
assert.deepEqual(
  checkManifest(linuxEntries().map((e) => (e.path === 'camelid' ? { ...e, mode: 0o644 } : e)), LINUX, {
    requireNvrtc: true,
    checkExecutableBit: false,
  }),
  [],
  'the mode check must stay off where the filesystem has no mode bits to lose',
)

// ---------------------------------------------------------------------------
// resolveSymlinkChain
// ---------------------------------------------------------------------------
{
  const map = (...es) => new Map(es.map((e) => [e.path, e]))
  const real = entry('libnvrtc.so.12.9.86')

  assert.equal(resolveSymlinkChain(map(real), 'libnvrtc.so.12.9.86').kind, 'resolved', 'a regular file resolves to itself')
  assert.equal(
    resolveSymlinkChain(map(real, link('libnvrtc.so.12', 'libnvrtc.so.12.9.86')), 'libnvrtc.so.12').path,
    'libnvrtc.so.12.9.86',
    'a one-hop link resolves to its target',
  )
  assert.equal(
    resolveSymlinkChain(map(real, link('a', 'b'), link('b', 'libnvrtc.so.12.9.86')), 'a').path,
    'libnvrtc.so.12.9.86',
    'a multi-hop chain resolves to the terminal file',
  )
  assert.equal(resolveSymlinkChain(map(link('a', 'missing')), 'a').kind, 'dangling')
  assert.equal(resolveSymlinkChain(map(link('a', 'a')), 'a').kind, 'cycle', 'a self-referential link is a cycle, not a satisfied target')
  assert.equal(resolveSymlinkChain(map(link('a', 'b'), link('b', 'a')), 'a').kind, 'cycle')
  assert.equal(resolveSymlinkChain(map(link('a', '/usr/lib/x')), 'a').kind, 'escape')
  assert.equal(resolveSymlinkChain(map(link('a', '../../etc/passwd')), 'a').kind, 'escape')
  {
    // A chain longer than the hop bound must terminate rather than spin.
    const long = []
    for (let i = 0; i < 12; i += 1) long.push(link(`l${i}`, `l${i + 1}`))
    assert.equal(resolveSymlinkChain(map(...long), 'l0').kind, 'too-deep')
  }
}

// ---------------------------------------------------------------------------
// NVRTC sonames must RESOLVE to the right library, not merely be named like one
//
// cudarc dlopen's the soname and dlopen follows links, so a link whose name
// matches while its target is the wrong library reproduces exactly the
// silent CPU-fallback packaging failure this gate exists to catch. The
// driverless boot smoke cannot catch it either, because it deliberately
// exercises the CPU path.
// ---------------------------------------------------------------------------
const withoutCompiler = [
  entry('camelid'),
  entry('README.md'),
  entry('LICENSE'),
  entry('THIRD_PARTY_NOTICES.md'),
  entry('libnvrtc-builtins.so.12.9'),
]

assert.deepEqual(
  checkManifest([...withoutCompiler, entry('libnvrtc.so.12.9.86'), link('libnvrtc.so.12', 'libnvrtc.so.12.9.86')], LINUX, { requireNvrtc: true }),
  [],
  'the real packaging shape (soname symlink -> versioned real file) must still pass',
)
assert.ok(
  has(checkManifest([...withoutCompiler, link('libnvrtc.so.12', 'libnvrtc-builtins.so.12.9')], LINUX, { requireNvrtc: true }), /not an NVRTC compiler library/),
  'a compiler soname pointing at the BUILTINS library must fail',
)
assert.ok(
  has(checkManifest([...withoutCompiler, link('libnvrtc.so.12', 'LICENSE')], LINUX, { requireNvrtc: true }), /not an NVRTC compiler library/),
  'a compiler soname pointing at an unrelated file must fail',
)
assert.ok(
  has(checkManifest([...withoutCompiler, link('libnvrtc.so.12', 'libnvrtc.so.12')], LINUX, { requireNvrtc: true }), /no usable NVRTC compiler/),
  'a self-referential compiler soname must fail',
)
assert.ok(
  has(
    checkManifest([...withoutCompiler, link('libnvrtc.so.12', 'libnvrtc.so.13'), link('libnvrtc.so.13', 'libnvrtc.so.12')], LINUX, {
      requireNvrtc: true,
    }),
    /no usable NVRTC compiler/,
  ),
  'a cyclic compiler soname must fail',
)
assert.ok(
  has(
    checkManifest(
      [entry('camelid'), entry('README.md'), entry('LICENSE'), entry('THIRD_PARTY_NOTICES.md'), entry('libnvrtc.so.12.9.86'), link('libnvrtc-builtins.so.12.9', 'libnvrtc.so.12.9.86')],
      LINUX,
      { requireNvrtc: true },
    ),
    /not an NVRTC builtins library/,
  ),
  'a builtins soname pointing at the compiler must fail',
)
assert.ok(
  has(
    checkManifest(
      [
        entry('camelid.exe'),
        entry('README.md'),
        entry('LICENSE'),
        entry('THIRD_PARTY_NOTICES.md'),
        entry('nvrtc-builtins64_129.dll'),
        link('nvrtc64_120_0.dll', 'nvrtc-builtins64_129.dll'),
      ],
      WINDOWS,
      { requireNvrtc: true },
    ),
    /not an NVRTC compiler library/,
  ),
  'the same wrong-family check must apply on Windows',
)

// ---------------------------------------------------------------------------
// manifest — symlink integrity, filesystem-backed
// ---------------------------------------------------------------------------
if (symlinksSupported) {
  assert.ok(
    has(
      await failuresFor('linux-dangling', { ...goodLinux, 'libnvrtc.so.12': { symlink: 'libnvrtc.so.does-not-exist' } }, LINUX, {
        requireNvrtc: true,
      }),
      /is dangling inside the archive/,
    ),
    'a link that resolves nowhere means dlopen fails and the GPU never engages',
  )
  assert.ok(
    has(await failuresFor('linux-escape', { ...goodLinux, 'libnvrtc.so.12': { symlink: '../../etc/passwd' } }, LINUX, {}), /escapes the payload/),
    'a link pointing outside the payload must fail',
  )
  assert.ok(
    has(await failuresFor('linux-absolute', { ...goodLinux, 'libnvrtc.so.12': { symlink: '/usr/lib/libnvrtc.so.12' } }, LINUX, {}), /escapes the payload/),
    'an absolute link target must fail',
  )
} else {
  console.log('test-check-release-artifact: symlink cases skipped (this host cannot create symlinks)')
}

// ---------------------------------------------------------------------------
// manifest — junk and label-specific forbidden entries
// ---------------------------------------------------------------------------
assert.ok(
  has(await failuresFor('windows-pdb', { ...goodWindows, 'camelid.pdb': 'symbols' }, WINDOWS, {}), /debug symbol database/),
  'a stray .pdb must not ship',
)
assert.ok(
  has(await failuresFor('linux-target-dir', { ...goodLinux, 'target/debug/leftover': 'x' }, LINUX, {}), /cargo target directory/),
  'a leaked target/ tree must not ship',
)
assert.ok(
  has(await failuresFor('linux-dotenv', { ...goodLinux, '.env': 'SECRET=1' }, LINUX, {}), /credential-shaped file/),
  'a credential-shaped file must not ship',
)
assert.ok(
  has(
    await failuresFor(
      'desktop-with-installer',
      { ...goodWindows, 'camelid-desktop.exe': 'PE', 'DESKTOP_README.md': 'd', 'camelid-desktop_0.4.4_x64-setup.exe': 'installer' },
      DESKTOP,
      { requireNvrtc: true },
    ),
    /must be moved out of the portable zip/,
  ),
  'the NSIS installer is a separate artifact and must not be inside the portable zip',
)
assert.deepEqual(
  (await failuresFor('windows-setup-allowed-elsewhere', { ...goodWindows, 'camelid_0.4.4_x64-setup.exe': 'installer' }, WINDOWS, {})).filter((f) =>
    /moved out of the portable zip/.test(f),
  ),
  [],
  'the installer rule is desktop-specific and must not leak onto other labels',
)

// ---------------------------------------------------------------------------
// closed payload contract
//
// Required-files-plus-junk-patterns is an OPEN set: it accepts anything nobody
// thought to forbid. These assert the allowlist actually closes it, so an
// accidental future glob or a staged credential cannot ride along into a
// public download.
// ---------------------------------------------------------------------------
assert.ok(
  has(await failuresFor('windows-credentials', { ...goodWindows, 'credentials.json': '{}' }, WINDOWS, { requireNvrtc: true }), /unexpected entry/),
  'a staged credential file must fail even though it matches no junk pattern',
)
assert.ok(
  has(await failuresFor('windows-extra-exe', { ...goodWindows, 'totally-unexpected.exe': 'PE' }, WINDOWS, { requireNvrtc: true }), /unexpected entry/),
  'an unexpected executable must fail',
)
assert.ok(
  has(await failuresFor('windows-nested-dir', { ...goodWindows, 'extra/inner.txt': 'x' }, WINDOWS, { requireNvrtc: true }), /unexpected entry/),
  'an arbitrary nested directory must fail',
)
assert.ok(
  has(await failuresFor('linux-extra-doc', { ...goodLinux, 'NOTES.md': 'notes' }, LINUX, { requireNvrtc: true }), /unexpected entry/),
  'even an innocuous extra document must fail: the payload is a closed set',
)
// NVRTC names are constrained, not merely "contains nvrtc".
assert.ok(
  has(await failuresFor('windows-odd-nvrtc', { ...goodWindows, 'nvrtc64_evil.dll': 'x' }, WINDOWS, { requireNvrtc: true }), /unexpected entry/),
  'an nvrtc-shaped name that does not match the versioned pattern must fail',
)
assert.ok(
  has(await failuresFor('linux-odd-nvrtc', { ...goodLinux, 'libnvrtc.so': 'x' }, LINUX, { requireNvrtc: true }), /unexpected entry/),
  'an unversioned libnvrtc.so must fail the constrained pattern',
)
assert.deepEqual(
  checkManifest(
    [
      entry('camelid'),
      entry('README.md'),
      entry('LICENSE'),
      entry('THIRD_PARTY_NOTICES.md'),
      entry('libnvrtc.so.12.9.86'),
      entry('libnvrtc-builtins.so.12.9'),
      link('libnvrtc.so.12', 'libnvrtc.so.12.9.86'),
    ],
    LINUX,
    { requireNvrtc: true },
  ),
  [],
  'the real staged NVRTC filenames must all satisfy the allowlist',
)

// ---------------------------------------------------------------------------
// manifest — executable bit (POSIX only)
// ---------------------------------------------------------------------------
if (process.platform !== 'win32') {
  const root = await makeTree('linux-nonexec', goodLinux)
  await chmod(join(root, 'camelid'), 0o644)
  assert.ok(
    has(checkManifest(await readTree(root), LINUX, { requireNvrtc: true, checkExecutableBit: true }), /is not executable/),
    'a tarball whose binary lost +x extracts fine and then cannot be run',
  )
  await chmod(join(root, 'camelid'), 0o755)
  assert.deepEqual(
    checkManifest(await readTree(root), LINUX, { requireNvrtc: true, checkExecutableBit: true }).filter((f) => /not executable/.test(f)),
    [],
    'a +x binary must pass the mode check',
  )
} else {
  console.log('test-check-release-artifact: executable-bit case skipped (Windows has no mode bits)')
}

await rm(workspace, { recursive: true, force: true })
console.log('test-check-release-artifact: all checks passed')
