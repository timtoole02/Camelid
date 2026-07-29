#!/usr/bin/env node
// check-release-artifact.mjs — static contract check for a packaged release
// artifact, run BETWEEN packaging and publishing.
//
// Why this exists. release.yml already guards the STAGING DIRECTORY:
// scripts/package-linux-cuda.sh fails on a dangling symlink and a duplicated
// payload (the v0.4.3 regression). Those guards are good, but three things are
// still unverified:
//
// (Its soname guard is weaker than it reads: `compgen -G "$dist/libnvrtc.so.*"`
// at package-linux-cuda.sh:72 also matches the three-component real file, so it
// passes on a stage that lost the `libnvrtc.so.<major>` link. The compiler-role
// check below asserts the soname for real, on the extracted archive.)
//
//   1. They run before `tar -czf` / `Compress-Archive`. Nothing has ever looked
//      inside the archive users actually download.
//   2. They live INSIDE a conditional packaging step
//      (`if [ "$label" = "linux-x86_64" ]`). A renamed matrix label silently
//      skips them and still ships. A check that runs on the EXTRACTED archive
//      cannot be skipped by a packaging branch that never ran.
//   3. The `.sha256` sidecar is written once on the build runner and never
//      re-verified after the artifact round-trips through
//      upload-artifact / download-artifact into the publish job.
//
// This file is deliberately execution-free — booting the binary is
// scripts/smoke-release-artifact.mjs. Keeping them apart lets the whole
// contract here stay a pure function of a directory listing, which is what
// makes it exhaustively unit-testable in scripts/test-check-release-artifact.mjs.
//
// One environmental requirement: an artifact must be extracted on a filesystem
// that preserves what it ships. The Linux tarball carries an NVRTC symlink, so
// extracting it somewhere symlinks collapse into copies (Windows without
// developer mode) makes the duplicate-payload guard fire on a packaging job
// that was actually correct. release.yml always verifies each artifact on the
// runner that built it, which is the platform it ships to.
//
// Usage:
//   node scripts/check-release-artifact.mjs --dir <extracted-dir> --label <label>
//        [--archive <file>] [--checksum <file.sha256>]
//        [--require-nvrtc] [--out <receipt.json>]
//
// Labels: macos-arm64 | linux-x86_64 | windows-x64 | desktop-windows-x64
// Exit 0 = every check passed. Exit 1 = one or more FAIL lines were printed.
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { lstat, readdir, readlink, writeFile, readFile } from 'node:fs/promises'
import { basename, join, posix, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

// ---------------------------------------------------------------------------
// Artifact specifications
//
// One declarative row per shipped artifact, transcribed from release.yml. The
// `layout` field is the subtle one and the most likely source of a wrong check:
//
//   nested — `tar -czf "$dist.tar.gz" "$dist"` archives the DIRECTORY, so the
//            extracted tree is `<dest>/camelid-<label>/...`.
//   flat   — `Compress-Archive -Path "$dist/*"` archives the CONTENTS, so the
//            extracted tree is `<dest>/...` with no wrapper directory.
//
// Asserting the expected layout per label means a packaging change that flips
// one into the other is caught here rather than by a user with a broken path.
// ---------------------------------------------------------------------------

/** NVRTC redistributable shapes, per platform family. */
const NVRTC = {
  unix: {
    // SELECTOR vs VALIDATOR — these are deliberately different patterns, and
    // collapsing them into one breaks the check in one direction or the other.
    //
    // `compilerSoname` is the name cudarc actually dlopen's: it builds a fixed
    // candidate list (`libnvrtc.so`, `libnvrtc.so.<major>`, ...) and never asks
    // for a three-component `libnvrtc.so.<major>.<minor>.<patch>`. So the
    // ARTIFACT must contain the soname itself; that is what we search for.
    //
    // `compiler` is the looser family test applied to what the soname RESOLVES
    // to, which legitimately IS the three-component real file. Anchoring the
    // selector and reusing it as the validator would reject every correctly
    // packaged tarball; leaving the validator loose and reusing it as the
    // selector passes a tarball that lost the soname entirely.
    compilerSoname: /^libnvrtc\.so\.\d+$/,
    compiler: /^libnvrtc\.so\.\d+/,
    builtins: /^libnvrtc-builtins\.so/,
    family: /^libnvrtc.*\.so/,
    // The alternate JIT backend cudarc never asks for; ~90 MB of dead weight
    // that package-linux-cuda.sh deliberately skips.
    forbidden: /\.alt\.so/,
  },
  windows: {
    // Same split as unix. On Windows the loadable name is already fully
    // versioned (nvrtc64_120_0.dll) and is a real file rather than a symlink,
    // so selector and validator coincide in practice — but keeping the shapes
    // explicit stops `nvrtc64_120_0.alt.dll` from satisfying the role.
    compilerSoname: /^nvrtc64_\d+_\d+\.dll$/i,
    compiler: /^nvrtc64_.*\.dll$/i,
    builtins: /^nvrtc-builtins64_.*\.dll$/i,
    family: /^nvrtc.*\.dll$/i,
    forbidden: /\.alt\.dll$/i,
  },
}

/** Build outputs and source-tree leakage that must never reach a user download. */
const JUNK_PATTERNS = [
  { re: /\.pdb$/i, why: 'debug symbol database' },
  { re: /\.(d|rlib|rmeta)$/i, why: 'cargo build intermediate' },
  { re: /\.(exp|ilk)$/i, why: 'linker intermediate' },
  { re: /^\.git(\/|$)/, why: 'version control metadata' },
  { re: /(^|\/)node_modules(\/|$)/, why: 'javascript dependency tree' },
  { re: /(^|\/)target(\/|$)/, why: 'cargo target directory' },
  { re: /\.(env|pem|key)$/i, why: 'credential-shaped file' },
]

const COMMON_DOCS = ['README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md']

/**
 * Allowlist patterns shared by every artifact. Anchored on purpose: the payload
 * is a CLOSED set, so anything not named here fails rather than riding along.
 */
const DOC_PATTERNS = [/^README\.md$/, /^LICENSE$/, /^THIRD_PARTY_NOTICES\.md$/]

/**
 * Constrained NVRTC filename shapes. `libnvrtc.so.12`, `libnvrtc.so.12.9.86`
 * and `libnvrtc-builtins.so.12.9` match; `libnvrtc.alt.so.12` and anything
 * merely containing "nvrtc" do not.
 */
const NVRTC_UNIX_PATTERNS = [/^libnvrtc\.so(\.\d+)+$/, /^libnvrtc-builtins\.so(\.\d+)+$/]
const NVRTC_WINDOWS_PATTERNS = [/^nvrtc64_\d+_\d+\.dll$/i, /^nvrtc-builtins64_\d+\.dll$/i]

/** Depth bound for symlink resolution; real payloads use one hop. */
const MAX_SYMLINK_HOPS = 8

const ARTIFACT_SPECS = {
  'macos-arm64': {
    archive: 'camelid-macos-arm64.tar.gz',
    layout: 'nested',
    payloadRoot: 'camelid-macos-arm64',
    binary: 'camelid',
    executable: true,
    required: ['camelid', ...COMMON_DOCS],
    allowed: [/^camelid$/, ...DOC_PATTERNS],
    nvrtc: null,
    forbidden: [],
  },
  'linux-x86_64': {
    archive: 'camelid-linux-x86_64.tar.gz',
    layout: 'nested',
    payloadRoot: 'camelid-linux-x86_64',
    binary: 'camelid',
    executable: true,
    required: ['camelid', ...COMMON_DOCS],
    allowed: [/^camelid$/, ...DOC_PATTERNS, ...NVRTC_UNIX_PATTERNS],
    nvrtc: NVRTC.unix,
    forbidden: [],
  },
  'windows-x64': {
    archive: 'camelid-windows-x64.zip',
    layout: 'flat',
    payloadRoot: null,
    binary: 'camelid.exe',
    executable: false,
    required: ['camelid.exe', ...COMMON_DOCS],
    allowed: [/^camelid\.exe$/, ...DOC_PATTERNS, ...NVRTC_WINDOWS_PATTERNS],
    nvrtc: NVRTC.windows,
    forbidden: [],
  },
  'desktop-windows-x64': {
    archive: 'camelid-desktop-windows-x64.zip',
    layout: 'flat',
    payloadRoot: null,
    binary: 'camelid-desktop.exe',
    executable: false,
    // The portable layout is desktop exe + sidecar + NVRTC beside each other,
    // which is what `beside-exe` sidecar resolution depends on.
    required: ['camelid-desktop.exe', 'camelid.exe', 'DESKTOP_README.md', ...COMMON_DOCS],
    allowed: [/^camelid-desktop\.exe$/, /^camelid\.exe$/, /^DESKTOP_README\.md$/, ...DOC_PATTERNS, ...NVRTC_WINDOWS_PATTERNS],
    nvrtc: NVRTC.windows,
    // The NSIS installer is `Move-Item`d OUT of the portable folder before it is
    // zipped, because it is published as its own artifact. If that move ever
    // regresses, the portable download silently doubles in size and ships an
    // installer inside a portable archive.
    forbidden: [{ re: /-setup\.exe$/i, why: 'NSIS installer must be moved out of the portable zip' }],
  },
}

const LABELS = Object.keys(ARTIFACT_SPECS)

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested directly)
// ---------------------------------------------------------------------------

/**
 * Parse a `.sha256` sidecar.
 *
 * Two producers write these and they do NOT agree on trailing bytes:
 *   `shasum -a 256 f > f.sha256`            -> "<hash>  <name>\n"
 *   PowerShell `Out-File -NoNewline`        -> "<hash>  <name>"   (no newline)
 * Both use two spaces, but tolerate one-or-more, the `*` binary marker that
 * coreutils emits in binary mode, CRLF, and a UTF-8 BOM so a future packaging
 * tweak does not fail for a cosmetic reason.
 *
 * EXACTLY ONE record is required. A sidecar carrying a valid first line
 * followed by anything else is rejected: `shasum -c` would consume every
 * record, so accepting only the first would verify something different from
 * what a user's own check does.
 *
 * @returns {{hash: string, name: string}}
 */
export function parseChecksumFile(text) {
  if (typeof text !== 'string') throw new TypeError('checksum content must be a string')
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (!lines.length) throw new Error('checksum file is empty')
  if (lines.length > 1) {
    throw new Error(`checksum file must contain exactly one record, found ${lines.length}`)
  }
  const match = /^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$/.exec(lines[0].trim())
  if (!match) throw new Error(`checksum line is not "<64-hex>  <name>": ${JSON.stringify(lines[0].trim())}`)
  return { hash: match[1].toLowerCase(), name: match[2] }
}

/** Normalize a version string for tag comparison: drop a leading `v`, trim. */
export function normalizeVersion(value) {
  return String(value ?? '').trim().replace(/^v/i, '')
}

/**
 * Compare a binary's self-reported version to the release tag.
 *
 * `VERSION` (src/main.rs) is `CAMELID_GIT_DESCRIBE` when git metadata is
 * available, else `CARGO_PKG_VERSION`. At a clean tagged checkout git describe
 * yields exactly the tag, so an exact match after normalization is the correct
 * assertion. Anything else — a `-N-gsha` suffix, `-dirty`, or a stale
 * Cargo.toml that was never bumped — means the published tag and the shipped
 * bytes disagree, which is precisely what this guard exists to catch.
 *
 * @returns {string|null} a failure message, or null when they agree
 */
export function compareVersionToTag(reported, tag) {
  const got = normalizeVersion(reported)
  const want = normalizeVersion(tag)
  if (!got) return `binary reported an empty version (expected ${want})`
  if (got === want) return null
  if (got.startsWith(`${want}-`)) {
    return `binary version ${got} is ahead of tag ${want} (git describe suffix means the tag is not at this commit)`
  }
  return `binary version ${got} does not match release tag ${want}`
}

/**
 * Extract the version token from `camelid --version` output.
 * clap prints "<bin-name> <version>"; be liberal about the binary name so a
 * rename does not break the parse.
 */
export function parseVersionOutput(text) {
  const line = String(text ?? '').split(/\r?\n/).find((l) => l.trim().length > 0)
  if (!line) return null
  const match = /^\S+\s+(\S+)/.exec(line.trim())
  return match ? match[1] : null
}

/**
 * Decide which subdirectory of the extraction destination holds the payload,
 * and verify the archive layout matches the spec.
 *
 * @param {string[]} topLevelNames entries directly under the extraction dir
 * @returns {{root: string|null, failures: string[]}} `root` is the payload
 *   subdirectory name, or null when the payload is the extraction dir itself
 */
export function resolvePayloadRoot(topLevelNames, spec) {
  const failures = []
  const names = [...topLevelNames].sort()
  if (spec.layout === 'nested') {
    if (names.length !== 1 || names[0] !== spec.payloadRoot) {
      failures.push(
        `layout: expected exactly one top-level directory "${spec.payloadRoot}/" ` +
          `(tar archives the directory), found [${names.join(', ')}]`,
      )
      // Recover when the expected root is present alongside strays so the rest
      // of the manifest still gets checked and reported in one run.
      return { root: names.includes(spec.payloadRoot) ? spec.payloadRoot : null, failures }
    }
    return { root: spec.payloadRoot, failures }
  }
  // flat
  if (names.length === 1 && names[0] === basename(spec.archive, '.zip')) {
    failures.push(
      `layout: expected archive contents at the root (Compress-Archive -Path "$dist/*"), ` +
        `but found a single wrapper directory "${names[0]}/"`,
    )
    return { root: names[0], failures }
  }
  return { root: null, failures }
}

/**
 * Follow a symlink chain to the regular file it ultimately names.
 *
 * dlopen resolves links, so verifying only that a link's immediate target
 * exists is not enough: `libnvrtc.so.12 -> libnvrtc-builtins.so.12.9` points
 * at a real staged file and still cannot serve as the compiler soname. Cycles
 * must terminate too, or a self-referential link looks like a valid target of
 * itself.
 *
 * @param {Map<string, object>} byPath payload entries keyed by POSIX path
 * @param {string} startPath entry to resolve
 * @returns {{kind: 'resolved'|'dangling'|'escape'|'cycle'|'too-deep', path: string,
 *   entry?: object, target?: string}}
 */
export function resolveSymlinkChain(byPath, startPath, maxHops = MAX_SYMLINK_HOPS) {
  const seen = new Set()
  let current = startPath
  for (let hop = 0; hop <= maxHops; hop += 1) {
    if (seen.has(current)) return { kind: 'cycle', path: current }
    seen.add(current)
    const entry = byPath.get(current)
    if (!entry) return { kind: 'dangling', path: current }
    if (entry.kind !== 'symlink') return { kind: 'resolved', path: current, entry }
    const target = entry.linkTarget ?? ''
    if (target.startsWith('/')) return { kind: 'escape', path: current, target }
    const next = posix.normalize(posix.join(posix.dirname(current), target))
    if (next === '..' || next.startsWith('../')) return { kind: 'escape', path: current, target }
    current = next
  }
  return { kind: 'too-deep', path: startPath }
}

/**
 * Require at least one entry named like `role` to RESOLVE to a regular file
 * that is also named like `role`.
 *
 * `selector` is what the caller searched for and is reported when nothing
 * matched; `pattern` validates the far end of the symlink chain. They differ
 * for the NVRTC compiler — see the NVRTC table.
 *
 * @returns {string[]} failure messages
 */
function checkNvrtcRole(candidates, byPath, pattern, role, consequence, selector = pattern) {
  if (!candidates.length) {
    return [`no NVRTC ${role} library matching ${selector} — ${consequence}`]
  }
  const diagnostics = []
  for (const candidate of candidates) {
    const resolution = resolveSymlinkChain(byPath, candidate.path)
    if (resolution.kind !== 'resolved') {
      diagnostics.push(`${candidate.path} does not resolve (${resolution.kind}${resolution.target ? ` -> ${resolution.target}` : ''})`)
      continue
    }
    if (resolution.entry.kind !== 'file') {
      diagnostics.push(`${candidate.path} resolves to a ${resolution.entry.kind} (${resolution.path})`)
      continue
    }
    if (!pattern.test(basename(resolution.path))) {
      diagnostics.push(`${candidate.path} resolves to ${resolution.path}, which is not an NVRTC ${role} library`)
      continue
    }
    return []
  }
  return [`no usable NVRTC ${role} library — ${consequence}. Checked: ${diagnostics.join('; ')}`]
}

/**
 * The whole manifest contract, as a pure function of a normalized entry list.
 *
 * @param {Array<{path: string, kind: 'file'|'dir'|'symlink', size: number,
 *   mode: number, linkTarget?: string, sha256?: string}>} entries
 *   paths are POSIX, relative to the payload root
 * @param {object} spec one of ARTIFACT_SPECS
 * @param {{requireNvrtc?: boolean, checkExecutableBit?: boolean}} options
 * @returns {string[]} failure messages (empty = pass)
 */
export function checkManifest(entries, spec, options = {}) {
  const { requireNvrtc = false, checkExecutableBit = false } = options
  const failures = []
  const files = entries.filter((e) => e.kind === 'file')
  const byPath = new Map(entries.map((e) => [e.path, e]))

  // --- required payload -----------------------------------------------------
  for (const required of spec.required) {
    const entry = byPath.get(required)
    if (!entry) {
      failures.push(`missing required entry: ${required}`)
      continue
    }
    if (entry.kind !== 'file') {
      failures.push(`required entry ${required} is a ${entry.kind}, expected a file`)
      continue
    }
    if (entry.size === 0) failures.push(`required entry ${required} is zero bytes`)
  }

  // --- the binary must be launchable ---------------------------------------
  const binary = byPath.get(spec.binary)
  if (binary && binary.kind === 'file' && spec.executable && checkExecutableBit) {
    // A tarball whose binary lost its +x bit extracts fine and then cannot be
    // run. Only meaningful on POSIX hosts; Windows has no mode bits to lose.
    if ((binary.mode & 0o111) === 0) {
      failures.push(`${spec.binary} is not executable (mode ${(binary.mode & 0o777).toString(8)})`)
    }
  }

  // --- NVRTC redistributables ----------------------------------------------
  if (spec.nvrtc) {
    const topLevel = entries.filter((e) => !e.path.includes('/') && (e.kind === 'file' || e.kind === 'symlink'))
    if (requireNvrtc) {
      // A NAME is not enough. cudarc dlopen's the soname, and dlopen follows
      // links, so the chain has to end at a real library of the right family.
      failures.push(
        ...checkNvrtcRole(
          // Search for the SONAME, not the family: a tarball holding only
          // libnvrtc.so.<major>.<minor>.<patch> cannot be dlopen'd by cudarc.
          topLevel.filter((e) => spec.nvrtc.compilerSoname.test(basename(e.path))),
          byPath,
          spec.nvrtc.compiler,
          'compiler',
          'the GPU path would silently fall back to CPU on a driver-only host',
          spec.nvrtc.compilerSoname,
        ),
      )
      failures.push(
        ...checkNvrtcRole(
          topLevel.filter((e) => spec.nvrtc.builtins.test(basename(e.path))),
          byPath,
          spec.nvrtc.builtins,
          'builtins',
          'the NVRTC pair is incomplete',
        ),
      )
    }
    for (const entry of topLevel) {
      if (spec.nvrtc.forbidden.test(entry.path)) {
        failures.push(`${entry.path} is the alternate JIT backend cudarc never loads and must not be shipped`)
      }
    }
  }

  // --- duplicated payload (the v0.4.3 regression, checked post-archive) -----
  // Scoped to the NVRTC family, mirroring package-linux-cuda.sh's `libnvrtc*`
  // glob: those are the only entries where duplication is a known failure mode,
  // and narrowing the scope means two benign identical files (say, a doc
  // copied twice) can never block a release.
  //
  // Within that scope this is stronger than upstream, which compares SIZES.
  // Comparing content hashes cannot false-positive on two genuinely different
  // libraries that happen to match in size, and still catches a dereferenced
  // symlink shipping the same 106 MB twice.
  const hashed = spec.nvrtc
    ? files.filter((e) => typeof e.sha256 === 'string' && e.sha256.length > 0 && spec.nvrtc.family.test(basename(e.path)))
    : []
  const byHash = new Map()
  for (const entry of hashed) {
    if (!byHash.has(entry.sha256)) byHash.set(entry.sha256, [])
    byHash.get(entry.sha256).push(entry)
  }
  for (const [hash, group] of byHash) {
    if (group.length < 2) continue
    const paths = group.map((e) => e.path).sort()
    failures.push(
      `duplicate payload: ${paths.join(' and ')} are byte-identical (${group[0].size} bytes, sha256 ${hash.slice(0, 12)}…) — ` +
        'symlinks must be preserved (cp -P), not dereferenced. If this artifact was extracted on a ' +
        'filesystem that cannot represent symlinks (e.g. Windows without developer mode), extract and ' +
        'verify it on the platform it ships to instead.',
    )
  }

  // --- symlink integrity ----------------------------------------------------
  // A link that resolves nowhere is worse than no link: dlopen fails and the
  // GPU silently never engages. A link that never terminates is worse still,
  // because a naive one-hop check reads a cycle as a satisfied target.
  for (const entry of entries) {
    if (entry.kind !== 'symlink') continue
    const resolution = resolveSymlinkChain(byPath, entry.path)
    const target = entry.linkTarget ?? ''
    if (resolution.kind === 'escape') {
      failures.push(`symlink ${entry.path} -> ${resolution.target ?? target} escapes the payload`)
    } else if (resolution.kind === 'dangling') {
      failures.push(`symlink ${entry.path} -> ${target} is dangling inside the archive (no entry at ${resolution.path})`)
    } else if (resolution.kind === 'cycle') {
      failures.push(`symlink ${entry.path} -> ${target} never terminates (cycle at ${resolution.path})`)
    } else if (resolution.kind === 'too-deep') {
      failures.push(`symlink ${entry.path} -> ${target} exceeds ${MAX_SYMLINK_HOPS} hops`)
    }
  }

  // --- closed payload contract ---------------------------------------------
  // Required-plus-junk-patterns is an OPEN set: it accepts anything nobody
  // thought to forbid. The shipped payloads are small and fully enumerable, so
  // the allowlist is the honest shape: an accidental future glob, a staged
  // credential, or a stray build output fails here instead of reaching a
  // public download.
  for (const entry of entries) {
    if (spec.allowed.some((re) => re.test(entry.path))) continue
    failures.push(`unexpected entry in the artifact: ${entry.path} (${entry.kind}) is not part of the ${spec.archive} payload`)
  }

  // --- junk and label-specific forbidden entries ---------------------------
  for (const entry of entries) {
    for (const { re, why } of JUNK_PATTERNS) {
      if (re.test(entry.path)) failures.push(`unexpected ${why} in the artifact: ${entry.path}`)
    }
    for (const { re, why } of spec.forbidden) {
      if (re.test(entry.path)) failures.push(`${entry.path} must not be present: ${why}`)
    }
  }

  return failures
}

// ---------------------------------------------------------------------------
// Filesystem shell
// ---------------------------------------------------------------------------

async function sha256File(path) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

/**
 * Walk `root` into the normalized entry shape checkManifest expects.
 * Symlinks are recorded, never followed — following them would hide exactly
 * the duplication and escape cases we are looking for.
 */
export async function readTree(root, { hashFiles = true } = {}) {
  const entries = []
  const walk = async (dir, prefix) => {
    const dirents = await readdir(dir, { withFileTypes: true })
    for (const dirent of dirents.sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const absolute = join(dir, dirent.name)
      const relative = prefix ? posix.join(prefix, dirent.name) : dirent.name
      const stats = await lstat(absolute)
      if (stats.isSymbolicLink()) {
        entries.push({
          path: relative,
          kind: 'symlink',
          size: 0,
          mode: stats.mode,
          linkTarget: (await readlink(absolute)).split('\\').join('/'),
        })
        continue
      }
      if (stats.isDirectory()) {
        entries.push({ path: relative, kind: 'dir', size: 0, mode: stats.mode })
        await walk(absolute, relative)
        continue
      }
      entries.push({
        path: relative,
        kind: 'file',
        size: stats.size,
        mode: stats.mode,
        sha256: hashFiles ? await sha256File(absolute) : undefined,
      })
    }
  }
  await walk(root, '')
  return entries
}

function parseArgs(argv) {
  const args = { requireNvrtc: false }
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    const next = () => {
      const value = argv[i + 1]
      if (value === undefined || value.startsWith('--')) throw new Error(`${arg} requires a value`)
      i += 1
      return value
    }
    switch (arg) {
      case '--dir': args.dir = next(); break
      case '--label': args.label = next(); break
      case '--archive': args.archive = next(); break
      case '--checksum': args.checksum = next(); break
      case '--out': args.out = next(); break
      case '--require-nvrtc': args.requireNvrtc = true; break
      case '--help': case '-h': args.help = true; break
      default: throw new Error(`unknown argument ${arg}`)
    }
  }
  return args
}

const USAGE = `Usage:
  node scripts/check-release-artifact.mjs --dir <extracted-dir> --label <label> [options]

Options:
  --dir <path>        directory the release archive was extracted into (required)
  --label <label>     ${LABELS.join(' | ')}
  --archive <path>    archive file to checksum-verify
  --checksum <path>   .sha256 sidecar to verify --archive against
  --require-nvrtc     fail when the NVRTC redistributables are absent
  --out <path>        write a JSON receipt
`

async function main() {
  let args
  try {
    args = parseArgs(process.argv.slice(2))
  } catch (error) {
    console.error(`check-release-artifact: ${error.message}\n\n${USAGE}`)
    process.exit(2)
  }
  if (args.help) {
    console.log(USAGE)
    return
  }

  const failures = []
  const fail = (message) => failures.push(message)

  if (!args.dir) fail('--dir is required')
  if (!args.label) fail('--label is required')
  if (args.label && !ARTIFACT_SPECS[args.label]) fail(`unknown --label ${args.label} (expected one of ${LABELS.join(', ')})`)
  if (args.archive && !args.checksum) fail('--archive requires --checksum')
  if (args.checksum && !args.archive) fail('--checksum requires --archive')
  if (failures.length) {
    console.error(`check-release-artifact FAILED (${failures.length}):`)
    for (const failure of failures) console.error(`  FAIL: ${failure}`)
    console.error(`\n${USAGE}`)
    process.exit(2)
  }

  const spec = ARTIFACT_SPECS[args.label]
  const extractDir = resolve(args.dir)
  console.log(`check-release-artifact: label=${args.label} layout=${spec.layout}`)

  // 1. checksum round-trip -------------------------------------------------
  let archiveHash = null
  if (args.archive) {
    const archivePath = resolve(args.archive)
    const archiveName = basename(archivePath)
    if (archiveName !== spec.archive) {
      fail(`archive is named ${archiveName}, but label ${args.label} ships ${spec.archive}`)
    }
    let expected
    try {
      expected = parseChecksumFile(await readFile(resolve(args.checksum), 'utf8'))
    } catch (error) {
      fail(`checksum sidecar unreadable: ${error.message}`)
    }
    if (expected) {
      archiveHash = await sha256File(archivePath)
      if (expected.name !== archiveName) {
        fail(`checksum sidecar names ${expected.name}, but the archive is ${archiveName}`)
      }
      if (expected.hash !== archiveHash) {
        fail(`sha256 mismatch for ${archiveName}: sidecar ${expected.hash}, actual ${archiveHash}`)
      } else {
        console.log(`  sha256 ok: ${archiveHash}`)
      }
    }
  }

  // 2. layout --------------------------------------------------------------
  let topLevel
  try {
    topLevel = (await readdir(extractDir, { withFileTypes: true })).map((d) => d.name)
  } catch (error) {
    console.error(`check-release-artifact FAILED: cannot read --dir ${args.dir}: ${error.message}`)
    process.exit(1)
  }
  const { root, failures: layoutFailures } = resolvePayloadRoot(topLevel, spec)
  layoutFailures.forEach(fail)

  // 3. manifest ------------------------------------------------------------
  const payloadDir = root ? join(extractDir, root) : extractDir
  const entries = await readTree(payloadDir)
  const manifestFailures = checkManifest(entries, spec, {
    requireNvrtc: args.requireNvrtc,
    // Mode bits only survive on POSIX filesystems; asserting them on Windows
    // would fail for a reason that has nothing to do with the artifact.
    checkExecutableBit: process.platform !== 'win32',
  })
  manifestFailures.forEach(fail)

  const fileCount = entries.filter((e) => e.kind === 'file').length
  const totalBytes = entries.reduce((sum, e) => sum + e.size, 0)
  console.log(`  payload: ${fileCount} file(s), ${(totalBytes / 1024 / 1024).toFixed(1)} MiB`)

  // 4. receipt -------------------------------------------------------------
  if (args.out) {
    const receipt = {
      schema: 'camelid.release-artifact-check/v1',
      checked_at: new Date().toISOString(),
      label: args.label,
      // Basenames and payload-relative paths only: a receipt must never carry
      // a runner or developer absolute path into an uploaded artifact.
      archive: args.archive ? basename(args.archive) : null,
      archive_sha256: archiveHash,
      layout: spec.layout,
      payload_root: root,
      require_nvrtc: args.requireNvrtc,
      entries: entries.map((e) => ({
        path: e.path,
        kind: e.kind,
        size: e.size,
        sha256: e.sha256 ?? null,
        link_target: e.linkTarget ?? null,
      })),
      failures,
      passed: failures.length === 0,
    }
    await writeFile(resolve(args.out), `${JSON.stringify(receipt, null, 2)}\n`, 'utf8')
    console.log(`  receipt: ${basename(args.out)}`)
  }

  if (failures.length) {
    console.error(`\ncheck-release-artifact FAILED (${failures.length}):`)
    for (const failure of failures) console.error(`  FAIL: ${failure}`)
    process.exit(1)
  }
  console.log(`\ncheck-release-artifact passed: ${args.label} is publishable`)
}

export { ARTIFACT_SPECS, JUNK_PATTERNS, LABELS, NVRTC, sha256File }

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
