#!/usr/bin/env node
// verify-release-checksums.mjs — the last gate before artifacts become public.
//
// Why this is not inline shell. The build-time checker validates a sidecar's
// embedded filename, but the publish step originally took only the first
// whitespace field (`awk '{print $1; exit}'`). That accepts two things a user
// running `shasum -c` would not:
//
//   * a sidecar whose hash is right but whose embedded filename names some
//     other file, so the user's check verifies a different (or missing) path;
//   * a valid first record followed by anything at all, because `shasum -c`
//     consumes every record while the publish gate read one.
//
// Reusing parseChecksumFile from check-release-artifact.mjs means the two
// gates cannot drift, and the tolerant-format handling (BOM, CRLF, the
// coreutils `*` marker, PowerShell's missing trailing newline) is written once.
//
// It also closes the staging directory: only the expected archives and their
// sidecars may be present, so diagnostic receipts or a stray desktop artifact
// cannot be swept into `files: dist/*`.
//
// Usage:
//   node scripts/verify-release-checksums.mjs --dir dist \
//     --expect camelid-macos-arm64.tar.gz \
//     --expect camelid-linux-x86_64.tar.gz \
//     --expect camelid-windows-x64.zip
//
// Exit 0 = every expected archive is present, alone, and matches its sidecar.
import { readdir, readFile, stat } from 'node:fs/promises'
import { basename, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import { parseChecksumFile, sha256File } from './check-release-artifact.mjs'

/** Sidecar suffix produced by both packaging lanes. */
export const CHECKSUM_SUFFIX = '.sha256'

/**
 * Compare what is staged against what may be published.
 *
 * Pure so the set logic is unit-testable without a filesystem.
 *
 * @param {string[]} presentNames every file name directly in the staging dir
 * @param {string[]} expectedArchives archive names that must be published
 * @returns {string[]} failure messages
 */
export function checkStagedSet(presentNames, expectedArchives) {
  const failures = []
  const present = new Set(presentNames)
  const allowed = new Set()
  for (const archive of expectedArchives) {
    allowed.add(archive)
    allowed.add(`${archive}${CHECKSUM_SUFFIX}`)
  }

  for (const archive of expectedArchives) {
    if (!present.has(archive)) failures.push(`missing expected archive: ${archive}`)
    if (!present.has(`${archive}${CHECKSUM_SUFFIX}`)) failures.push(`missing checksum sidecar: ${archive}${CHECKSUM_SUFFIX}`)
  }
  for (const name of [...present].sort()) {
    if (!allowed.has(name)) failures.push(`refusing to publish unexpected file: ${name}`)
  }
  return failures
}

/**
 * Validate one sidecar against the archive it names.
 *
 * @param {string} archiveName expected file name
 * @param {string} sidecarText raw sidecar contents
 * @param {string} actualHash sha256 of the archive on disk
 * @returns {string[]} failure messages
 */
export function checkChecksumRecord(archiveName, sidecarText, actualHash) {
  let record
  try {
    record = parseChecksumFile(sidecarText)
  } catch (error) {
    return [`${archiveName}${CHECKSUM_SUFFIX}: ${error.message}`]
  }
  const failures = []
  if (record.name !== archiveName) {
    failures.push(
      `${archiveName}${CHECKSUM_SUFFIX} names ${JSON.stringify(record.name)}, not ${JSON.stringify(archiveName)} — ` +
        'a user running `shasum -c` would check a different file',
    )
  }
  if (record.hash !== actualHash) {
    failures.push(`${archiveName}: sha256 mismatch — sidecar ${record.hash}, actual ${actualHash}`)
  }
  return failures
}

function parseArgs(argv) {
  const args = { expect: [] }
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
      case '--expect': args.expect.push(next()); break
      case '--help': case '-h': args.help = true; break
      default: throw new Error(`unknown argument ${arg}`)
    }
  }
  return args
}

const USAGE = `Usage:
  node scripts/verify-release-checksums.mjs --dir <staging-dir> --expect <archive> [--expect <archive> ...]
`

async function main() {
  let args
  try {
    args = parseArgs(process.argv.slice(2))
  } catch (error) {
    console.error(`verify-release-checksums: ${error.message}\n\n${USAGE}`)
    process.exit(2)
  }
  if (args.help) {
    console.log(USAGE)
    return
  }
  if (!args.dir || !args.expect.length) {
    console.error(`verify-release-checksums: --dir and at least one --expect are required\n\n${USAGE}`)
    process.exit(2)
  }

  const dir = resolve(args.dir)
  const failures = []

  const dirents = await readdir(dir, { withFileTypes: true })
  for (const dirent of dirents) {
    if (!dirent.isFile()) failures.push(`refusing to publish non-file entry: ${dirent.name}`)
  }
  const names = dirents.filter((d) => d.isFile()).map((d) => d.name)
  failures.push(...checkStagedSet(names, args.expect))

  for (const archive of args.expect) {
    const archivePath = join(dir, archive)
    const sidecarPath = `${archivePath}${CHECKSUM_SUFFIX}`
    let sidecarText
    try {
      sidecarText = await readFile(sidecarPath, 'utf8')
    } catch {
      continue // already reported as missing by checkStagedSet
    }
    let size
    try {
      size = (await stat(archivePath)).size
    } catch {
      continue
    }
    if (size === 0) {
      failures.push(`${archive} is zero bytes`)
      continue
    }
    const actualHash = await sha256File(archivePath)
    const recordFailures = checkChecksumRecord(basename(archivePath), sidecarText, actualHash)
    failures.push(...recordFailures)
    if (!recordFailures.length) console.log(`  verified ${archive} (${(size / 1024 / 1024).toFixed(1)} MiB)`)
  }

  if (failures.length) {
    console.error(`\nverify-release-checksums FAILED (${failures.length}):`)
    for (const failure of failures) console.error(`  FAIL: ${failure}`)
    process.exit(1)
  }
  console.log(`\nverify-release-checksums passed: ${args.expect.length} archive(s) ready to publish`)
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
