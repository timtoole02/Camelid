#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, extname, join, relative, resolve } from 'node:path'

const args = parseArgs(process.argv.slice(2))
const src = args.get('src')
const dst = args.get('dst')

if (!src || !dst) {
  console.error('usage: node scripts/publish-public-evidence-bundle.mjs --src <dir> --dst <dir>')
  process.exit(1)
}

const srcDir = resolve(src)
const dstDir = resolve(dst)

await rm(dstDir, { recursive: true, force: true })
await mkdir(dstDir, { recursive: true })
await copyTree(srcDir, dstDir)
await writeSha256Sums(dstDir)

console.log(`published=${dstDir}`)

async function copyTree(fromDir, toDir) {
  const entries = await readdir(fromDir, { withFileTypes: true })
  for (const entry of entries) {
    const srcPath = join(fromDir, entry.name)
    const dstPath = join(toDir, entry.name)
    if (entry.isDirectory()) {
      await mkdir(dstPath, { recursive: true })
      await copyTree(srcPath, dstPath)
      continue
    }
    if (entry.name === 'SHA256SUMS') continue
    await mkdir(dirname(dstPath), { recursive: true })
    const content = await readFile(srcPath)
    const ext = extname(entry.name).toLowerCase()
    if (ext === '.json') {
      const payload = JSON.parse(content.toString('utf8'))
      await writeFile(dstPath, `${JSON.stringify(sanitizeValue(payload), null, 2)}\n`, 'utf8')
      continue
    }
    if (isSanitizedTextExtension(ext)) {
      await writeFile(dstPath, sanitizeText(content.toString('utf8')), 'utf8')
      continue
    }
    await writeFile(dstPath, content)
  }
}

function sanitizeValue(value) {
  if (Array.isArray(value)) return value.map(sanitizeValue)
  if (!value || typeof value !== 'object') {
    return typeof value === 'string' ? sanitizeText(value) : value
  }
  return Object.fromEntries(Object.entries(value).map(([key, inner]) => [key, sanitizeValue(inner)]))
}

function sanitizeText(input) {
  return String(input)
    .replace(/\/Users\/[^/\n"']+\/\.openclaw\/workspace\/projects\/Camelid-v0\.1-evidence/g, '$CAMELID_WORKTREE')
    .replace(/\/Users\/[^/\n"']+\/\.openclaw\/workspace\/projects\/Camelid/g, '$CAMELID_PRIMARY_WORKTREE')
    .replace(/\/Users\/[^/\n"']+/g, '$HOME')
    .replace(/\/Volumes\/[^/\n"']+\/Camelid\/models\//g, '$CAMELID_MODEL_DIR/')
    .replace(/\/Volumes\/[^/\n"']+\/Camelid\/release-captain\/v0\.1-evidence\//g, '$CAMELID_RELEASE_BUILD_ROOT/')
    .replace(/\/Volumes\/[^/\n"']+/g, '$EXTERNAL_VOLUME')
    .replace(/\/home\/[^/]+\/work\/Camelid[^/]*\/target\//g, 'target/')
    .replace(/\/home\/[^/]+\/work\/Camelid[^/]*\/frontend\/scripts\/smoke\.mjs/g, 'frontend/scripts/smoke.mjs')
    .replace(/\/home\/[^/]+\/work\/Camelid[^/]*\/scripts\/summarize-generation-timings\.mjs/g, 'scripts/summarize-generation-timings.mjs')
    .replace(/\/home\/[^/]+\/work\/llama\.cpp\/build\/bin\//g, '$CAMELID_LLAMA_CPP_BIN/')
    .replace(/\/home\/[^/]+\/work\/Camelid[^\s"]*/g, '$CAMELID_WORKTREE')
    .replace(/\/home\/[^/]+\/\.nvm\/versions\/node\/[^/]+\/bin\/node/g, 'node')
    .replace(/\/home\/[^/]+\/models\//g, '$CAMELID_MODEL_DIR/')
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, 'canonical-private-ubuntu-validation-host')
    .replace(/\bip-(?:\d+-){3}\d+\b/g, 'canonical-private-ubuntu-validation-host')
    .replace(/[ \t]+$/gm, '')
}

function isSanitizedTextExtension(ext) {
  return ext === '.md' || ext === '.txt' || ext === '.log' || ext === '.tsv' || ext === '.status'
}

async function writeSha256Sums(rootDir) {
  const files = []
  await collectFiles(rootDir, rootDir, files)
  const lines = []
  for (const file of files.sort()) {
    if (file === 'SHA256SUMS') continue
    const fullPath = join(rootDir, file)
    const hash = createHash('sha256').update(await readFile(fullPath)).digest('hex')
    lines.push(`${hash}  ${file}`)
  }
  await writeFile(join(rootDir, 'SHA256SUMS'), `${lines.join('\n')}\n`, 'utf8')
}

async function collectFiles(rootDir, currentDir, output) {
  const entries = await readdir(currentDir, { withFileTypes: true })
  for (const entry of entries) {
    const fullPath = join(currentDir, entry.name)
    if (entry.isDirectory()) {
      await collectFiles(rootDir, fullPath, output)
      continue
    }
    const info = await stat(fullPath)
    if (!info.isFile()) continue
    output.push(relative(rootDir, fullPath))
  }
}

function parseArgs(argv) {
  const parsed = new Map()
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (!arg.startsWith('--')) continue
    const [key, inline] = arg.slice(2).split('=', 2)
    const next = argv[i + 1]
    const value = inline ?? (next && !next.startsWith('--') ? argv[++i] : 'true')
    parsed.set(key, value)
  }
  return parsed
}
