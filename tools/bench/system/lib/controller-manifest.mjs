import { readdir, readFile } from 'node:fs/promises'
import { join, relative, sep } from 'node:path'

import { canonicalJson, sha256Bytes } from './digest.mjs'

export async function controllerManifest(systemRoot) {
  const files = await sourceFiles(systemRoot)
  const entries = []
  for (const path of files) {
    const bytes = await readFile(path)
    entries.push({
      path: relative(systemRoot, path).split(sep).join('/'),
      size_bytes: bytes.length,
      sha256: sha256Bytes(bytes),
    })
  }
  const manifest = {
    schema: 'camelid.benchmark.controller-manifest/v1',
    files: entries,
  }
  return {
    manifest,
    sha256: sha256Bytes(Buffer.from(canonicalJson(manifest), 'utf8')),
  }
}

async function sourceFiles(root) {
  const files = []
  await walk(root, files)
  return files.sort((left, right) => left.localeCompare(right))
}

async function walk(directory, files) {
  const entries = await readdir(directory, { withFileTypes: true })
  for (const entry of entries) {
    if (['fixtures', 'examples'].includes(entry.name)) continue
    const path = join(directory, entry.name)
    if (entry.isDirectory()) {
      await walk(path, files)
      continue
    }
    if (!entry.isFile()) continue
    const relativeName = entry.name.toLowerCase()
    if (relativeName.endsWith('.mjs') || relativeName.endsWith('.schema.json') || relativeName.endsWith('.sh')) files.push(path)
  }
}
