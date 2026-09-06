#!/usr/bin/env node
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))
const repositoryRoot = resolve(systemRoot, '../../..')
const cli = resolve(systemRoot, 'cli.mjs')

const usage = run([])
assert.equal(usage.status, 0, usage.stderr)
assert.match(usage.stdout, /native-run/)
assert.match(usage.stdout, /--wsl-gpu <true\|false>/)
assert.match(usage.stdout, /--max-tokens-per-step <n>/)

const missing = run(['native-run'])
assert.equal(missing.status, 1)
assert.match(missing.stderr, /--workspace is required/)

const sameRoot = resolve(repositoryRoot, 'target/native-same-root')
const refused = run([
  'native-run',
  '--workspace', sameRoot,
  '--out', sameRoot,
])
assert.equal(refused.status, 1)
assert.match(refused.stderr, /--workspace and --out must be different paths/)

console.log('benchmark Phase 3 native CLI contract: PASS')

function run(args) {
  return spawnSync(process.execPath, [cli, ...args], {
    cwd: repositoryRoot,
    encoding: 'utf8',
    windowsHide: true,
  })
}