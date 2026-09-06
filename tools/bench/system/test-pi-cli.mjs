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
assert.match(usage.stdout, /pi-run/)
assert.match(usage.stdout, /--pi-archive <windows-archive>/)
assert.match(usage.stdout, /--linux-pi-dir <linux-dir>/)
assert.match(usage.stdout, /--context-window <tokens>/)

const missing = run(['pi-run'])
assert.equal(missing.status, 1)
assert.match(missing.stderr, /--workspace is required/)

const sameRoot = resolve(repositoryRoot, 'target/pi-same-root')
const refused = run([
  'pi-run',
  '--workspace', sameRoot,
  '--out', sameRoot,
])
assert.equal(refused.status, 1)
assert.match(refused.stderr, /--workspace and --out must be different paths/)

console.log('benchmark Phase 4 Pi CLI contract: PASS')

function run(args) {
  return spawnSync(process.execPath, [cli, ...args], {
    cwd: repositoryRoot,
    encoding: 'utf8',
    windowsHide: true,
  })
}