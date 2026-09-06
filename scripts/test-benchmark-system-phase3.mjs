#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
const tests = [
  'tools/bench/system/test-agent-exec-trace.mjs',
  'tools/bench/system/test-native-contracts.mjs',
  'tools/bench/system/test-native-adapter.mjs',
  'tools/bench/system/test-native-bundle.mjs',
  'tools/bench/system/test-native-cli.mjs',
]

for (const test of tests) {
  console.log(`== ${test}`)
  const result = spawnSync(process.execPath, [resolve(root, test)], {
    cwd: root,
    stdio: 'inherit',
    windowsHide: true,
  })
  if (result.status !== 0) process.exit(result.status ?? 1)
}

console.log(`benchmark Phase 3 validation: PASS (${tests.length} tests)`)