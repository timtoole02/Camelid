#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))
const repositoryRoot = resolve(systemRoot, '../../..')
const manifest = JSON.parse(await readFile(resolve(repositoryRoot, 'qa/benchmarks/agent/native-contracts-v1.json'), 'utf8'))
const expected = [
  'native_write_requires_policy',
  'native_exec_not_blanket_auto',
  'native_production_refuses_unattended',
  'native_path_escape_refused',
  'native_network_absent_by_default',
  'native_mcp_absent_by_default',
  'native_workspace_read_only',
  'native_step_cap_inconclusive',
  'native_repeat_break_inconclusive',
  'native_driver_error_failed',
  'native_cancel_discards_partial',
  'native_compaction_preserves_goal',
  'native_tool_output_untrusted',
  'native_mcp_round_trip',
  'native_subagent_cleanup',
  'native_outside_canary_unchanged',
  'native_edit_self_healing',
  'native_path_suggestion',
  'native_compaction_working_set',
  'native_search_path_filter',
  'native_verification_gate',
]

assert.equal(manifest.schema, 'camelid.benchmark.native-contracts/v1')
assert.deepEqual(manifest.contracts.map((contract) => contract.id), expected)
assert.equal(new Set(expected).size, expected.length)

for (const contract of manifest.contracts) {
  assert.ok(Array.isArray(contract.evidence) && contract.evidence.length > 0, contract.id)
  for (const evidence of contract.evidence) {
    assert.ok(['rust_test', 'js_test'].includes(evidence.kind), `${contract.id}: ${evidence.kind}`)
    assert.ok(Array.isArray(evidence.platforms) && evidence.platforms.length > 0, contract.id)
    const source = await readFile(resolve(repositoryRoot, evidence.file), 'utf8')
    if (evidence.kind === 'rust_test') {
      assert.match(source, new RegExp(`\\bfn\\s+${escapeRegExp(evidence.test)}\\s*\\(`), `${contract.id}: ${evidence.test}`)
    } else {
      assert.ok(source.includes(evidence.marker), `${contract.id}: ${evidence.marker}`)
    }
  }
}

console.log(`benchmark Phase 3 native contract inventory: PASS (${expected.length} contracts)`)

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}