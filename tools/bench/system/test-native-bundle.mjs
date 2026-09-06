#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { verifyBundleChecksums, writeNativeAgentBundle } from './bundle.mjs'

const root = await mkdtemp(join(tmpdir(), 'camelid-native-bundle-'))
const outputDir = join(root, 'bundle')

try {
  const result = fixtureResult()
  const bundle = await writeNativeAgentBundle({ outputDir, result }, {
    generatedUtc: '2026-08-23T00:00:00Z',
  })
  assert.equal(bundle.manifest.state, 'COMPLETE')
  assert.equal(bundle.manifest.workspace_included, false)
  assert.equal(bundle.manifest.boundary, 'wsl-bwrap')
  assert.equal(bundle.manifest.gpu_enabled, false)
  assert.equal(bundle.manifest.checkpoints_enabled, false)
  assert.equal(bundle.manifest.tool_profile, 'benchmark_shared')
  assert.equal(bundle.manifest.max_output_tokens_per_step, 1024)
  assert.equal((await verifyBundleChecksums(outputDir)).ok, true)
  assert.equal(await exists(join(outputDir, 'workspace')), false)
  assert.match(await readFile(join(outputDir, 'summary.md'), 'utf8'), /PASS_COMPARABLE/)
  await assert.rejects(writeNativeAgentBundle({ outputDir, result }), /not empty/)

  const unlistedPath = join(outputDir, 'unlisted.txt')
  await writeFile(unlistedPath, 'not sealed\n')
  const unlisted = await verifyBundleChecksums(outputDir)
  assert.equal(unlisted.ok, false)
  assert.ok(unlisted.failures.some((failure) => failure.includes('unlisted.txt: not listed')))
  await rm(unlistedPath)

  await writeFile(join(outputDir, 'attempt.json'), '{}\n')
  const tampered = await verifyBundleChecksums(outputDir)
  assert.equal(tampered.ok, false)
  assert.ok(tampered.failures.some((failure) => failure.includes('attempt.json')))
} finally {
  await rm(root, { recursive: true, force: true })
}

console.log('benchmark Phase 3 native bundle and checksums: PASS')

function fixtureResult() {
  const trace = {
    schema: 'camelid.agent-exec-trace/v1',
    terminal: { reason: 'answered', outcome: 'completed', exit_code: 0, wall_ms: 25 },
    summary: { model_steps: 1, tool_calls: 0, tool_errors: 0, compactions: 0, model_ms: 20, output_tokens: 7 },
    steps: [{
      index: 0,
      model_ms: 20,
      ttft_ms: 3,
      output_tokens: 7,
      context: {
        prompt_tokens: 100,
        generation_tokens: 32,
        budget_tokens: 4096,
        system_tokens_estimate: 10,
        tool_definition_tokens_estimate: 20,
        message_tokens_estimate: 30,
        recent_memory_tokens_estimate: 0,
        retrieved_memory_tokens_estimate: 0,
        evidence_memory_tokens_estimate: 0,
        tool_result_tokens_estimate: 40,
      },
    }],
    audit_events: [],
  }
  return {
    boundary: 'wsl-bwrap',
    gpu_enabled: false,
    checkpoints_enabled: false,
    tool_profile: 'benchmark_shared',
    max_output_tokens_per_step: 1024,
    address: 'loopback_ephemeral',
    trace_error: null,
    identity: {
      source_sha: 'a'.repeat(40),
      binary_sha256: 'b'.repeat(64),
      model_sha256: 'c'.repeat(64),
      controller_manifest_sha256: 'd'.repeat(64),
      task_definition_sha256: 'e'.repeat(64),
      fixture_manifest_sha256: 'f'.repeat(64),
      scorer_manifest_sha256: '0'.repeat(64),
    },
    execution: {
      state: 'exited',
      exitCode: 0,
      signal: null,
      timedOut: false,
      durationMs: 25,
      cleanupPassed: true,
      cleanupDetail: null,
      error: null,
      stdout: { preview: 'done\n', totalBytes: 5, capturedBytes: 5, truncated: false },
      stderr: { preview: '', totalBytes: 0, capturedBytes: 0, truncated: false },
    },
    repository_score: {
      outcome: 'PASS_COMPARABLE',
      required_checks: 1,
      passed_checks: 1,
      diff_sha256: 'd'.repeat(64),
    },
    trace,
    attempt: {
      schema: 'camelid.benchmark.agent-attempt/v1',
      campaign_id: 'native-bundle-test',
      task_id: 'agent_local_logic_fix',
      adapter: 'camelid-native',
      attempt: 0,
      comparability: 'comparable',
      terminal: { class: 'answered', exit_code: 0, reason: 'agent exec trace: answered' },
      score: {
        outcome: 'PASS_COMPARABLE',
        required_checks: 1,
        passed_checks: 1,
        diff_sha256: 'd'.repeat(64),
      },
      usage: { model_steps: 1, tool_calls: 0, input_tokens: 100, output_tokens: 7, unavailable_reason: null },
      timing: { wall_ms: 25, model_ms: 20, ttft_ms: 3 },
      process: { cleanup_passed: true },
    },
  }
}

async function exists(path) {
  try {
    await readFile(path)
    return true
  } catch (error) {
    if (error.code === 'ENOENT') return false
    throw error
  }
}