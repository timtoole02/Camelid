#!/usr/bin/env node
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { verifyBundleChecksums, writePiAgentBundle } from './bundle.mjs'
import { canonicalJson, sha256Bytes, sha256File } from './lib/digest.mjs'

const root = await mkdtemp(join(tmpdir(), 'camelid-pi-bundle-'))
const outputDir = join(root, 'bundle')

try {
  const result = fixtureResult()
  const bundle = await writePiAgentBundle({ outputDir, result }, {
    generatedUtc: '2026-08-25T00:00:00Z',
  })
  assert.equal(bundle.manifest.state, 'COMPLETE')
  assert.equal(bundle.manifest.workspace_included, false)
  assert.equal(bundle.manifest.boundary, 'wsl-bwrap')
  assert.equal(bundle.manifest.gpu_enabled, false)
  assert.equal(bundle.manifest.tool_profile, 'benchmark_shared')
  assert.equal(bundle.manifest.max_output_tokens_per_step, 256)
  assert.equal((await verifyBundleChecksums(outputDir)).ok, true)
  assert.match(await readFile(join(outputDir, 'summary.md'), 'utf8'), /PASS_COMPARABLE/)
  assert.equal(JSON.parse(await readFile(join(outputDir, 'events.json'), 'utf8')).summary.model_steps, 2)
  assert.equal(await sha256File(join(outputDir, 'models.json')), result.identity.pi_config_sha256)
  assert.equal((await readFile(join(outputDir, 'models.json'), 'utf8')).includes('camelid-benchmark-local'), false)
  await assert.rejects(writePiAgentBundle({ outputDir, result }), /not empty/)

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

console.log('benchmark Phase 4 Pi bundle and checksums: PASS')

function fixtureResult() {
  const piConfig = {
    providers: {
      'camelid-benchmark': {
        baseUrl: 'http://127.0.0.1:8231/v1',
        api: 'openai-completions',
        apiKey: '$CAMELID_PI_API_KEY',
        models: [{ id: 'model' }],
      },
    },
  }
  const piConfigSha256 = sha256Bytes(Buffer.from(canonicalJson(piConfig), 'utf8'))
  const events = {
    session: { type: 'session', version: 3, id: 'session-1' },
    events: [
      { type: 'session', version: 3, id: 'session-1' },
      { type: 'agent_start' },
      { type: 'agent_end', messages: [] },
      { type: 'agent_settled' },
    ],
    summary: { model_steps: 2, tool_calls: 1, tool_errors: 0, input_tokens: 20, output_tokens: 10 },
  }
  return {
    boundary: 'wsl-bwrap',
    gpu_enabled: false,
    tool_profile: 'benchmark_shared',
    max_output_tokens_per_step: 256,
    address: 'http://127.0.0.1:8231/v1',
    event_error: null,
    identity: {
      source_sha: 'a'.repeat(40),
      binary_sha256: 'b'.repeat(64),
      model_sha256: 'c'.repeat(64),
      pi_version: '0.84.3',
      pi_source_commit: 'd'.repeat(40),
      pi_archive_sha256: 'e'.repeat(64),
      pi_executable_sha256: 'f'.repeat(64),
      pi_config_sha256: piConfigSha256,
      pi_supervisor_sha256: '1'.repeat(64),
      controller_manifest_sha256: '2'.repeat(64),
      task_definition_sha256: '3'.repeat(64),
      fixture_manifest_sha256: '4'.repeat(64),
      scorer_manifest_sha256: '5'.repeat(64),
    },
    pi_config: piConfig,
    execution: {
      state: 'exited',
      exitCode: 0,
      signal: null,
      timedOut: false,
      durationMs: 25,
      cleanupPassed: true,
      cleanupDetail: null,
      error: null,
      stdout: { preview: '', totalBytes: 0, capturedBytes: 0, truncated: false },
      stderr: { preview: 'CAMELID_PI_SERVER_READY\n', totalBytes: 24, capturedBytes: 24, truncated: false },
    },
    repository_score: {
      outcome: 'PASS_COMPARABLE',
      required_checks: 1,
      passed_checks: 1,
      diff_sha256: '6'.repeat(64),
    },
    events,
    attempt: {
      schema: 'camelid.benchmark.agent-attempt/v1',
      campaign_id: 'pi-bundle-test',
      task_id: 'agent_local_logic_fix',
      adapter: 'pi',
      attempt: 0,
      comparability: 'comparable',
      terminal: { class: 'answered', exit_code: 0, reason: 'Pi JSON stream completed' },
      score: {
        outcome: 'PASS_COMPARABLE',
        required_checks: 1,
        passed_checks: 1,
        diff_sha256: '6'.repeat(64),
      },
      usage: { model_steps: 2, tool_calls: 1, input_tokens: 20, output_tokens: 10, unavailable_reason: null },
      timing: { wall_ms: 25, model_ms: null, ttft_ms: null },
      process: { cleanup_passed: true },
    },
  }
}