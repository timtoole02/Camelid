#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

import {
  parsePiJsonLines,
  PiEventError,
  PI_BENCHMARK_SYSTEM_PROMPT,
  piJsonArgs,
  PINNED_PI_RELEASE,
  piProviderConfig,
  piSandboxEnvironment,
  PI_SHARED_TOOLS,
} from './adapters/pi-camelid.mjs'

const fixtureRoot = resolve(import.meta.dirname, 'fixtures/pi')

assert.deepEqual(PINNED_PI_RELEASE, {
  version: '0.84.3',
  sourceCommit: '4e58f324fae8ebfa98a3d45181fb248072a2afac',
  archiveName: 'pi-linux-x64.tar.gz',
  archiveSizeBytes: 42458773,
  archiveSha256: '6f8bb67c21bc6b8a8a106d354f56d7fd4a190a3cd8ad3a32db45f6d281a5d008',
  executableSha256: 'ca858fde375ab91531353b22fac6ebdf29c0a153efe754f5f9b8a72a7423ed08',
  releaseUrl: 'https://github.com/earendil-works/pi/releases/tag/v0.84.3',
})

const config = piProviderConfig({
  baseUrl: 'http://127.0.0.1:8231/v1',
  modelId: 'exact-model-id',
  contextWindow: 4096,
  maxTokens: 256,
})
const provider = config.providers['camelid-benchmark']
assert.equal(provider.api, 'openai-completions')
assert.equal(provider.apiKey, '$CAMELID_PI_API_KEY')
assert.equal(provider.authHeader, true)
assert.equal(provider.compat.supportsDeveloperRole, false)
assert.equal(provider.compat.supportsReasoningEffort, false)
assert.equal(provider.compat.supportsUsageInStreaming, true)
assert.equal(provider.compat.maxTokensField, 'max_tokens')
assert.equal(provider.compat.requiresToolResultName, false)
assert.equal(provider.models[0].id, 'exact-model-id')
assert.equal(provider.models[0].maxTokens, 256)
assert.equal(JSON.stringify(config).includes('benchmark-secret'), false)
for (const invalid of [
  'https://127.0.0.1:8231/v1',
  'http://localhost:8231/v1',
  'http://127.0.0.1:8231',
  'http://127.0.0.1:8231/v1?unexpected=1',
]) {
  assert.throws(() => piProviderConfig({ baseUrl: invalid, modelId: 'm', contextWindow: 2, maxTokens: 1 }))
}

const args = piJsonArgs({ modelId: 'exact-model-id', goal: 'fix the exact task' })
assert.deepEqual(PI_SHARED_TOOLS, ['read', 'bash', 'edit', 'write', 'ls'])
for (const flag of [
  '--mode', '--provider', '--model', '--thinking', '--no-session', '--no-approve',
  '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes',
  '--no-context-files', '--tools', '--',
]) {
  assert.ok(args.includes(flag), flag)
}
assert.equal(args[args.indexOf('--mode') + 1], 'json')
assert.equal(args[args.indexOf('--tools') + 1], PI_SHARED_TOOLS.join(','))
assert.equal(args[args.indexOf('--extension') + 1], '/opt/controller/pi-camelid-provider.mjs')
assert.equal(args[args.indexOf('--append-system-prompt') + 1], PI_BENCHMARK_SYSTEM_PROMPT)
assert.match(PI_BENCHMARK_SYSTEM_PROMPT, /Inspect the workspace/)
assert.match(PI_BENCHMARK_SYSTEM_PROMPT, /Never invent workspace facts/)
assert.equal(PI_BENCHMARK_SYSTEM_PROMPT.includes('pricing.cjs'), false)
assert.equal(PI_BENCHMARK_SYSTEM_PROMPT.includes('10000'), false)
assert.equal(args.at(-1), 'fix the exact task')

assert.deepEqual(piSandboxEnvironment({ apiKey: 'benchmark-secret' }), {
  PATH: '/usr/bin:/bin',
  HOME: '/tmp/home',
  PI_CODING_AGENT_DIR: '/tmp/pi-agent',
  PI_OFFLINE: '1',
  PI_SKIP_VERSION_CHECK: '1',
  PI_TELEMETRY: '0',
  CAMELID_PI_API_KEY: 'benchmark-secret',
})

const valid = parsePiJsonLines(await readFile(resolve(fixtureRoot, 'valid-events-v3.jsonl'), 'utf8'))
assert.equal(valid.session.version, 3)
assert.equal(valid.events.at(-1).type, 'agent_settled')
assert.deepEqual(valid.summary, {
  model_steps: 1,
  tool_calls: 1,
  tool_errors: 0,
  input_tokens: 100,
  output_tokens: 7,
})

const truncated = await readFile(resolve(fixtureRoot, 'invalid-truncated-events-v3.jsonl'), 'utf8')
assert.throws(() => parsePiJsonLines(truncated), PiEventError)
assert.throws(() => parsePiJsonLines('{"type":"session","version":4}\n'), /unsupported Pi JSON stream version 4/)
assert.throws(
  () => parsePiJsonLines('{"type":"session","version":3}\n{"type":"agent_start"}\n{"type":"tool_execution_end","toolCallId":"missing","isError":false}\n{"type":"agent_end"}\n'),
  /no matching start/,
)
assert.throws(
  () => parsePiJsonLines('{"type":"session","version":3}\n{"type":"agent_start"}\n'),
  /one agent_end/,
)

const twoStep = parsePiJsonLines([
  '{"type":"session","version":3}',
  '{"type":"agent_start"}',
  '{"type":"message_start","message":{"role":"assistant"}}',
  '{"type":"message_update","usage":{"input":10,"output":5},"assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"a"}}',
  '{"type":"message_end","message":{"role":"assistant","stopReason":"toolUse"}}',
  '{"type":"message_start","message":{"role":"assistant"}}',
  '{"type":"message_update","usage":{"input":12,"output":3},"assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"b"}}',
  '{"type":"message_end","message":{"role":"assistant","stopReason":"stop"}}',
  '{"type":"agent_end","messages":[]}',
  '{"type":"agent_settled"}',
].join('\n'))
assert.equal(twoStep.summary.model_steps, 2)
assert.equal(twoStep.summary.input_tokens, 22)
assert.equal(twoStep.summary.output_tokens, 8)

console.log('benchmark Phase 4 Pi contract: PASS')