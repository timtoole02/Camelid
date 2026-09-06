#!/usr/bin/env node
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

import { PINNED_PI_RELEASE, PI_BENCHMARK_SYSTEM_PROMPT, PI_SHARED_TOOLS } from './adapters/pi-camelid.mjs'

const fixture = JSON.parse(await readFile(
  resolve(import.meta.dirname, 'fixtures/pi/openai-compatibility-v1.json'),
  'utf8',
))

assert.equal(fixture.schema, 'camelid.benchmark.pi-openai-compatibility/v1')
assert.equal(fixture.pi.version, PINNED_PI_RELEASE.version)
assert.equal(fixture.pi.archive_sha256, PINNED_PI_RELEASE.archiveSha256)
assert.equal(fixture.pi.executable_sha256, PINNED_PI_RELEASE.executableSha256)
assert.equal(fixture.provider_api, 'openai-completions')
assert.equal(
  fixture.system_prompt_append_sha256,
  createHash('sha256').update(PI_BENCHMARK_SYSTEM_PROMPT).digest('hex'),
)
assert.equal(fixture.observed_requests.length, 2)

const [initial, continuation] = fixture.observed_requests
const expectedFields = ['max_tokens', 'messages', 'model', 'stream', 'stream_options', 'temperature', 'tools']
assert.deepEqual(initial.fields, expectedFields)
assert.deepEqual(initial.roles, ['system', 'user'])
assert.equal(initial.stream, true)
assert.equal(initial.stream_include_usage, true)
assert.equal(initial.max_tokens, 256)
assert.deepEqual(initial.tools, PI_SHARED_TOOLS)
assert.ok(initial.strict_tool_fields.every((value) => value === null))

assert.deepEqual(continuation.fields, expectedFields)
assert.deepEqual(continuation.roles, ['system', 'user', 'assistant', 'tool'])
assert.equal(continuation.stream, true)
assert.equal(continuation.stream_include_usage, true)
assert.equal(continuation.max_tokens, 256)
assert.equal(continuation.tool_result_has_name, false)

assert.deepEqual(fixture.observed_responses, {
  tool_call_id_stable: true,
  tool_arguments_stream_as_json: true,
  tool_finish_reason: 'tool_calls',
  final_finish_reason: 'stop',
  terminal_usage_chunk: true,
  done_marker: true,
})
assert.deepEqual(fixture.omitted_request_fields, [
  'developer_role',
  'max_completion_tokens',
  'reasoning_effort',
  'store',
  'strict_tools',
])
assert.deepEqual(fixture.explicitly_unqualified, [
  'reasoning_controls',
  'store',
  'strict_tools',
  'grammar_tools',
])

console.log('benchmark Phase 4 Pi OpenAI compatibility contract: PASS')