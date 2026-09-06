#!/usr/bin/env node
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { ContractError, validateAgentExecTrace } from './lib/contracts.mjs'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))
const schema = JSON.parse(await readFile(resolve(systemRoot, 'schemas/agent-exec-trace-v1.schema.json'), 'utf8'))
assert.equal(schema.$schema, 'https://json-schema.org/draft/2020-12/schema')
assert.equal(schema.$id, 'https://camelid.ai/schemas/benchmark/agent-exec-trace-v1.schema.json')
assert.equal(schema.additionalProperties, false)

const trace = validTrace()
assert.equal(validateAgentExecTrace(trace), trace)

for (const [reason, outcome, exitCode] of [
  ['answered', 'completed', 0],
  ['driver_error', 'failed', 1],
  ['aborted', 'inconclusive', 3],
  ['step_capped', 'inconclusive', 3],
  ['repeated', 'inconclusive', 3],
]) {
  const terminal = structuredClone(trace)
  terminal.terminal = { reason, outcome, exit_code: exitCode, wall_ms: 10 }
  assert.equal(validateAgentExecTrace(terminal), terminal)
}

const leaked = structuredClone(trace)
leaked.audit_events[0].args = { token: 'secret' }
fails(() => validateAgentExecTrace(leaked), '$.audit_events[0].args is not allowed')

const wrongExit = structuredClone(trace)
wrongExit.terminal.exit_code = 1
fails(() => validateAgentExecTrace(wrongExit), '$.terminal.exit_code must equal 0 for reason answered')

const wrongSteps = structuredClone(trace)
wrongSteps.summary.model_steps = 2
fails(() => validateAgentExecTrace(wrongSteps), '$.summary.model_steps must equal $.steps.length')

const wrongTools = structuredClone(trace)
wrongTools.summary.tool_calls = 0
fails(() => validateAgentExecTrace(wrongTools), '$.summary.tool_calls must equal the agent.tool_call count')

console.log('benchmark Phase 3 agent exec trace contract: PASS')

function validTrace() {
  return {
    schema: 'camelid.agent-exec-trace/v1',
    terminal: { reason: 'answered', outcome: 'completed', exit_code: 0, wall_ms: 25 },
    summary: { model_steps: 1, tool_calls: 1, tool_errors: 0, compactions: 0, model_ms: 20, output_tokens: 7 },
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
    audit_events: [
      {
        event: 'agent.tool_call',
        tool: 'read_file',
        approval_tier: 'auto',
        args_digest: `sha256:${'a'.repeat(64)}`,
        outcome: null,
        duration_ms: null,
      },
      {
        event: 'agent.tool_result',
        tool: 'read_file',
        approval_tier: 'auto',
        args_digest: `sha256:${'a'.repeat(64)}`,
        outcome: 'ok',
        duration_ms: 2,
      },
    ],
  }
}

function fails(action, expected) {
  assert.throws(action, (error) => {
    assert.ok(error instanceof ContractError)
    assert.ok(error.message.includes(expected), error.message)
    return true
  })
}