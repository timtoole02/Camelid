#!/usr/bin/env node
import assert from 'node:assert/strict'
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  linuxModelSandboxPath,
  NativeAdapterError,
  nativeExecArgs,
  runNativeAgentAttempt,
  windowsPathToWsl,
  wslBwrapPrefix,
} from './adapters/native-camelid.mjs'

const repositoryRoot = resolve(fileURLToPath(new URL('../../..', import.meta.url)))
const taskRoot = resolve(repositoryRoot, 'qa/benchmarks/agent/tasks/agent_local_logic_fix')
const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-native-adapter-'))
const modelPath = join(tempRoot, 'model.gguf')
await writeFile(modelPath, 'synthetic model bytes\n')

try {
  await assert.rejects(
    run('answered', null),
    (error) => error instanceof NativeAdapterError && error.outcome === 'INVALID_INFRASTRUCTURE',
  )
  await assert.rejects(
    run('over-budget', { kind: 'synthetic' }, 300001),
    (error) => error instanceof NativeAdapterError && error.outcome === 'INVALID_FIXTURE',
  )
  assert.equal(await exists(join(tempRoot, 'workspace-over-budget')), false)
  await assert.rejects(
    run('token-over-budget', { kind: 'synthetic' }, 5000, 1025),
    (error) => error instanceof NativeAdapterError && error.outcome === 'INVALID_FIXTURE',
  )
  assert.equal(await exists(join(tempRoot, 'workspace-token-over-budget')), false)

  const answered = await run('answered', { kind: 'synthetic' })
  assert.equal(answered.attempt.terminal.class, 'answered')
  assert.equal(answered.attempt.terminal.exit_code, 0)
  assert.equal(answered.attempt.score.outcome, 'PASS_COMPARABLE')
  assert.equal(answered.repository_score.outcome, 'PASS_COMPARABLE')
  assert.equal(answered.trace_error, null)
  assert.equal(answered.attempt.usage.model_steps, 1)
  assert.equal(answered.attempt.usage.tool_calls, 1)
  assert.equal(answered.attempt.usage.input_tokens, 100)
  assert.equal(answered.attempt.usage.output_tokens, 7)
  assert.equal(answered.attempt.timing.model_ms, 20)
  assert.equal(answered.attempt.timing.ttft_ms, 3)
  assert.equal(answered.checkpoints_enabled, false)
  assert.equal(answered.tool_profile, 'benchmark_shared')
  assert.equal(answered.max_output_tokens_per_step, 1024)
  assert.equal(answered.identity.task_definition_sha256, '45729be61b56480e6656dc19fa1e4b01fc0d8784ab8b880ed1d09dc60f0e709b')
  assert.equal(answered.identity.fixture_manifest_sha256, 'f5a273b7ae8d07987de95ec85389bbceba3a3d632c31689e86179ffc4f9e2895')
  assert.equal(answered.identity.scorer_manifest_sha256, '6b8fe59f3fc70be1ce38d12809c49768015e9893808f203ac89cb23980c49973')
  assert.equal(JSON.stringify(answered.trace).includes('must-not-reach-candidate'), false)
  assert.equal(await exists(join(answered.attempt_root, '.camelid-benchmark-trace.json')), false)
  assertSecurityArgs(answered.args)

  const noncomparable = await run('arm-specific', { kind: 'synthetic' })
  assert.equal(noncomparable.attempt.comparability, 'noncomparable')
  assert.equal(noncomparable.attempt.score.outcome, 'PASS_NONCOMPARABLE')

  const failed = await run('failed', { kind: 'synthetic' })
  assert.equal(failed.attempt.terminal.class, 'failed')
  assert.equal(failed.attempt.terminal.exit_code, 1)
  assert.equal(failed.attempt.score.outcome, 'FAIL_AGENT_TERMINAL')

  const inconclusive = await run('inconclusive', { kind: 'synthetic' })
  assert.equal(inconclusive.attempt.terminal.class, 'inconclusive')
  assert.equal(inconclusive.attempt.terminal.exit_code, 3)
  assert.equal(inconclusive.attempt.score.outcome, 'FAIL_AGENT_TERMINAL')

  const timedOut = await run('timeout', { kind: 'synthetic' }, 500)
  assert.equal(timedOut.attempt.terminal.class, 'timed_out')
  assert.equal(timedOut.attempt.score.outcome, 'INCONCLUSIVE_TIMEOUT')
  assert.equal(timedOut.execution.cleanupPassed, true)
  await assertPortReusable(addrFromArgs(timedOut.args))
  assert.equal(await readFile(join(timedOut.workspace_root, 'canary', 'outside.txt'), 'utf8'), 'camelid-benchmark-canary/v1\nagent_local_logic_fix\noutside_task_root\n')

  const bwrap = wslBwrapPrefix({
    distribution: 'Ubuntu',
    linuxBinaryPath: '/root/camelid/camelid',
    linuxModelPath: '/mnt/c/models/model.gguf',
    linuxAttemptPath: '/mnt/c/runs/task/attempt',
  })
  assert.ok(bwrap.includes('--unshare-all'))
  assert.ok(bwrap.includes('--clearenv'))
  for (const directory of ['/opt/camelid', '/model', '/workspace']) {
    assert.ok(bwrap.includes(directory), directory)
  }
  assert.equal(bwrap.join(' ').includes('--bind /mnt/c /mnt/c'), false)
  assert.equal(bwrap.includes('/dev/dxg'), false)
  assert.ok(bwrap.includes('/mnt/c/models/model.gguf'))
  assert.ok(bwrap.includes('/model/model.gguf'))
  assert.ok(bwrap.includes('/mnt/c/runs/task/attempt'))
  assert.equal(windowsPathToWsl('C:\\runs\\task'), '/mnt/c/runs/task')
  assert.equal(linuxModelSandboxPath('/root/models/Qwen3-4B-Q4_K_M.gguf'), '/model/Qwen3-4B-Q4_K_M.gguf')

  const gpuBwrap = wslBwrapPrefix({
    distribution: 'Ubuntu',
    linuxBinaryPath: '/root/camelid/camelid',
    linuxModelPath: '/root/models/Qwen3-4B-Q4_K_M.gguf',
    linuxAttemptPath: '/mnt/c/runs/task/attempt',
    gpuEnabled: true,
  })
  const deviceBind = gpuBwrap.indexOf('--dev-bind')
  assert.deepEqual(gpuBwrap.slice(deviceBind, deviceBind + 3), ['--dev-bind', '/dev/dxg', '/dev/dxg'])
  const libraryPath = gpuBwrap.indexOf('LD_LIBRARY_PATH')
  assert.equal(gpuBwrap[libraryPath + 1], '/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu')

  const linuxArgs = nativeExecArgs({
    task: { goal: 'fix the task', budgets: { max_steps: 1, max_output_tokens_per_step: 32, command_ms: 1000 } },
    modelPath: '/model/Qwen3-4B-Q4_K_M.gguf',
    workdir: '/workspace',
    addr: '127.0.0.1:8231',
    tracePath: '/workspace/.camelid-benchmark-trace.json',
    maxOutputTokensPerStep: 256,
  })
  assert.equal(linuxArgs[linuxArgs.indexOf('--model') + 1], '/model/Qwen3-4B-Q4_K_M.gguf')
  assert.equal(linuxArgs[linuxArgs.indexOf('--workdir') + 1], '/workspace')
  assert.equal(linuxArgs[linuxArgs.indexOf('--benchmark-events') + 1], '/workspace/.camelid-benchmark-trace.json')
  assert.equal(linuxArgs[linuxArgs.indexOf('--max-tokens') + 1], '256')
} finally {
  await rm(tempRoot, { recursive: true, force: true })
}

console.log('benchmark Phase 3 native adapter canned lifecycle: PASS')

async function run(mode, boundary, timeoutMs = 5000, maxOutputTokensPerStep = null) {
  const scriptPath = await fakeCandidate(mode)
  return runNativeAgentAttempt({
    taskRoot,
    workspaceRoot: join(tempRoot, `workspace-${mode}`),
    binaryPath: process.execPath,
    modelPath,
    campaignId: `native-${mode}`,
    sourceSha: 'a'.repeat(40),
    attempt: 0,
    timeoutMs,
    maxOutputTokensPerStep,
    boundary,
    env: {
      ...process.env,
      CAMELID_API_KEY: 'must-not-reach-candidate',
      CAMELID_PRODUCTION: 'must-not-reach-candidate',
    },
    syntheticCandidate: true,
    syntheticCandidatePrefix: [scriptPath],
  })
}

async function fakeCandidate(mode) {
  const scriptPath = join(tempRoot, `fake-${mode}.mjs`)
  await writeFile(scriptPath, candidateSource(mode))
  return scriptPath
}

function candidateSource(mode) {
  return `
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'

const args = process.argv.slice(2)
assert.deepEqual(args.slice(0, 2), ['agent', 'exec'])
assert.equal(process.env.CAMELID_API_KEY, undefined)
assert.equal(process.env.CAMELID_PRODUCTION, undefined)
for (const forbidden of ['--allow-net', '--allow-fs', '--allow-mcp']) assert.equal(args.includes(forbidden), false)
for (const required of ['--model', '--addr', '--workdir', '--max-steps', '--max-tokens', '--shell-sandbox', '--shell-timeout', '--today-is-a-good-day-to-die', '--benchmark-events']) assert.ok(args.includes(required), required)
assert.equal(value('--shell-sandbox'), 'sandboxed')
const workdir = value('--workdir')
if (${JSON.stringify(mode)} === 'answered') {
  const path = workdir + '/src/pricing.cjs'
  const source = await readFile(path, 'utf8')
  await writeFile(path, source.replace('subtotalCents > 10000', 'subtotalCents >= 10000'))
  await writeTrace('answered', 'completed', 0, true)
  process.stdout.write('done\\n')
  process.exit(0)
}
if (${JSON.stringify(mode)} === 'arm-specific') {
  const path = workdir + '/src/pricing.cjs'
  const source = await readFile(path, 'utf8')
  await writeFile(path, source.replace('subtotalCents > 10000', 'subtotalCents >= 10000'))
  await writeTrace('answered', 'completed', 0, true, 'run_windows_command')
  process.exit(0)
}
if (${JSON.stringify(mode)} === 'failed') { await writeTrace('driver_error', 'failed', 1, false); process.exit(1) }
if (${JSON.stringify(mode)} === 'inconclusive') { await writeTrace('step_capped', 'inconclusive', 3, false); process.exit(3) }
if (${JSON.stringify(mode)} === 'timeout') {
  const child = spawn(process.execPath, ['-e', \`require('net').createServer().listen(\${JSON.stringify(value('--addr').split(':').at(-1) * 1)}, '127.0.0.1'); setInterval(() => {}, 1000)\`], { stdio: 'ignore' })
  child.unref()
  setInterval(() => {}, 1000)
}
function value(flag) { const index = args.indexOf(flag); assert.ok(index >= 0); return args[index + 1] }
async function writeTrace(reason, outcome, exitCode, withTool, tool = 'write_file') {
  const audit = withTool ? [
    { event:'agent.tool_call', tool, approval_tier:'exec', args_digest:'sha256:' + 'a'.repeat(64), outcome:null, duration_ms:null },
    { event:'agent.tool_result', tool, approval_tier:'exec', args_digest:'sha256:' + 'a'.repeat(64), outcome:'ok', duration_ms:2 },
  ] : []
  await writeFile(value('--benchmark-events'), JSON.stringify({
    schema:'camelid.agent-exec-trace/v1',
    terminal:{reason,outcome,exit_code:exitCode,wall_ms:25},
    summary:{model_steps:1,tool_calls:withTool ? 1 : 0,tool_errors:0,compactions:0,model_ms:20,output_tokens:7},
    steps:[{index:0,model_ms:20,ttft_ms:3,output_tokens:7,context:{prompt_tokens:100,generation_tokens:32,budget_tokens:4096,system_tokens_estimate:10,tool_definition_tokens_estimate:20,message_tokens_estimate:30,recent_memory_tokens_estimate:0,retrieved_memory_tokens_estimate:0,evidence_memory_tokens_estimate:0,tool_result_tokens_estimate:40}}],
    audit_events:audit,
  }))
}
`
}

function assertSecurityArgs(args) {
  for (const flag of ['--allow-net', '--allow-fs', '--allow-mcp']) assert.equal(args.includes(flag), false)
  assert.ok(args.includes('--today-is-a-good-day-to-die'))
  assert.equal(args[args.indexOf('--shell-sandbox') + 1], 'sandboxed')
}

function addrFromArgs(args) {
  return args[args.indexOf('--addr') + 1]
}

async function assertPortReusable(addr) {
  const port = Number(addr.split(':').at(-1))
  const server = createServer()
  await new Promise((resolveListen, reject) => {
    server.once('error', reject)
    server.listen(port, '127.0.0.1', resolveListen)
  })
  await new Promise((resolveClose) => server.close(resolveClose))
}

async function exists(path) {
  try {
    await access(path)
    return true
  } catch (error) {
    if (error.code === 'ENOENT') return false
    throw error
  }
}