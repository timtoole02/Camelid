#!/usr/bin/env node
import assert from 'node:assert/strict'
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { piBoundaryProbeCommand, piWslBwrapPrefix, PiAdapterError, runPiAgentAttempt } from './adapters/pi-agent.mjs'

const repositoryRoot = resolve(fileURLToPath(new URL('../../..', import.meta.url)))
const taskRoot = resolve(repositoryRoot, 'qa/benchmarks/agent/tasks/agent_local_logic_fix')
const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-pi-adapter-'))
const binaryPath = join(tempRoot, 'camelid')
const modelPath = join(tempRoot, 'model.gguf')
await writeFile(binaryPath, 'synthetic Camelid bytes\n')
await writeFile(modelPath, 'synthetic model bytes\n')

try {
  await assert.rejects(
    run('answered', null),
    (error) => error instanceof PiAdapterError && error.outcome === 'INVALID_INFRASTRUCTURE',
  )
  await assert.rejects(
    run('over-budget', 'synthetic', 300001),
    (error) => error instanceof PiAdapterError && error.outcome === 'INVALID_FIXTURE',
  )
  assert.equal(await exists(join(tempRoot, 'workspace-over-budget')), false)

  const answered = await run('answered', 'synthetic')
  assert.equal(answered.attempt.adapter, 'pi')
  assert.equal(answered.attempt.terminal.class, 'answered')
  assert.equal(answered.attempt.score.outcome, 'PASS_COMPARABLE')
  assert.equal(answered.repository_score.outcome, 'PASS_COMPARABLE')
  assert.equal(answered.event_error, null)
  assert.equal(answered.attempt.usage.model_steps, 1)
  assert.equal(answered.attempt.usage.tool_calls, 1)
  assert.equal(answered.attempt.usage.input_tokens, 100)
  assert.equal(answered.attempt.usage.output_tokens, 7)
  assert.equal(answered.identity.pi_version, '0.84.3')
  assert.equal(answered.identity.pi_archive_sha256, null)
  assert.match(answered.identity.pi_supervisor_sha256, /^[0-9a-f]{64}$/)
  assert.equal(answered.identity.task_definition_sha256, '45729be61b56480e6656dc19fa1e4b01fc0d8784ab8b880ed1d09dc60f0e709b')
  assert.equal(answered.tool_profile, 'benchmark_shared')
  assert.equal(answered.max_output_tokens_per_step, 256)
  assert.equal(answered.execution.stdout.truncated, false)
  assert.equal(JSON.stringify(answered).includes('must-not-reach-pi'), false)
  assertPiArgs(answered.args)

  const failed = await run('failed', 'synthetic')
  assert.equal(failed.attempt.terminal.class, 'failed')
  assert.equal(failed.attempt.score.outcome, 'FAIL_AGENT_TERMINAL')

  const malformed = await run('malformed', 'synthetic')
  assert.equal(malformed.attempt.terminal.class, 'adapter_error')
  assert.equal(malformed.attempt.score.outcome, 'INVALID_INFRASTRUCTURE')
  assert.match(malformed.event_error, /invalid/)

  const tampered = await run('config-tamper', 'synthetic')
  assert.equal(tampered.attempt.terminal.class, 'adapter_error')
  assert.equal(tampered.attempt.score.outcome, 'INVALID_INFRASTRUCTURE')
  assert.match(tampered.event_error, /config changed/)

  const timedOut = await run('timeout', 'synthetic', 500)
  assert.equal(timedOut.attempt.terminal.class, 'timed_out')
  assert.equal(timedOut.attempt.score.outcome, 'INCONCLUSIVE_TIMEOUT')
  assert.equal(timedOut.execution.cleanupPassed, true)
  await assertPortReusable(Number(await readFile(join(tempRoot, 'timeout-port.txt'), 'utf8')))
  assert.equal(await readFile(join(timedOut.workspace_root, 'canary', 'outside.txt'), 'utf8'), 'camelid-benchmark-canary/v1\nagent_local_logic_fix\noutside_task_root\n')

  const bwrap = piWslBwrapPrefix({
    distribution: 'Ubuntu',
    linuxPiDirPath: '/root/pi-0.84.3',
    linuxBinaryPath: '/root/camelid/camelid',
    linuxModelPath: '/root/models/model.gguf',
    linuxAttemptPath: '/mnt/c/runs/task/attempt',
    linuxConfigPath: '/mnt/c/runs/task/control/pi-agent/models.json',
    linuxSupervisorPath: '/mnt/c/repo/tools/bench/system/process/pi-camelid-supervisor.sh',
    linuxProviderExtensionPath: '/mnt/c/repo/tools/bench/system/adapters/pi-camelid-provider.mjs',
    modelId: 'exact-model-id',
    contextWindow: 40960,
    sourceSha: 'a'.repeat(40),
    modelSha256: 'b'.repeat(64),
    piArgs: ['--mode', 'json', '--', 'fix it'],
    readyTimeoutSeconds: 300,
  })
  assert.ok(bwrap.includes('--unshare-all'))
  assert.ok(bwrap.includes('--clearenv'))
  assert.ok(bwrap.includes('/opt/pi'))
  assert.ok(bwrap.includes('/opt/camelid/camelid'))
  assert.ok(bwrap.includes('/opt/controller/pi-camelid-supervisor.sh'))
  assert.ok(bwrap.includes('/opt/controller/pi-camelid-provider.mjs'))
  assert.ok(bwrap.includes('/tmp/pi-agent/models.json'))
  assert.ok(bwrap.includes('/workspace'))
  assert.ok(bwrap.includes('exact-model-id'))
  assert.ok(bwrap.includes('40960'))
  assert.ok(bwrap.includes('a'.repeat(40)))
  assert.ok(bwrap.includes('b'.repeat(64)))
  assert.equal(bwrap.join(' ').includes('--bind /mnt/c /mnt/c'), false)
  assert.equal(bwrap.includes('/dev/dxg'), false)
  assert.ok(bwrap.includes('CAMELID_NO_REMOTE_DIMS'))
  assert.equal(bwrap.at(-1), 'fix it')

  const cpuProbe = piBoundaryProbeCommand(false)
  assert.equal(cpuProbe.includes('$('), false)
  assert.match(cpuProbe, /test -x \/usr\/bin\/find/)
  assert.match(cpuProbe, /test -x \/usr\/bin\/grep/)
  assert.equal(cpuProbe.endsWith('/opt/pi/pi --version'), true)
  const gpuProbe = piBoundaryProbeCommand(true)
  assert.equal(gpuProbe.includes('$('), false)
  assert.match(gpuProbe, /nvidia-smi -L/)

  const supervisor = await readFile(resolve(import.meta.dirname, 'process/pi-camelid-supervisor.sh'), 'utf8')
  assert.match(supervisor, /else 2 if len\(data\) == 0 else 1/)
  assert.match(supervisor, /model_status.*-ne 2/s)
} finally {
  await rm(tempRoot, { recursive: true, force: true })
}

console.log('benchmark Phase 4 Pi adapter canned lifecycle: PASS')

async function run(mode, boundary, timeoutMs = 5000) {
  const scriptPath = join(tempRoot, `fake-pi-${mode}.mjs`)
  await writeFile(scriptPath, candidateSource(mode))
  return runPiAgentAttempt({
    taskRoot,
    workspaceRoot: join(tempRoot, `workspace-${mode}`),
    piExecutablePath: process.execPath,
    binaryPath,
    modelPath,
    modelId: 'synthetic-model',
    baseUrl: 'http://127.0.0.1:8231/v1',
    campaignId: `pi-${mode}`,
    sourceSha: 'a'.repeat(40),
    attempt: 0,
    timeoutMs,
    contextWindow: 4096,
    maxOutputTokensPerStep: 256,
    boundary,
    syntheticCandidate: true,
    syntheticCandidatePrefix: [scriptPath],
    env: { ...process.env, OPENAI_API_KEY: 'must-not-reach-pi' },
  })
}

function candidateSource(mode) {
  return `
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'

const args = process.argv.slice(2)
assert.equal(process.env.OPENAI_API_KEY, undefined)
assert.equal(process.env.PI_OFFLINE, '1')
assert.equal(process.env.PI_TELEMETRY, '0')
for (const flag of ['--no-session','--no-approve','--no-extensions','--no-skills','--no-prompt-templates','--no-themes','--no-context-files']) assert.ok(args.includes(flag), flag)
assert.equal(value('--mode'), 'json')
assert.equal(value('--provider'), 'camelid-benchmark')
assert.equal(value('--model'), 'synthetic-model')
assert.match(value('--append-system-prompt'), /Never invent workspace facts/)
assert.equal(value('--append-system-prompt').includes('pricing.cjs'), false)
assert.equal(value('--tools'), 'read,bash,edit,write,ls')
assert.equal(value('--extension'), '/opt/controller/pi-camelid-provider.mjs')
const configPath = process.env.PI_CODING_AGENT_DIR + '/models.json'
const config = JSON.parse(await readFile(configPath, 'utf8'))
assert.equal(config.providers['camelid-benchmark'].models[0].maxTokens, 256)
if (${JSON.stringify(mode)} === 'timeout') {
  const server = createServer()
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  await writeFile(${JSON.stringify(join(tempRoot, 'timeout-port.txt'))}, String(server.address().port))
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
  child.unref()
  setInterval(() => {}, 1000)
}
if (${JSON.stringify(mode)} === 'malformed') {
  process.stdout.write('{"type":"session","version":3}\\n{"type":"agent_start"')
  process.exit(0)
}
if (${JSON.stringify(mode)} === 'config-tamper') await writeFile(configPath, '{}\\n')
if (${JSON.stringify(mode)} === 'answered' || ${JSON.stringify(mode)} === 'config-tamper') {
  const path = process.cwd() + '/src/pricing.cjs'
  const source = await readFile(path, 'utf8')
  await writeFile(path, source.replace('subtotalCents > 10000', 'subtotalCents >= 10000'))
}
emit({type:'session',version:3,id:'synthetic',timestamp:'2026-08-25T00:00:00Z',cwd:process.cwd()})
emit({type:'agent_start'})
if (${JSON.stringify(mode)} === 'failed') {
  emit({type:'message_end',message:{role:'assistant',content:[],stopReason:'error',errorMessage:'synthetic provider failure'}})
  emit({type:'agent_end',messages:[]})
  process.exit(1)
}
emit({type:'message_update',usage:{input:100,output:7,cacheRead:0,cacheWrite:0,totalTokens:107},assistantMessageEvent:{type:'text_delta',contentIndex:0,delta:'done'}})
emit({type:'tool_execution_start',toolCallId:'call-1',toolName:'write',args:{path:'src/pricing.cjs'}})
emit({type:'tool_execution_end',toolCallId:'call-1',toolName:'write',result:{content:'ok'},isError:false})
emit({type:'message_end',message:{role:'assistant',content:[{type:'text',text:'done'}],stopReason:'stop'}})
emit({type:'agent_end',messages:[]})
function value(flag) { const index = args.indexOf(flag); assert.ok(index >= 0); return args[index + 1] }
function emit(event) { process.stdout.write(JSON.stringify(event) + '\\n') }
`
}

function assertPiArgs(args) {
  assert.equal(args[args.indexOf('--mode') + 1], 'json')
  assert.equal(args[args.indexOf('--provider') + 1], 'camelid-benchmark')
  assert.match(args[args.indexOf('--append-system-prompt') + 1], /Inspect the workspace/)
  assert.equal(args[args.indexOf('--tools') + 1], 'read,bash,edit,write,ls')
}

async function assertPortReusable(port) {
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