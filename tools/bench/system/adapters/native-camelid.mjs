import { createServer } from 'node:net'
import { readFile, rm } from 'node:fs/promises'
import { join, resolve } from 'node:path'

import { validateAgentAttempt, validateAgentExecTrace } from '../lib/contracts.mjs'
import { controllerManifest } from '../lib/controller-manifest.mjs'
import { sha256File } from '../lib/digest.mjs'
import { runProcess } from '../process/runner.mjs'
import { loadTaskPackage, materializeTask, scoreTaskAttempt } from '../tasks/package.mjs'

const FORBIDDEN_SHARED_FLAGS = new Set(['--allow-net', '--allow-fs', '--allow-mcp'])
const COMMON_NATIVE_TOOLS = new Set(['read_file', 'list_dir', 'search', 'write_file', 'edit_file', 'run_shell'])
const systemRoot = resolve(import.meta.dirname, '..')

export async function runNativeAgentAttempt(options) {
  const normalized = validateOptions(options)
  const taskPackage = await loadTaskPackage(normalized.taskRoot)
  if (taskPackage.task.network !== 'deny') {
    throw new NativeAdapterError('INVALID_FIXTURE', 'shared native tasks must deny network access')
  }
  if (normalized.timeoutMs > taskPackage.task.budgets.wall_ms) {
    throw new NativeAdapterError('INVALID_FIXTURE', 'native adapter timeout cannot exceed the task wall budget')
  }
  const maxOutputTokensPerStep = normalized.maxOutputTokensPerStep
    ?? taskPackage.task.budgets.max_output_tokens_per_step
  if (maxOutputTokensPerStep > taskPackage.task.budgets.max_output_tokens_per_step) {
    throw new NativeAdapterError('INVALID_FIXTURE', 'native adapter token cap cannot exceed the task token budget')
  }

  // Resolve immutable inputs before materializing any writable task state.
  const [binarySha256, modelSha256] = await Promise.all([
    sha256File(normalized.binaryPath),
    sha256File(normalized.modelPath),
  ])
  const controller = await controllerManifest(systemRoot)
  if (normalized.boundary.kind === 'wsl-bwrap') {
    await verifyWslBoundary(normalized.boundary)
    await verifyWslIdentity(normalized.boundary, binarySha256, modelSha256)
  }

  const materialized = await materializeTask(taskPackage, normalized.workspaceRoot)
  const tracePath = join(materialized.attemptRoot, '.camelid-benchmark-trace.json')
  const launch = await prepareLaunch(normalized, materialized.attemptRoot, tracePath)
  const args = nativeExecArgs({
    task: taskPackage.task,
    modelPath: launch.modelPath,
    workdir: launch.workdir,
    addr: launch.addr,
    tracePath: launch.tracePath,
    maxOutputTokensPerStep,
  })
  const startedAt = performance.now()
  const execution = await runProcess({
    file: launch.file,
    args: [...launch.prefixArgs, ...args],
    cwd: materialized.attemptRoot,
    env: isolatedNativeEnv(normalized.env),
    timeoutMs: normalized.timeoutMs,
  })
  const wallMs = performance.now() - startedAt

  const traceState = await readNativeTrace(tracePath, execution)
  await rm(tracePath, { force: true })

  // The scorer runs only after runProcess has observed a terminal child and,
  // on timeout, completed exact descendant cleanup.
  const repositoryScore = await scoreTaskAttempt(normalized.taskRoot, materialized.workspaceRoot)
  const terminal = terminalFromExecution(execution, traceState.trace)
  const comparability = comparabilityFromTrace(traceState.trace)
  const outcome = attemptOutcome(
    terminal,
    execution.cleanupPassed,
    repositoryScore.outcome,
    traceState.error,
    comparability,
  )
  const usage = usageFromTrace(traceState.trace)
  const timing = timingFromTrace(traceState.trace, wallMs)
  const attemptRecord = validateAgentAttempt({
    schema: 'camelid.benchmark.agent-attempt/v1',
    campaign_id: normalized.campaignId,
    task_id: taskPackage.task.id,
    adapter: 'camelid-native',
    attempt: normalized.attempt,
    comparability,
    terminal,
    score: {
      outcome,
      required_checks: Math.max(1, repositoryScore.required_checks),
      passed_checks: repositoryScore.passed_checks,
      diff_sha256: repositoryScore.diff_sha256,
    },
    usage,
    timing,
    process: {
      cleanup_passed: execution.cleanupPassed,
    },
  })

  return {
    identity: {
      source_sha: normalized.sourceSha,
      binary_sha256: binarySha256,
      model_sha256: modelSha256,
      controller_manifest_sha256: controller.sha256,
      task_definition_sha256: taskPackage.taskDefinitionSha256,
      fixture_manifest_sha256: taskPackage.fixture.sha256,
      scorer_manifest_sha256: taskPackage.scorer.sha256,
    },
    boundary: normalized.boundary.kind,
    gpu_enabled: normalized.boundary.gpuEnabled,
    checkpoints_enabled: false,
    tool_profile: 'benchmark_shared',
    max_output_tokens_per_step: maxOutputTokensPerStep,
    address: 'loopback_ephemeral',
    args,
    execution,
    trace: traceState.trace,
    trace_error: traceState.error,
    repository_score: repositoryScore,
    attempt: attemptRecord,
    workspace_root: materialized.workspaceRoot,
    attempt_root: materialized.attemptRoot,
  }
}

async function prepareLaunch(options, attemptRoot, tracePath) {
  if (options.boundary.kind === 'synthetic') {
    return {
      file: options.binaryPath,
      prefixArgs: options.syntheticCandidatePrefix,
      addr: await reserveLoopbackAddress(),
      modelPath: options.modelPath,
      workdir: attemptRoot,
      tracePath,
    }
  }
  const sandboxModelPath = linuxModelSandboxPath(options.boundary.linuxModelPath)
  return {
    file: options.boundary.wslExecutable,
    prefixArgs: wslBwrapPrefix({
      distribution: options.boundary.distribution,
      linuxBinaryPath: options.boundary.linuxBinaryPath,
      linuxModelPath: options.boundary.linuxModelPath,
      linuxAttemptPath: windowsPathToWsl(attemptRoot),
      gpuEnabled: options.boundary.gpuEnabled,
    }),
    // Each bwrap process owns a fresh network namespace.
    addr: '127.0.0.1:8231',
    modelPath: sandboxModelPath,
    workdir: '/workspace',
    tracePath: '/workspace/.camelid-benchmark-trace.json',
  }
}

export function wslBwrapPrefix({ distribution, linuxBinaryPath, linuxModelPath, linuxAttemptPath, gpuEnabled = false }) {
  const sandboxModelPath = linuxModelSandboxPath(linuxModelPath)
  const gpuDeviceArgs = gpuEnabled ? ['--dev-bind', '/dev/dxg', '/dev/dxg'] : []
  const gpuEnvironmentArgs = gpuEnabled
    ? ['--setenv', 'LD_LIBRARY_PATH', '/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu']
    : []
  return [
    '-d', distribution, '--',
    'bwrap',
    '--unshare-all',
    '--new-session',
    '--die-with-parent',
    '--ro-bind', '/usr', '/usr',
    '--ro-bind', '/lib', '/lib',
    '--ro-bind', '/lib64', '/lib64',
    '--ro-bind', '/bin', '/bin',
    '--ro-bind', '/sys', '/sys',
    '--proc', '/proc',
    '--dev', '/dev',
    ...gpuDeviceArgs,
    '--tmpfs', '/tmp',
    '--dir', '/opt',
    '--dir', '/opt/camelid',
    '--dir', '/model',
    '--dir', '/workspace',
    '--dir', '/tmp/home',
    '--ro-bind', linuxBinaryPath, '/opt/camelid/camelid',
    '--ro-bind', linuxModelPath, sandboxModelPath,
    '--bind', linuxAttemptPath, '/workspace',
    '--chdir', '/workspace',
    '--clearenv',
    '--setenv', 'PATH', '/usr/bin:/bin',
    '--setenv', 'HOME', '/tmp/home',
    ...gpuEnvironmentArgs,
    '/opt/camelid/camelid',
  ]
}

export function linuxModelSandboxPath(path) {
  const filename = path.split('/').at(-1)
  if (!filename || filename === '.' || filename === '..') {
    throw new TypeError('linuxModelPath must name a model file')
  }
  return `/model/${filename}`
}

export function windowsPathToWsl(path) {
  const match = path.match(/^([A-Za-z]):[\\/](.*)$/)
  if (!match) {
    if (path.startsWith('/')) return path
    throw new NativeAdapterError('INVALID_INFRASTRUCTURE', `WSL boundary requires a drive-qualified Windows path: ${path}`)
  }
  return `/mnt/${match[1].toLowerCase()}/${match[2].replaceAll('\\', '/')}`
}

export function nativeExecArgs({ task, modelPath, workdir, addr, tracePath, maxOutputTokensPerStep = task.budgets.max_output_tokens_per_step }) {
  const args = [
    'agent',
    'exec',
    task.goal,
    '--model',
    modelPath,
    '--addr',
    addr,
    '--workdir',
    workdir,
    '--max-steps',
    String(task.budgets.max_steps),
    '--max-tokens',
    String(maxOutputTokensPerStep),
    '--shell-sandbox',
    'sandboxed',
    '--shell-timeout',
    String(Math.max(1, Math.ceil(task.budgets.command_ms / 1000))),
    '--today-is-a-good-day-to-die',
    '--benchmark-events',
    tracePath,
  ]
  for (const flag of FORBIDDEN_SHARED_FLAGS) {
    if (args.includes(flag)) throw new NativeAdapterError('INVALID_FIXTURE', `shared task enabled forbidden flag ${flag}`)
  }
  return args
}

export async function reserveLoopbackAddress() {
  const server = createServer()
  await new Promise((resolveListen, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolveListen)
  })
  const address = server.address()
  await new Promise((resolveClose, reject) => server.close((error) => error ? reject(error) : resolveClose()))
  if (address === null || typeof address === 'string') throw new NativeAdapterError('INVALID_INFRASTRUCTURE', 'could not reserve a loopback TCP port')
  return `127.0.0.1:${address.port}`
}

export class NativeAdapterError extends Error {
  constructor(outcome, message) {
    super(message)
    this.name = 'NativeAdapterError'
    this.outcome = outcome
  }
}

function validateOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('native adapter options must be an object')
  }
  for (const name of ['taskRoot', 'workspaceRoot', 'binaryPath', 'modelPath']) {
    if (typeof options[name] !== 'string' || options[name].length === 0) throw new TypeError(`${name} must be a non-empty string`)
  }
  if (typeof options.campaignId !== 'string' || options.campaignId.length === 0) throw new TypeError('campaignId must be non-empty')
  if (typeof options.sourceSha !== 'string' || !/^[0-9a-f]{40}$/.test(options.sourceSha)) throw new TypeError('sourceSha must be 40 lowercase hex characters')
  if (!Number.isSafeInteger(options.attempt) || options.attempt < 0) throw new TypeError('attempt must be a non-negative safe integer')
  if (!Number.isSafeInteger(options.timeoutMs) || options.timeoutMs < 1) throw new TypeError('timeoutMs must be a positive safe integer')
  return {
    taskRoot: resolve(options.taskRoot),
    workspaceRoot: resolve(options.workspaceRoot),
    binaryPath: resolve(options.binaryPath),
    modelPath: resolve(options.modelPath),
    campaignId: options.campaignId,
    sourceSha: options.sourceSha,
    attempt: options.attempt,
    timeoutMs: options.timeoutMs,
    maxOutputTokensPerStep: optionalPositiveInteger(options.maxOutputTokensPerStep, 'maxOutputTokensPerStep'),
    env: options.env ?? process.env,
    boundary: validateBoundary(options),
    syntheticCandidatePrefix: syntheticPrefix(options),
  }
}

function validateBoundary(options) {
  const boundary = options.boundary
  if (boundary === null || typeof boundary !== 'object' || Array.isArray(boundary)) {
    throw new NativeAdapterError('INVALID_INFRASTRUCTURE', 'unattended native agent execution requires a verified boundary')
  }
  if (boundary.kind === 'synthetic') {
    if (options.syntheticCandidate !== true) throw new NativeAdapterError('INVALID_INFRASTRUCTURE', 'synthetic boundary is test-only')
    return { kind: 'synthetic', gpuEnabled: false }
  }
  if (boundary.kind !== 'wsl-bwrap') throw new NativeAdapterError('INVALID_INFRASTRUCTURE', `unsupported native boundary ${boundary.kind}`)
  if (boundary.gpuEnabled !== undefined && typeof boundary.gpuEnabled !== 'boolean') {
    throw new TypeError('WSL gpuEnabled must be a boolean')
  }
  if (typeof boundary.distribution !== 'string' || !/^[A-Za-z0-9_.-]+$/.test(boundary.distribution)) {
    throw new TypeError('WSL distribution must be a simple name')
  }
  for (const name of ['linuxBinaryPath', 'linuxModelPath']) {
    if (typeof boundary[name] !== 'string' || !boundary[name].startsWith('/')) throw new TypeError(`${name} must be an absolute Linux path`)
  }
  const systemRoot = process.env.SYSTEMROOT ?? 'C:\\Windows'
  return {
    kind: 'wsl-bwrap',
    distribution: boundary.distribution,
    linuxBinaryPath: boundary.linuxBinaryPath,
    linuxModelPath: boundary.linuxModelPath,
    gpuEnabled: boundary.gpuEnabled ?? false,
    wslExecutable: boundary.wslExecutable
      ? resolve(boundary.wslExecutable)
      : process.platform === 'win32'
        ? resolve(systemRoot, 'System32', 'wsl.exe')
        : 'wsl',
  }
}

function syntheticPrefix(options) {
  if (options.syntheticCandidatePrefix === undefined) return []
  if (options.syntheticCandidate !== true) {
    throw new NativeAdapterError('INVALID_INFRASTRUCTURE', 'a synthetic candidate prefix is test-only')
  }
  if (!Array.isArray(options.syntheticCandidatePrefix)
    || options.syntheticCandidatePrefix.length === 0
    || options.syntheticCandidatePrefix.some((item) => typeof item !== 'string' || item.length === 0)) {
    throw new TypeError('syntheticCandidatePrefix must be a non-empty string array')
  }
  return [...options.syntheticCandidatePrefix]
}

function terminalFromExecution(execution, trace) {
  if (execution.timedOut) return { class: 'timed_out', exit_code: execution.exitCode, reason: 'native agent wall timeout expired' }
  if (execution.state !== 'exited') return { class: 'adapter_error', exit_code: execution.exitCode, reason: execution.error ?? execution.state }
  if (execution.exitCode === 0) return { class: 'answered', exit_code: 0, reason: trace ? `agent exec trace: ${trace.terminal.reason}` : 'agent exec returned answered' }
  if (execution.exitCode === 1) return { class: 'failed', exit_code: 1, reason: trace ? `agent exec trace: ${trace.terminal.reason}` : 'agent exec returned failed or blocked' }
  if (execution.exitCode === 3) return { class: 'inconclusive', exit_code: 3, reason: trace ? `agent exec trace: ${trace.terminal.reason}` : 'agent exec returned inconclusive without a valid trace' }
  return { class: 'adapter_error', exit_code: execution.exitCode, reason: `unexpected agent exec exit ${execution.exitCode}` }
}

function attemptOutcome(terminal, cleanupPassed, repositoryOutcome, traceError, comparability) {
  if (!cleanupPassed) return 'INVALID_INFRASTRUCTURE'
  if (traceError) return 'INVALID_INFRASTRUCTURE'
  if (terminal.class === 'timed_out') return 'INCONCLUSIVE_TIMEOUT'
  if (terminal.class === 'adapter_error') return 'INVALID_INFRASTRUCTURE'
  if (terminal.class !== 'answered') return 'FAIL_AGENT_TERMINAL'
  if (repositoryOutcome === 'PASS_COMPARABLE' && comparability === 'noncomparable') return 'PASS_NONCOMPARABLE'
  return repositoryOutcome
}

function comparabilityFromTrace(trace) {
  if (!trace) return 'noncomparable'
  const used = trace.audit_events
    .filter((event) => event.event === 'agent.tool_call')
    .map((event) => event.tool)
  return used.every((tool) => COMMON_NATIVE_TOOLS.has(tool)) ? 'comparable' : 'noncomparable'
}

export function isolatedNativeEnv(source) {
  const allowed = new Set([
    'APPDATA',
    'HOME',
    'LOCALAPPDATA',
    'PATH',
    'PATHEXT',
    'PROGRAMDATA',
    'SYSTEMROOT',
    'TEMP',
    'TMP',
    'TMPDIR',
    'USERPROFILE',
    'WINDIR',
  ])
  return Object.fromEntries(Object.entries(source).filter(([key]) => allowed.has(key.toUpperCase())))
}

async function readNativeTrace(path, execution) {
  if (execution.timedOut || execution.state !== 'exited') return { trace: null, error: null }
  try {
    const trace = validateAgentExecTrace(JSON.parse(await readFile(path, 'utf8')))
    if (trace.terminal.exit_code !== execution.exitCode) {
      return { trace: null, error: `trace exit ${trace.terminal.exit_code} does not match process exit ${execution.exitCode}` }
    }
    return { trace, error: null }
  } catch (error) {
    return { trace: null, error: `native trace unavailable or invalid: ${error.message}` }
  }
}

function usageFromTrace(trace) {
  if (!trace) {
    return {
      model_steps: null,
      tool_calls: null,
      input_tokens: null,
      output_tokens: null,
      unavailable_reason: 'native trace unavailable',
    }
  }
  const contextsKnown = trace.steps.every((step) => step.context !== null)
  return {
    model_steps: trace.summary.model_steps,
    tool_calls: trace.summary.tool_calls,
    input_tokens: contextsKnown
      ? trace.steps.reduce((sum, step) => sum + step.context.prompt_tokens, 0)
      : null,
    output_tokens: trace.summary.output_tokens,
    unavailable_reason: contextsKnown && trace.summary.output_tokens !== null
      ? null
      : 'one or more native model steps did not report token metrics',
  }
}

function timingFromTrace(trace, wallMs) {
  return {
    wall_ms: wallMs,
    model_ms: trace?.summary.model_ms ?? null,
    ttft_ms: trace?.steps[0]?.ttft_ms ?? null,
  }
}

async function verifyWslBoundary(boundary) {
  const gpuDeviceArgs = boundary.gpuEnabled ? ['--dev-bind', '/dev/dxg', '/dev/dxg'] : []
  const gpuEnvironmentArgs = boundary.gpuEnabled
    ? ['--setenv', 'LD_LIBRARY_PATH', '/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu']
    : []
  const gpuProbe = boundary.gpuEnabled
    ? ' && /usr/lib/wsl/lib/nvidia-smi -L >/dev/null 2>&1'
    : ''
  const execution = await runProcess({
    file: boundary.wslExecutable,
    args: [
      '-d', boundary.distribution, '--',
      'bwrap',
      '--unshare-all',
      '--new-session',
      '--die-with-parent',
      '--ro-bind', '/usr', '/usr',
      '--ro-bind', '/lib', '/lib',
      '--ro-bind', '/lib64', '/lib64',
      '--ro-bind', '/bin', '/bin',
      '--proc', '/proc',
      '--dev', '/dev',
      ...gpuDeviceArgs,
      '--tmpfs', '/tmp',
      '--clearenv',
      '--setenv', 'PATH', '/usr/bin:/bin',
      ...gpuEnvironmentArgs,
      '/bin/sh', '-c',
      `test ! -e /mnt/c && ! /usr/bin/curl -fsS --max-time 2 https://example.com >/dev/null 2>&1${gpuProbe} && printf BOUNDARY_OK`,
    ],
    env: isolatedNativeEnv(process.env),
    timeoutMs: 10000,
  })
  if (!processSucceeded(execution) || execution.stdout.preview !== 'BOUNDARY_OK') {
    throw new NativeAdapterError('INVALID_INFRASTRUCTURE', `WSL bubblewrap boundary preflight failed: ${execution.stderr.preview || execution.stdout.preview || execution.state}`)
  }
}

async function verifyWslIdentity(boundary, binarySha256, modelSha256) {
  const execution = await runProcess({
    file: boundary.wslExecutable,
    args: [
      '-d', boundary.distribution, '--',
      'sha256sum', '--', boundary.linuxBinaryPath, boundary.linuxModelPath,
    ],
    env: isolatedNativeEnv(process.env),
    timeoutMs: 300000,
  })
  if (!processSucceeded(execution) || execution.stdout.truncated) {
    throw new NativeAdapterError('INVALID_INFRASTRUCTURE', `could not hash WSL candidate/model inputs: ${execution.stderr.preview || execution.state}`)
  }
  const hashes = execution.stdout.preview.trim().split(/\r?\n/).map((line) => line.split(/\s+/, 1)[0])
  if (hashes.length !== 2 || hashes[0] !== binarySha256 || hashes[1] !== modelSha256) {
    throw new NativeAdapterError('INVALID_INFRASTRUCTURE', 'Windows-visible and WSL-invoked candidate/model bytes do not match')
  }
}

function processSucceeded(execution) {
  return execution.state === 'exited'
    && execution.exitCode === 0
    && execution.timedOut === false
    && execution.cleanupPassed === true
}

function optionalPositiveInteger(value, name) {
  if (value === undefined || value === null) return null
  if (!Number.isSafeInteger(value) || value < 1) throw new TypeError(`${name} must be a positive safe integer`)
  return value
}