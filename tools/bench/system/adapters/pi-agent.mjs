import { mkdir, stat, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'

import { validateAgentAttempt } from '../lib/contracts.mjs'
import { controllerManifest } from '../lib/controller-manifest.mjs'
import { canonicalJson, sha256Bytes, sha256File } from '../lib/digest.mjs'
import { runProcess } from '../process/runner.mjs'
import { loadTaskPackage, materializeTask, scoreTaskAttempt } from '../tasks/package.mjs'
import { isolatedNativeEnv, linuxModelSandboxPath, windowsPathToWsl } from './native-camelid.mjs'
import {
  parsePiJsonFile,
  piJsonArgs,
  PINNED_PI_RELEASE,
  piProviderConfig,
  piSandboxEnvironment,
  PI_SHARED_TOOLS,
} from './pi-camelid.mjs'

const systemRoot = resolve(import.meta.dirname, '..')
const sharedTools = new Set(PI_SHARED_TOOLS)
const supervisorPath = resolve(systemRoot, 'process/pi-camelid-supervisor.sh')
const providerExtensionPath = resolve(systemRoot, 'adapters/pi-camelid-provider.mjs')
const SANDBOX_ADDR = '127.0.0.1:8231'
export const PI_SANDBOX_BASE_URL = `http://${SANDBOX_ADDR}/v1`

export async function runPiAgentAttempt(options) {
  const normalized = validateOptions(options)
  const taskPackage = await loadTaskPackage(normalized.taskRoot)
  if (taskPackage.task.network !== 'deny') {
    throw new PiAdapterError('INVALID_FIXTURE', 'shared Pi tasks must deny network access')
  }
  if (normalized.timeoutMs > taskPackage.task.budgets.wall_ms) {
    throw new PiAdapterError('INVALID_FIXTURE', 'Pi adapter timeout cannot exceed the task wall budget')
  }
  const maxOutputTokensPerStep = normalized.maxOutputTokensPerStep
    ?? taskPackage.task.budgets.max_output_tokens_per_step
  if (maxOutputTokensPerStep > taskPackage.task.budgets.max_output_tokens_per_step) {
    throw new PiAdapterError('INVALID_FIXTURE', 'Pi adapter token cap cannot exceed the task token budget')
  }
  if (maxOutputTokensPerStep > normalized.contextWindow) {
    throw new PiAdapterError('INVALID_FIXTURE', 'Pi adapter token cap cannot exceed the declared context window')
  }

  const [piExecutableSha256, binarySha256, modelSha256, supervisorSha256] = await Promise.all([
    sha256File(normalized.piExecutablePath),
    sha256File(normalized.binaryPath),
    sha256File(normalized.modelPath),
    sha256File(supervisorPath),
  ])
  let piArchiveSha256 = null
  if (normalized.boundary.kind === 'wsl-bwrap') {
    if (piExecutableSha256 !== PINNED_PI_RELEASE.executableSha256) {
      throw new PiAdapterError('INVALID_INFRASTRUCTURE', `Pi executable digest is ${piExecutableSha256}, expected ${PINNED_PI_RELEASE.executableSha256}`)
    }
    piArchiveSha256 = await verifyPinnedPiArchive(normalized.piArchivePath)
    await verifyPiWslBoundary(normalized.boundary)
    await verifyPiWslIdentity(normalized.boundary, {
      archive: piArchiveSha256,
      piExecutable: piExecutableSha256,
      binary: binarySha256,
      model: modelSha256,
    })
  }
  const controller = await controllerManifest(systemRoot)
  const materialized = await materializeTask(taskPackage, normalized.workspaceRoot)
  const controlRoot = join(materialized.workspaceRoot, 'control')
  const piAgentDir = join(controlRoot, 'pi-agent')
  const piHome = join(controlRoot, 'home')
  const eventsPath = join(controlRoot, 'events.jsonl')
  await mkdir(piAgentDir, { recursive: true })
  await mkdir(piHome, { recursive: true })

  const config = piProviderConfig({
    baseUrl: normalized.baseUrl,
    modelId: normalized.modelId,
    contextWindow: normalized.contextWindow,
    maxTokens: maxOutputTokensPerStep,
  })
  const configBytes = Buffer.from(canonicalJson(config), 'utf8')
  const configPath = join(piAgentDir, 'models.json')
  await writeFile(configPath, configBytes, { flag: 'wx' })
  const configSha256 = sha256Bytes(configBytes)
  const args = piJsonArgs({ modelId: normalized.modelId, goal: taskPackage.task.goal })
  const launch = prepareLaunch(normalized, materialized.attemptRoot, configPath, args, modelSha256)

  const startedAt = performance.now()
  const execution = await runProcess({
    file: launch.file,
    args: launch.args,
    cwd: materialized.attemptRoot,
    env: launch.env,
    timeoutMs: normalized.timeoutMs,
    stdoutFile: eventsPath,
  })
  const wallMs = performance.now() - startedAt

  const eventState = await readPiEvents(eventsPath, execution)
  const finalConfigSha256 = await sha256File(configPath).catch(() => null)
  if (finalConfigSha256 !== configSha256) {
    eventState.error = `Pi provider config changed during execution: expected ${configSha256}, found ${finalConfigSha256}`
  }

  const repositoryScore = await scoreTaskAttempt(normalized.taskRoot, materialized.workspaceRoot)
  const terminal = terminalFromExecution(execution, eventState.parsed, eventState.error)
  const comparability = comparabilityFromEvents(eventState.parsed)
  const outcome = attemptOutcome(terminal, execution.cleanupPassed, repositoryScore.outcome, eventState.error, comparability)
  const usage = usageFromEvents(eventState.parsed)
  const attemptRecord = validateAgentAttempt({
    schema: 'camelid.benchmark.agent-attempt/v1',
    campaign_id: normalized.campaignId,
    task_id: taskPackage.task.id,
    adapter: 'pi',
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
    timing: {
      wall_ms: wallMs,
      model_ms: null,
      ttft_ms: null,
    },
    process: {
      cleanup_passed: execution.cleanupPassed,
    },
  })

  return {
    identity: {
      source_sha: normalized.sourceSha,
      binary_sha256: binarySha256,
      model_sha256: modelSha256,
      pi_version: PINNED_PI_RELEASE.version,
      pi_source_commit: PINNED_PI_RELEASE.sourceCommit,
      pi_archive_sha256: piArchiveSha256,
      pi_executable_sha256: piExecutableSha256,
      pi_config_sha256: configSha256,
      pi_supervisor_sha256: supervisorSha256,
      controller_manifest_sha256: controller.sha256,
      task_definition_sha256: taskPackage.taskDefinitionSha256,
      fixture_manifest_sha256: taskPackage.fixture.sha256,
      scorer_manifest_sha256: taskPackage.scorer.sha256,
    },
    boundary: normalized.boundary.kind,
    gpu_enabled: normalized.boundary.gpuEnabled,
    tool_profile: 'benchmark_shared',
    max_output_tokens_per_step: maxOutputTokensPerStep,
    address: normalized.baseUrl,
    args,
    execution,
    pi_config: config,
    events: eventState.parsed,
    event_error: eventState.error,
    repository_score: repositoryScore,
    attempt: attemptRecord,
    workspace_root: materialized.workspaceRoot,
    attempt_root: materialized.attemptRoot,
  }
}

export class PiAdapterError extends Error {
  constructor(outcome, message) {
    super(message)
    this.name = 'PiAdapterError'
    this.outcome = outcome
  }
}

function prepareLaunch(options, attemptRoot, configPath, piArgs, modelSha256) {
  if (options.boundary.kind === 'synthetic') {
    return {
      file: options.piExecutablePath,
      args: [...options.syntheticCandidatePrefix, ...piArgs],
      env: piSandboxEnvironment({
        apiKey: 'camelid-benchmark-local',
        agentDir: join(options.workspaceRoot, 'control/pi-agent'),
        home: join(options.workspaceRoot, 'control/home'),
        path: environmentValue(options.env, 'PATH') ?? '',
        systemRoot: environmentValue(options.env, 'SYSTEMROOT'),
        temp: environmentValue(options.env, 'TEMP') ?? environmentValue(options.env, 'TMP'),
      }),
    }
  }
  return {
    file: options.boundary.wslExecutable,
    args: piWslBwrapPrefix({
      distribution: options.boundary.distribution,
      linuxPiDirPath: options.boundary.linuxPiDirPath,
      linuxBinaryPath: options.boundary.linuxBinaryPath,
      linuxModelPath: options.boundary.linuxModelPath,
      linuxAttemptPath: windowsPathToWsl(attemptRoot),
      linuxConfigPath: windowsPathToWsl(configPath),
      linuxSupervisorPath: windowsPathToWsl(supervisorPath),
      linuxProviderExtensionPath: windowsPathToWsl(providerExtensionPath),
      modelId: options.modelId,
      contextWindow: options.contextWindow,
      sourceSha: options.sourceSha,
      modelSha256,
      piArgs,
      readyTimeoutSeconds: Math.max(1, Math.ceil(options.timeoutMs / 1000)),
      gpuEnabled: options.boundary.gpuEnabled,
    }),
    env: isolatedNativeEnv(options.env),
  }
}

function validateOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('Pi adapter options must be an object')
  }
  for (const name of ['taskRoot', 'workspaceRoot', 'piExecutablePath', 'binaryPath', 'modelPath', 'modelId', 'baseUrl']) {
    if (typeof options[name] !== 'string' || options[name].length === 0) throw new TypeError(`${name} must be a non-empty string`)
  }
  if (typeof options.campaignId !== 'string' || options.campaignId.length === 0) throw new TypeError('campaignId must be non-empty')
  if (typeof options.sourceSha !== 'string' || !/^[0-9a-f]{40}$/.test(options.sourceSha)) throw new TypeError('sourceSha must be 40 lowercase hex characters')
  if (!Number.isSafeInteger(options.attempt) || options.attempt < 0) throw new TypeError('attempt must be a non-negative safe integer')
  if (!Number.isSafeInteger(options.timeoutMs) || options.timeoutMs < 1) throw new TypeError('timeoutMs must be a positive safe integer')
  if (!Number.isSafeInteger(options.contextWindow) || options.contextWindow < 1) throw new TypeError('contextWindow must be a positive safe integer')
  const boundary = validateBoundary(options)
  if (boundary.kind === 'wsl-bwrap' && options.baseUrl !== PI_SANDBOX_BASE_URL) {
    throw new PiAdapterError('INVALID_INFRASTRUCTURE', `WSL Pi baseUrl must be ${PI_SANDBOX_BASE_URL}`)
  }
  return {
    taskRoot: resolve(options.taskRoot),
    workspaceRoot: resolve(options.workspaceRoot),
    piExecutablePath: resolve(options.piExecutablePath),
    binaryPath: resolve(options.binaryPath),
    modelPath: resolve(options.modelPath),
    modelId: options.modelId,
    baseUrl: options.baseUrl,
    piArchivePath: options.piArchivePath ? resolve(options.piArchivePath) : null,
    campaignId: options.campaignId,
    sourceSha: options.sourceSha,
    attempt: options.attempt,
    timeoutMs: options.timeoutMs,
    contextWindow: options.contextWindow,
    maxOutputTokensPerStep: optionalPositiveInteger(options.maxOutputTokensPerStep, 'maxOutputTokensPerStep'),
    env: options.env ?? process.env,
    boundary,
    syntheticCandidatePrefix: options.syntheticCandidatePrefix ? [...options.syntheticCandidatePrefix] : [],
  }
}

function validateBoundary(options) {
  if (options.boundary === 'synthetic') {
    if (options.syntheticCandidate !== true) throw new PiAdapterError('INVALID_INFRASTRUCTURE', 'synthetic Pi boundary is test-only')
    if (!Array.isArray(options.syntheticCandidatePrefix)
      || options.syntheticCandidatePrefix.length === 0
      || options.syntheticCandidatePrefix.some((item) => typeof item !== 'string' || item.length === 0)) {
      throw new TypeError('syntheticCandidatePrefix must be a non-empty string array')
    }
    return { kind: 'synthetic', gpuEnabled: false }
  }
  const boundary = options.boundary
  if (boundary === null || typeof boundary !== 'object' || Array.isArray(boundary) || boundary.kind !== 'wsl-bwrap') {
    throw new PiAdapterError('INVALID_INFRASTRUCTURE', 'Pi execution requires a synthetic test boundary or WSL bubblewrap')
  }
  if (typeof options.piArchivePath !== 'string' || options.piArchivePath.length === 0) {
    throw new TypeError('piArchivePath must be a non-empty string for WSL execution')
  }
  if (boundary.gpuEnabled !== undefined && typeof boundary.gpuEnabled !== 'boolean') throw new TypeError('WSL gpuEnabled must be a boolean')
  if (typeof boundary.distribution !== 'string' || !/^[A-Za-z0-9_.-]+$/.test(boundary.distribution)) {
    throw new TypeError('WSL distribution must be a simple name')
  }
  for (const name of ['linuxPiDirPath', 'linuxPiArchivePath', 'linuxBinaryPath', 'linuxModelPath']) {
    if (typeof boundary[name] !== 'string' || !boundary[name].startsWith('/')) throw new TypeError(`${name} must be an absolute Linux path`)
  }
  const windowsSystemRoot = process.env.SYSTEMROOT ?? 'C:\\Windows'
  return {
    kind: 'wsl-bwrap',
    distribution: boundary.distribution,
    linuxPiDirPath: boundary.linuxPiDirPath,
    linuxPiArchivePath: boundary.linuxPiArchivePath,
    linuxBinaryPath: boundary.linuxBinaryPath,
    linuxModelPath: boundary.linuxModelPath,
    gpuEnabled: boundary.gpuEnabled ?? false,
    wslExecutable: boundary.wslExecutable
      ? resolve(boundary.wslExecutable)
      : process.platform === 'win32'
        ? resolve(windowsSystemRoot, 'System32', 'wsl.exe')
        : 'wsl',
  }
}

export function piWslBwrapPrefix({
  distribution,
  linuxPiDirPath,
  linuxBinaryPath,
  linuxModelPath,
  linuxAttemptPath,
  linuxConfigPath,
  linuxSupervisorPath,
  linuxProviderExtensionPath,
  modelId,
  contextWindow,
  sourceSha,
  modelSha256,
  piArgs,
  readyTimeoutSeconds,
  gpuEnabled = false,
}) {
  const sandboxModelPath = linuxModelSandboxPath(linuxModelPath)
  const gpuDeviceArgs = gpuEnabled ? [
    '--dev-bind-try', '/dev/dxg', '/dev/dxg',
    '--dev-bind-try', '/dev/nvidia0', '/dev/nvidia0',
    '--dev-bind-try', '/dev/nvidiactl', '/dev/nvidiactl',
    '--dev-bind-try', '/dev/nvidia-uvm', '/dev/nvidia-uvm',
  ] : []
  const gpuEnvironmentArgs = gpuEnabled
    ? ['--setenv', 'LD_LIBRARY_PATH', '/usr/lib/wsl/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu']
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
    '--dir', '/opt/controller',
    '--dir', '/model',
    '--dir', '/workspace',
    '--dir', '/tmp/home',
    '--dir', '/tmp/pi-agent',
    '--ro-bind', linuxPiDirPath, '/opt/pi',
    '--ro-bind', linuxBinaryPath, '/opt/camelid/camelid',
    '--ro-bind', linuxModelPath, sandboxModelPath,
    '--ro-bind', linuxSupervisorPath, '/opt/controller/pi-camelid-supervisor.sh',
    '--ro-bind', linuxProviderExtensionPath, '/opt/controller/pi-camelid-provider.mjs',
    '--ro-bind', linuxConfigPath, '/tmp/pi-agent/models.json',
    '--bind', linuxAttemptPath, '/workspace',
    '--chdir', '/workspace',
    '--clearenv',
    '--setenv', 'PATH', '/usr/bin:/bin',
    '--setenv', 'HOME', '/tmp/home',
    '--setenv', 'PI_CODING_AGENT_DIR', '/tmp/pi-agent',
    '--setenv', 'PI_OFFLINE', '1',
    '--setenv', 'PI_SKIP_VERSION_CHECK', '1',
    '--setenv', 'PI_TELEMETRY', '0',
    '--setenv', 'CAMELID_PI_API_KEY', 'camelid-benchmark-local',
    '--setenv', 'CAMELID_NO_REMOTE_DIMS', '1',
    '--setenv', 'CAMELID_PI_READY_TIMEOUT_SECONDS', String(readyTimeoutSeconds),
    ...gpuEnvironmentArgs,
    '/bin/sh', '/opt/controller/pi-camelid-supervisor.sh',
    '/opt/camelid/camelid', sandboxModelPath, SANDBOX_ADDR, modelId, String(contextWindow), sourceSha, modelSha256, '/opt/pi/pi',
    ...piArgs,
  ]
}

async function verifyPinnedPiArchive(path) {
  const info = await stat(path)
  if (!info.isFile() || info.size !== PINNED_PI_RELEASE.archiveSizeBytes) {
    throw new PiAdapterError('INVALID_INFRASTRUCTURE', `Pi archive must be ${PINNED_PI_RELEASE.archiveSizeBytes} bytes`)
  }
  const actual = await sha256File(path)
  if (actual !== PINNED_PI_RELEASE.archiveSha256) {
    throw new PiAdapterError('INVALID_INFRASTRUCTURE', `Pi archive digest is ${actual}, expected ${PINNED_PI_RELEASE.archiveSha256}`)
  }
  return actual
}

async function verifyPiWslBoundary(boundary) {
  const gpuDeviceArgs = boundary.gpuEnabled ? [
    '--dev-bind-try', '/dev/dxg', '/dev/dxg',
    '--dev-bind-try', '/dev/nvidia0', '/dev/nvidia0',
    '--dev-bind-try', '/dev/nvidiactl', '/dev/nvidiactl',
    '--dev-bind-try', '/dev/nvidia-uvm', '/dev/nvidia-uvm',
  ] : []
  const gpuEnvironmentArgs = boundary.gpuEnabled
    ? ['--setenv', 'LD_LIBRARY_PATH', '/usr/lib/wsl/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu']
    : []
  const command = piBoundaryProbeCommand(boundary.gpuEnabled)
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
      '--ro-bind', '/sys', '/sys',
      '--proc', '/proc',
      '--dev', '/dev',
      ...gpuDeviceArgs,
      '--tmpfs', '/tmp',
      '--dir', '/opt',
      '--dir', '/opt/pi',
      '--ro-bind', boundary.linuxPiDirPath, '/opt/pi',
      '--chdir', '/tmp',
      '--clearenv',
      '--setenv', 'PATH', '/usr/bin:/bin',
      '--setenv', 'HOME', '/tmp',
      '--setenv', 'PI_CODING_AGENT_DIR', '/tmp/pi-agent',
      '--setenv', 'PI_OFFLINE', '1',
      '--setenv', 'PI_TELEMETRY', '0',
      ...gpuEnvironmentArgs,
      '/bin/sh', '-c', command,
    ],
    env: isolatedNativeEnv(process.env),
    timeoutMs: 20000,
  })
  if (!processSucceeded(execution) || execution.stdout.preview.trim() !== PINNED_PI_RELEASE.version) {
    throw new PiAdapterError('INVALID_INFRASTRUCTURE', `Pi WSL boundary preflight failed: ${execution.stderr.preview || execution.stdout.preview || execution.state}`)
  }
}

export function piBoundaryProbeCommand(gpuEnabled) {
  const gpuProbe = gpuEnabled ? ' && (/usr/lib/wsl/lib/nvidia-smi -L || /usr/bin/nvidia-smi -L || nvidia-smi -L) >/dev/null 2>&1' : ''
  return `test ! -e /mnt/c && test -x /usr/bin/find && test -x /usr/bin/grep && ! /usr/bin/curl -fsS --max-time 2 https://example.com >/dev/null 2>&1${gpuProbe} && /opt/pi/pi --version`
}

async function verifyPiWslIdentity(boundary, expected) {
  const execution = await runProcess({
    file: boundary.wslExecutable,
    args: [
      '-d', boundary.distribution, '--',
      'sha256sum', '--',
      boundary.linuxPiArchivePath,
      `${boundary.linuxPiDirPath}/pi`,
      boundary.linuxBinaryPath,
      boundary.linuxModelPath,
    ],
    env: isolatedNativeEnv(process.env),
    timeoutMs: 300000,
  })
  if (!processSucceeded(execution) || execution.stdout.truncated) {
    throw new PiAdapterError('INVALID_INFRASTRUCTURE', `could not hash WSL Pi/Camelid/model inputs: ${execution.stderr.preview || execution.state}`)
  }
  const hashes = execution.stdout.preview.trim().split(/\r?\n/).map((line) => line.split(/\s+/, 1)[0])
  const wanted = [expected.archive, expected.piExecutable, expected.binary, expected.model]
  if (hashes.length !== wanted.length || hashes.some((hash, index) => hash !== wanted[index])) {
    throw new PiAdapterError('INVALID_INFRASTRUCTURE', 'Windows-visible and WSL-invoked Pi/Camelid/model bytes do not match')
  }
}

function processSucceeded(execution) {
  return execution.state === 'exited'
    && execution.exitCode === 0
    && execution.timedOut === false
    && execution.cleanupPassed === true
}

async function readPiEvents(path, execution) {
  if (execution.timedOut || execution.state !== 'exited') return { parsed: null, error: null }
  try {
    return { parsed: await parsePiJsonFile(path), error: null }
  } catch (error) {
    return { parsed: null, error: `Pi event stream unavailable or invalid: ${error.message}` }
  }
}

function terminalFromExecution(execution, parsed, eventError) {
  if (execution.timedOut) return { class: 'timed_out', exit_code: execution.exitCode, reason: 'Pi agent wall timeout expired' }
  if (execution.state !== 'exited') return { class: 'adapter_error', exit_code: execution.exitCode, reason: execution.error ?? execution.state }
  if (eventError || !parsed) return { class: 'adapter_error', exit_code: execution.exitCode, reason: eventError ?? 'Pi event stream missing' }
  const finalMessage = [...parsed.events].reverse().find((event) => event.type === 'message_end' && event.message?.role === 'assistant')?.message
  if (finalMessage?.stopReason === 'aborted') return { class: 'cancelled', exit_code: execution.exitCode, reason: 'Pi reported an aborted assistant message' }
  if (finalMessage?.stopReason === 'error' || execution.exitCode !== 0) {
    return { class: 'failed', exit_code: execution.exitCode, reason: finalMessage?.errorMessage ?? `Pi exited ${execution.exitCode}` }
  }
  return { class: 'answered', exit_code: 0, reason: `Pi JSON stream completed with ${parsed.summary.model_steps} assistant message(s)` }
}

function attemptOutcome(terminal, cleanupPassed, repositoryOutcome, eventError, comparability) {
  if (!cleanupPassed || eventError) return 'INVALID_INFRASTRUCTURE'
  if (terminal.class === 'timed_out') return 'INCONCLUSIVE_TIMEOUT'
  if (terminal.class === 'adapter_error') return 'INVALID_INFRASTRUCTURE'
  if (terminal.class !== 'answered') return 'FAIL_AGENT_TERMINAL'
  if (repositoryOutcome === 'PASS_COMPARABLE' && comparability === 'noncomparable') return 'PASS_NONCOMPARABLE'
  return repositoryOutcome
}

function comparabilityFromEvents(parsed) {
  if (!parsed) return 'noncomparable'
  const tools = parsed.events
    .filter((event) => event.type === 'tool_execution_start')
    .map((event) => event.toolName)
  return tools.every((tool) => sharedTools.has(tool)) ? 'comparable' : 'noncomparable'
}

function usageFromEvents(parsed) {
  if (!parsed) {
    return {
      model_steps: null,
      tool_calls: null,
      input_tokens: null,
      output_tokens: null,
      unavailable_reason: 'Pi event stream unavailable',
    }
  }
  const tokensKnown = parsed.summary.input_tokens !== null && parsed.summary.output_tokens !== null
  return {
    model_steps: parsed.summary.model_steps,
    tool_calls: parsed.summary.tool_calls,
    input_tokens: parsed.summary.input_tokens,
    output_tokens: parsed.summary.output_tokens,
    unavailable_reason: tokensKnown ? null : 'Pi provider did not report token usage',
  }
}

function optionalPositiveInteger(value, name) {
  if (value === undefined || value === null) return null
  if (!Number.isSafeInteger(value) || value < 1) throw new TypeError(`${name} must be a positive safe integer`)
  return value
}

function environmentValue(environment, name) {
  const match = Object.entries(environment).find(([key]) => key.toUpperCase() === name)
  return match?.[1] ?? null
}