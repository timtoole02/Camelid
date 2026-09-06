import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'

import { buildComparison } from './aggregate.mjs'
import {
  validateAgentAttempt,
  validateAgentExecTrace,
  validatePlan,
  validateRuntimeSample,
} from './lib/contracts.mjs'
import { canonicalJson, sha256File } from './lib/digest.mjs'

export async function writeBenchmarkBundle(input, options = {}) {
  const { plan, preparedArms, samples, executions } = input
  const preparationModes = new Set(['built_from_plan', 'supplied_local_ablation'])
  if (!preparationModes.has(input.preparationMode)) {
    throw new TypeError(`preparationMode must be one of ${[...preparationModes].join(', ')}`)
  }
  validatePlan(plan)
  samples.forEach(validateRuntimeSample)
  const outputDir = resolve(input.outputDir)
  const generatedUtc = options.generatedUtc ?? new Date().toISOString()
  const comparison = buildComparison(plan, samples, options.stats)
  await mkdir(outputDir, { recursive: true })
  await writeCanonical(join(outputDir, 'plan.json'), plan)
  await writeCanonical(join(outputDir, 'prepared-arms.json'), preparedArms)
  await writeCanonical(join(outputDir, 'executions.json'), executions)

  const sampleFiles = []
  for (const sample of samples) {
    const path = join(
      outputDir,
      'runtime',
      sample.workload_id,
      sample.arm_id,
      `block-${String(sample.process_block).padStart(3, '0')}.json`,
    )
    await writeCanonical(path, sample)
    sampleFiles.push(relativePath(outputDir, path))
  }
  await writeCanonical(join(outputDir, 'comparison.json'), comparison)

  const invalidSamples = samples.filter((sample) => sample.validity !== 'valid')
  const manifest = {
    schema: 'camelid.benchmark.bundle/v1',
    campaign_id: plan.campaign_id,
    generated_utc: generatedUtc,
    state: invalidSamples.length === 0 ? 'COMPLETE_VALID' : 'COMPLETE_WITH_FINDINGS',
    plan_sha256: await sha256File(join(outputDir, 'plan.json')),
    prepared_arm_count: preparedArms.length,
    sample_count: samples.length,
    valid_sample_count: samples.length - invalidSamples.length,
    invalid_sample_count: invalidSamples.length,
    sample_files: sampleFiles.sort(),
    raw_execution_count: executions.length,
    executions_file: 'executions.json',
    preparation_mode: input.preparationMode,
    comparison_file: 'comparison.json',
    claim_boundary: 'Local informational Phase 1 runtime comparison only; no numeric gate or public performance claim.',
  }
  await writeCanonical(join(outputDir, 'manifest.json'), manifest)
  await writeFile(join(outputDir, 'summary.md'), renderSummary(manifest, comparison), 'utf8')
  await writeChecksums(outputDir)
  return { manifest, comparison, outputDir }
}

export async function verifyBundleChecksums(outputDir) {
  const root = resolve(outputDir)
  const text = await readFile(join(root, 'SHA256SUMS'), 'utf8')
  const failures = []
  const expectedFiles = new Set()
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (line.length === 0) continue
    const match = line.match(/^([0-9a-f]{64})  (.+)$/)
    if (!match) {
      failures.push(`line ${index + 1} is malformed`)
      continue
    }
    const [, expected, relativeFile] = match
    if (!isSafeChecksumPath(relativeFile)) {
      failures.push(`line ${index + 1} has an unsafe path: ${relativeFile}`)
      continue
    }
    if (expectedFiles.has(relativeFile)) {
      failures.push(`line ${index + 1} duplicates ${relativeFile}`)
      continue
    }
    expectedFiles.add(relativeFile)
    let actual
    try {
      actual = await sha256File(join(root, ...relativeFile.split('/')))
    } catch (error) {
      failures.push(`${relativeFile}: ${error.message}`)
      continue
    }
    if (actual !== expected) failures.push(`${relativeFile}: expected ${expected}, got ${actual}`)
  }
  let actualFiles = []
  try {
    actualFiles = (await walk(root))
      .map((path) => relativePath(root, path))
      .filter((path) => path !== 'SHA256SUMS')
  } catch (error) {
    failures.push(error.message)
  }
  for (const actualFile of actualFiles) {
    if (!expectedFiles.has(actualFile)) failures.push(`${actualFile}: not listed in SHA256SUMS`)
  }
  return { ok: failures.length === 0, failures }
}

export async function writeNativeAgentBundle(input, options = {}) {
  const outputDir = resolve(input.outputDir)
  await requireEmptyDirectory(outputDir)
  validateAgentAttempt(input.result.attempt)
  if (input.result.trace !== null) validateAgentExecTrace(input.result.trace)
  await mkdir(outputDir, { recursive: true })
  await writeCanonical(join(outputDir, 'attempt.json'), input.result.attempt)
  await writeCanonical(join(outputDir, 'score.json'), input.result.repository_score)
  await writeCanonical(join(outputDir, 'identity.json'), input.result.identity)
  await writeCanonical(join(outputDir, 'adapter.json'), {
    boundary: input.result.boundary,
    gpu_enabled: input.result.gpu_enabled,
    checkpoints_enabled: input.result.checkpoints_enabled,
    tool_profile: input.result.tool_profile,
    max_output_tokens_per_step: input.result.max_output_tokens_per_step,
    address: input.result.address,
    trace_error: input.result.trace_error,
  })
  if (input.result.trace !== null) await writeCanonical(join(outputDir, 'trace.json'), input.result.trace)
  await writeCanonical(join(outputDir, 'execution.json'), boundedExecution(input.result.execution))

  const manifest = {
    schema: 'camelid.benchmark.native-bundle/v1',
    campaign_id: input.result.attempt.campaign_id,
    task_id: input.result.attempt.task_id,
    generated_utc: options.generatedUtc ?? new Date().toISOString(),
    state: input.result.attempt.score.outcome.startsWith('INVALID_')
      ? 'INCOMPLETE'
      : 'COMPLETE',
    outcome: input.result.attempt.score.outcome,
    terminal_class: input.result.attempt.terminal.class,
    cleanup_passed: input.result.attempt.process.cleanup_passed,
    files: {
      attempt: 'attempt.json',
      score: 'score.json',
      identity: 'identity.json',
      adapter: 'adapter.json',
      execution: 'execution.json',
      trace: input.result.trace === null ? null : 'trace.json',
    },
    workspace_included: false,
    boundary: input.result.boundary,
    gpu_enabled: input.result.gpu_enabled,
    checkpoints_enabled: input.result.checkpoints_enabled,
    tool_profile: input.result.tool_profile,
    max_output_tokens_per_step: input.result.max_output_tokens_per_step,
    claim_boundary: 'Local Phase 3 native-agent evidence only; scorer outcome is authoritative and no public model-quality claim is made.',
  }
  await writeCanonical(join(outputDir, 'manifest.json'), manifest)
  await writeFile(join(outputDir, 'summary.md'), renderNativeSummary(manifest), 'utf8')
  await writeChecksums(outputDir)
  return { outputDir, manifest }
}

export async function writePiAgentBundle(input, options = {}) {
  const outputDir = resolve(input.outputDir)
  await requireEmptyDirectory(outputDir)
  validateAgentAttempt(input.result.attempt)
  if (input.result.attempt.adapter !== 'pi') throw new TypeError('Pi bundle requires a Pi agent attempt')
  if (input.result.events !== null
    && (!Array.isArray(input.result.events.events) || input.result.events.session?.version !== 3)) {
    throw new TypeError('Pi bundle events must be a parsed Pi JSON v3 stream or null')
  }
  await mkdir(outputDir, { recursive: true })
  await writeCanonical(join(outputDir, 'attempt.json'), input.result.attempt)
  await writeCanonical(join(outputDir, 'score.json'), input.result.repository_score)
  await writeCanonical(join(outputDir, 'identity.json'), input.result.identity)
  await writeCanonical(join(outputDir, 'models.json'), input.result.pi_config)
  const configSha256 = await sha256File(join(outputDir, 'models.json'))
  if (configSha256 !== input.result.identity.pi_config_sha256) {
    throw new TypeError(`Pi config digest is ${configSha256}, identity pins ${input.result.identity.pi_config_sha256}`)
  }
  await writeCanonical(join(outputDir, 'adapter.json'), {
    boundary: input.result.boundary,
    gpu_enabled: input.result.gpu_enabled,
    tool_profile: input.result.tool_profile,
    max_output_tokens_per_step: input.result.max_output_tokens_per_step,
    address: input.result.address,
    event_error: input.result.event_error,
  })
  if (input.result.events !== null) await writeCanonical(join(outputDir, 'events.json'), input.result.events)
  await writeCanonical(join(outputDir, 'execution.json'), boundedExecution(input.result.execution))

  const manifest = {
    schema: 'camelid.benchmark.pi-bundle/v1',
    campaign_id: input.result.attempt.campaign_id,
    task_id: input.result.attempt.task_id,
    generated_utc: options.generatedUtc ?? new Date().toISOString(),
    state: input.result.attempt.score.outcome.startsWith('INVALID_')
      ? 'INCOMPLETE'
      : 'COMPLETE',
    outcome: input.result.attempt.score.outcome,
    terminal_class: input.result.attempt.terminal.class,
    cleanup_passed: input.result.attempt.process.cleanup_passed,
    files: {
      attempt: 'attempt.json',
      score: 'score.json',
      identity: 'identity.json',
      config: 'models.json',
      adapter: 'adapter.json',
      execution: 'execution.json',
      events: input.result.events === null ? null : 'events.json',
    },
    workspace_included: false,
    boundary: input.result.boundary,
    gpu_enabled: input.result.gpu_enabled,
    tool_profile: input.result.tool_profile,
    max_output_tokens_per_step: input.result.max_output_tokens_per_step,
    claim_boundary: 'Local Phase 4 Pi-through-Camelid evidence only; scorer outcome is authoritative and no public model-quality or cross-agent claim is made.',
  }
  await writeCanonical(join(outputDir, 'manifest.json'), manifest)
  await writeFile(join(outputDir, 'summary.md'), renderPiSummary(manifest), 'utf8')
  await writeChecksums(outputDir)
  return { outputDir, manifest }
}

function boundedExecution(execution) {
  return {
    state: execution.state,
    exit_code: execution.exitCode,
    signal: execution.signal,
    timed_out: execution.timedOut,
    duration_ms: execution.durationMs,
    cleanup_passed: execution.cleanupPassed,
    cleanup_detail: execution.cleanupDetail,
    error: execution.error,
    stdout: execution.stdout,
    stderr: execution.stderr,
  }
}

function renderNativeSummary(manifest) {
  return [
    '# Camelid Phase 3 Native Agent Attempt',
    '',
    `- Campaign: \`${manifest.campaign_id}\``,
    `- Task: \`${manifest.task_id}\``,
    `- State: **${manifest.state}**`,
    `- Terminal: \`${manifest.terminal_class}\``,
    `- Scorer outcome: **${manifest.outcome}**`,
    `- Cleanup passed: ${manifest.cleanup_passed}`,
    `- Boundary: \`${manifest.boundary}\``,
    `- Workspace included: ${manifest.workspace_included}`,
    `- Claim boundary: ${manifest.claim_boundary}`,
    '',
  ].join('\n')
}

function renderPiSummary(manifest) {
  return [
    '# Camelid Phase 4 Pi Agent Attempt',
    '',
    `- Campaign: \`${manifest.campaign_id}\``,
    `- Task: \`${manifest.task_id}\``,
    `- State: **${manifest.state}**`,
    `- Terminal: \`${manifest.terminal_class}\``,
    `- Scorer outcome: **${manifest.outcome}**`,
    `- Cleanup passed: ${manifest.cleanup_passed}`,
    `- Boundary: \`${manifest.boundary}\``,
    `- Workspace included: ${manifest.workspace_included}`,
    `- Claim boundary: ${manifest.claim_boundary}`,
    '',
  ].join('\n')
}

async function requireEmptyDirectory(path) {
  try {
    const info = await stat(path)
    if (!info.isDirectory()) throw new Error(`agent bundle output exists and is not a directory: ${path}`)
    const entries = await readdir(path)
    if (entries.length > 0) throw new Error(`agent bundle output directory is not empty: ${path}`)
  } catch (error) {
    if (error.code === 'ENOENT') return
    throw error
  }
}

async function writeChecksums(outputDir) {
  const sumsPath = join(outputDir, 'SHA256SUMS')
  await rm(sumsPath, { force: true })
  const files = await walk(outputDir)
  const lines = []
  for (const path of files) {
    const relativeFile = relativePath(outputDir, path)
    lines.push(`${await sha256File(path)}  ${relativeFile}`)
  }
  await writeFile(sumsPath, `${lines.join('\n')}\n`, 'utf8')
}

async function walk(directory) {
  const files = []
  const entries = await readdir(directory, { withFileTypes: true })
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) files.push(...await walk(path))
    else if (entry.isFile()) files.push(path)
    else throw new Error(`bundle contains unsupported entry: ${path}`)
  }
  return files
}

function isSafeChecksumPath(path) {
  if (path === 'SHA256SUMS' || path.includes('\\') || isAbsolute(path)) return false
  return path.split('/').every((part) => part.length > 0 && part !== '.' && part !== '..')
}

async function writeCanonical(path, value) {
  await mkdir(resolve(path, '..'), { recursive: true })
  await writeFile(path, canonicalJson(value), 'utf8')
}

function renderSummary(manifest, comparison) {
  const lines = [
    '# Camelid Phase 1 Runtime Benchmark',
    '',
    `- Campaign: \`${manifest.campaign_id}\``,
    `- State: **${manifest.state}**`,
    `- Samples: ${manifest.valid_sample_count} valid / ${manifest.invalid_sample_count} invalid`,
    `- Claim boundary: ${manifest.claim_boundary}`,
    '',
    '| Workload | Metric | Valid pairs | Excluded | Base median | Head median | Head/base | 95% bootstrap CI | Direction | Verdict |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |',
  ]
  for (const row of comparison.runtime) {
    lines.push(`| ${cell(row.workload_id)} | ${cell(row.metric)} | ${row.valid_pairs} | ${row.excluded_pairs.length} | ${number(row.base_median)} | ${number(row.head_median)} | ${number(row.median_ratio_head_over_base)} | ${row.bootstrap_ci95 ? row.bootstrap_ci95.map(number).join(' - ') : '-'} | ${row.observed_direction} | ${row.verdict} |`)
  }
  lines.push('', 'Invalid and excluded samples remain in the machine-readable bundle.', '')
  return lines.join('\n')
}

function cell(value) {
  return String(value).replaceAll('|', '\\|')
}

function number(value) {
  return value === null ? '-' : Number(value).toFixed(6)
}

function relativePath(root, path) {
  return relative(root, path).split(sep).join('/')
}
