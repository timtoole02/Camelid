#!/usr/bin/env node
import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { basename, dirname, isAbsolute, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  writeBenchmarkBundle,
  writeNativeAgentBundle,
  writePiAgentBundle,
  verifyBundleChecksums,
} from './bundle.mjs'
import { runNativeAgentAttempt } from './adapters/native-camelid.mjs'
import { PI_SANDBOX_BASE_URL, runPiAgentAttempt } from './adapters/pi-agent.mjs'
import { controllerManifest } from './lib/controller-manifest.mjs'
import { canonicalJson } from './lib/digest.mjs'
import { acquireCampaignLock, assertMinimumFreeDisk } from './lib/safety.mjs'
import { resolveCampaignPlan, serializePlan } from './planner.mjs'
import { prepareArms } from './prepare.mjs'
import { runRuntimeCampaign } from './adapters/runtime-camelid.mjs'
import { loadTaskPackage, materializeTask, scoreTaskAttempt } from './tasks/package.mjs'

const systemRoot = resolve(fileURLToPath(new URL('.', import.meta.url)))
const args = parseArgs(process.argv.slice(2))
const command = args.positionals[0]
let activeRun = null

try {
  if (command === 'digest') {
    const controller = await controllerManifest(systemRoot)
    console.log(controller.sha256)
  } else if (command === 'plan') {
    const { plan } = await loadAndResolve(args)
    const text = serializePlan(plan)
    if (args.values.has('out')) await writeText(resolve(args.values.get('out')), text)
    process.stdout.write(text)
  } else if (command === 'run') {
    const { request, plan } = await loadAndResolve(args)
    const outRoot = resolve(args.values.get('out-root') ?? joinDefaultOutput(plan.repository_root))
    const outputDir = resolve(outRoot, plan.campaign_id)
    await mkdir(outRoot, { recursive: true })
    await assertMinimumFreeDisk(
      [outRoot, ...plan.source_arms.map((arm) => arm.build.target_dir)],
      plan.resources.minimum_free_disk_bytes,
    )
    const lock = await acquireCampaignLock(resolve(outRoot, '.camelid-benchmark.lock'), {
      campaignId: plan.campaign_id,
      createdUtc: plan.created_utc,
    })
    try {
      await requireNewDirectory(outputDir)
      await mkdir(outputDir, { recursive: true })
      activeRun = { outputDir, campaignId: plan.campaign_id }
      await writeText(resolve(outputDir, 'plan.json'), serializePlan(plan))

      let preparedArms
      let preparationMode
      if (args.values.has('prepared')) {
        if (request.mode !== 'ablation' || request.security.trust_tier !== 'local_exploratory') {
          throw new Error('--prepared is restricted to local_exploratory ablation campaigns')
        }
        preparedArms = JSON.parse(await readFile(resolve(args.values.get('prepared')), 'utf8'))
        preparationMode = 'supplied_local_ablation'
      } else {
        preparedArms = await prepareArms(plan, {
          logDir: resolve(outputDir, 'build'),
        })
        preparationMode = 'built_from_plan'
      }

      const runtime = await runRuntimeCampaign(plan, preparedArms, { outputDir })
      const bundle = await writeBenchmarkBundle({
        plan,
        preparedArms,
        samples: runtime.samples,
        executions: runtime.executions,
        outputDir,
        preparationMode,
      })
      const verification = await verifyBundleChecksums(outputDir)
      if (!verification.ok) throw new Error(`bundle checksum verification failed: ${verification.failures.join('; ')}`)
      console.log(`bundle_dir=${outputDir}`)
      console.log(`state=${bundle.manifest.state}`)
      activeRun = null
      if (bundle.manifest.state !== 'COMPLETE_VALID') process.exitCode = 1
    } finally {
      await lock.release()
    }
  } else if (command === 'task-verify') {
    const taskPath = requiredValue(args, 'task')
    const taskPackage = await loadTaskPackage(resolve(taskPath))
    process.stdout.write(canonicalJson({
      schema: 'camelid.benchmark.task-package-verification/v1',
      task_id: taskPackage.task.id,
      task_definition_sha256: taskPackage.taskDefinitionSha256,
      fixture_manifest_sha256: taskPackage.fixture.sha256,
      scorer_manifest_sha256: taskPackage.scorer.sha256,
      verified: true,
    }))
  } else if (command === 'task-materialize') {
    const taskPath = requiredValue(args, 'task')
    const workspacePath = requiredValue(args, 'workspace')
    const taskPackage = await loadTaskPackage(resolve(taskPath))
    const materialized = await materializeTask(taskPackage, resolve(workspacePath))
    process.stdout.write(canonicalJson({
      schema: 'camelid.benchmark.task-materialization/v1',
      task_id: taskPackage.task.id,
      workspace_root: materialized.workspaceRoot,
      attempt_root: materialized.attemptRoot,
      setup_passed: true,
    }))
  } else if (command === 'task-score') {
    const taskPath = resolve(requiredValue(args, 'task'))
    const workspacePath = resolve(requiredValue(args, 'workspace'))
    const result = await scoreTaskAttempt(taskPath, workspacePath)
    let taskId = basename(taskPath)
    try {
      taskId = JSON.parse(await readFile(resolve(taskPath, 'task.json'), 'utf8')).id ?? taskId
    } catch {
      // The invalid score remains the authority when the manifest itself is unreadable.
    }
    const record = canonicalJson({
      schema: 'camelid.benchmark.task-score/v1',
      task_id: taskId,
      ...result,
    })
    if (args.values.has('out')) await writeText(resolve(args.values.get('out')), record)
    process.stdout.write(record)
    if (result.outcome.startsWith('INVALID_')) process.exitCode = 1
  } else if (command === 'native-run') {
    const workspaceRoot = resolve(requiredValue(args, 'workspace'))
    const outputDir = resolve(requiredValue(args, 'out'))
    if (workspaceRoot === outputDir) throw new Error('--workspace and --out must be different paths')
    const result = await runNativeAgentAttempt({
      taskRoot: resolve(requiredValue(args, 'task')),
      workspaceRoot,
      binaryPath: resolve(requiredValue(args, 'binary')),
      modelPath: resolve(requiredValue(args, 'model')),
      campaignId: requiredValue(args, 'campaign-id'),
      sourceSha: requiredValue(args, 'source-sha'),
      attempt: integerValue(args, 'attempt', 0, true),
      timeoutMs: integerValue(args, 'timeout-ms', null, false),
      maxOutputTokensPerStep: optionalIntegerValue(args, 'max-tokens-per-step'),
      boundary: {
        kind: 'wsl-bwrap',
        distribution: args.values.get('wsl-distribution') ?? 'Ubuntu',
        linuxBinaryPath: requiredValue(args, 'linux-binary'),
        linuxModelPath: requiredValue(args, 'linux-model'),
        gpuEnabled: booleanValue(args, 'wsl-gpu', false),
      },
    })
    const bundle = await writeNativeAgentBundle({ outputDir, result })
    const verification = await verifyBundleChecksums(outputDir)
    if (!verification.ok) throw new Error(`native bundle checksum verification failed: ${verification.failures.join('; ')}`)
    await rm(workspaceRoot, { recursive: true, force: true })
    console.log(`bundle_dir=${bundle.outputDir}`)
    console.log(`outcome=${result.attempt.score.outcome}`)
    console.log(`terminal=${result.attempt.terminal.class}`)
    if (!result.attempt.score.outcome.startsWith('PASS_')) process.exitCode = 1
  } else if (command === 'pi-run') {
    const workspaceRoot = resolve(requiredValue(args, 'workspace'))
    const outputDir = resolve(requiredValue(args, 'out'))
    if (workspaceRoot === outputDir) throw new Error('--workspace and --out must be different paths')
    const result = await runPiAgentAttempt({
      taskRoot: resolve(requiredValue(args, 'task')),
      workspaceRoot,
      piArchivePath: resolve(requiredValue(args, 'pi-archive')),
      piExecutablePath: resolve(requiredValue(args, 'pi')),
      binaryPath: resolve(requiredValue(args, 'binary')),
      modelPath: resolve(requiredValue(args, 'model')),
      modelId: requiredValue(args, 'model-id'),
      baseUrl: PI_SANDBOX_BASE_URL,
      contextWindow: integerValue(args, 'context-window', null, false),
      campaignId: requiredValue(args, 'campaign-id'),
      sourceSha: requiredValue(args, 'source-sha'),
      attempt: integerValue(args, 'attempt', 0, true),
      timeoutMs: integerValue(args, 'timeout-ms', null, false),
      maxOutputTokensPerStep: optionalIntegerValue(args, 'max-tokens-per-step'),
      boundary: {
        kind: 'wsl-bwrap',
        distribution: args.values.get('wsl-distribution') ?? 'Ubuntu',
        linuxPiDirPath: requiredValue(args, 'linux-pi-dir'),
        linuxPiArchivePath: requiredValue(args, 'linux-pi-archive'),
        linuxBinaryPath: requiredValue(args, 'linux-binary'),
        linuxModelPath: requiredValue(args, 'linux-model'),
        gpuEnabled: booleanValue(args, 'wsl-gpu', false),
      },
    })
    const bundle = await writePiAgentBundle({ outputDir, result })
    const verification = await verifyBundleChecksums(outputDir)
    if (!verification.ok) throw new Error(`Pi bundle checksum verification failed: ${verification.failures.join('; ')}`)
    await rm(workspaceRoot, { recursive: true, force: true })
    console.log(`bundle_dir=${bundle.outputDir}`)
    console.log(`outcome=${result.attempt.score.outcome}`)
    console.log(`terminal=${result.attempt.terminal.class}`)
    if (!result.attempt.score.outcome.startsWith('PASS_')) process.exitCode = 1
  } else {
    process.stdout.write(usage())
    process.exitCode = command ? 2 : 0
  }
} catch (error) {
  if (activeRun) {
    await writeText(resolve(activeRun.outputDir, 'failure.json'), canonicalJson({
      schema: 'camelid.benchmark.failure/v1',
      campaign_id: activeRun.campaignId,
      state: 'INCOMPLETE',
      error_type: error.name,
      error_message: error.message,
    })).catch(() => {})
  }
  console.error(`benchmark system ${command ?? 'command'} failed: ${error.message}`)
  process.exitCode = 1
}

async function loadAndResolve(parsed) {
  const configPath = parsed.values.get('config')
  if (!configPath) throw new Error('--config is required')
  const absoluteConfig = resolve(configPath)
  const base = dirname(absoluteConfig)
  const request = JSON.parse(await readFile(absoluteConfig, 'utf8'))
  resolvePaths(request, base)
  const controller = await controllerManifest(systemRoot)
  if (request.controller?.source_manifest_sha256 !== controller.sha256) {
    throw new Error(`controller digest is ${controller.sha256}, config pins ${request.controller?.source_manifest_sha256 ?? 'nothing'}`)
  }
  const plan = await resolveCampaignPlan(request)
  return { request, plan }
}

function resolvePaths(request, base) {
  request.repository_root = localPath(base, request.repository_root)
  for (const arm of request.source_arms ?? []) {
    arm.source_dir = localPath(base, arm.source_dir)
    arm.cargo_path = localPath(base, arm.cargo_path)
    arm.target_dir = localPath(base, arm.target_dir)
  }
  for (const model of request.models ?? []) model.artifact_path = localPath(base, model.artifact_path)
  for (const workload of request.workloads ?? []) workload.prompt_file = localPath(base, workload.prompt_file)
}

function localPath(base, value) {
  return isAbsolute(value) ? value : resolve(base, value)
}

function parseArgs(argv) {
  const positionals = []
  const values = new Map()
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (!token.startsWith('--')) {
      positionals.push(token)
      continue
    }
    const [name, inline] = token.slice(2).split('=', 2)
    if (inline !== undefined) {
      values.set(name, inline)
      continue
    }
    const next = argv[index + 1]
    if (!next || next.startsWith('--')) throw new Error(`--${name} requires a value`)
    values.set(name, next)
    index += 1
  }
  return { positionals, values }
}

function requiredValue(parsed, name) {
  const value = parsed.values.get(name)
  if (!value) throw new Error(`--${name} is required`)
  return value
}

function integerValue(parsed, name, fallback, allowZero) {
  const raw = parsed.values.get(name)
  if (raw === undefined && fallback !== null) return fallback
  if (raw === undefined) throw new Error(`--${name} is required`)
  const value = Number(raw)
  if (!Number.isSafeInteger(value) || value < (allowZero ? 0 : 1)) {
    throw new Error(`--${name} must be ${allowZero ? 'a non-negative' : 'a positive'} safe integer`)
  }
  return value
}

function booleanValue(parsed, name, fallback) {
  const raw = parsed.values.get(name)
  if (raw === undefined) return fallback
  if (raw !== 'true' && raw !== 'false') throw new Error(`--${name} must be true or false`)
  return raw === 'true'
}

function optionalIntegerValue(parsed, name) {
  if (!parsed.values.has(name)) return null
  return integerValue(parsed, name, null, false)
}

async function requireNewDirectory(path) {
  try {
    const info = await stat(path)
    if (!info.isDirectory()) throw new Error(`output path exists and is not a directory: ${path}`)
    const entries = await readdir(path)
    if (entries.length > 0) throw new Error(`output directory already contains files: ${path}`)
  } catch (error) {
    if (error.code === 'ENOENT') return
    throw error
  }
}

async function writeText(path, text) {
  await mkdir(dirname(path), { recursive: true })
  await writeFile(path, text, 'utf8')
}

function joinDefaultOutput(repositoryRoot) {
  return resolve(repositoryRoot, 'target', 'benchmark-runs')
}

function usage() {
  return `Camelid benchmark system\n\n` +
    `  node tools/bench/system/cli.mjs digest\n` +
    `  node tools/bench/system/cli.mjs plan --config <campaign.json> [--out <plan.json>]\n` +
    `  node tools/bench/system/cli.mjs run --config <campaign.json> [--out-root <dir>] [--prepared <prepared-arms.json>]\n` +
    `  node tools/bench/system/cli.mjs task-verify --task <task-dir>\n` +
    `  node tools/bench/system/cli.mjs task-materialize --task <task-dir> --workspace <new-workspace>\n` +
    `  node tools/bench/system/cli.mjs task-score --task <task-dir> --workspace <workspace> [--out <score.json>]\n` +
    `  node tools/bench/system/cli.mjs native-run --task <task-dir> --workspace <new-workspace> --binary <windows-visible-linux-binary> --linux-binary <linux-path> --model <windows-model> --linux-model <linux-path> --source-sha <sha> --campaign-id <id> --timeout-ms <ms> --out <bundle-dir> [--wsl-gpu <true|false>] [--max-tokens-per-step <n>]\n` +
    `  node tools/bench/system/cli.mjs pi-run --task <task-dir> --workspace <new-workspace> --pi-archive <windows-archive> --pi <windows-visible-linux-pi> --linux-pi-archive <linux-path> --linux-pi-dir <linux-dir> --binary <windows-visible-linux-binary> --linux-binary <linux-path> --model <windows-model> --linux-model <linux-path> --model-id <exact-id> --context-window <tokens> --source-sha <sha> --campaign-id <id> --timeout-ms <ms> --out <bundle-dir> [--wsl-gpu <true|false>] [--max-tokens-per-step <n>]\n`
}
