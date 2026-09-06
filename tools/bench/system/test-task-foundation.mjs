#!/usr/bin/env node
import assert from 'node:assert/strict'
import { chmod, cp, mkdir, mkdtemp, readFile, rm, symlink, unlink, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import {
  applyTaskOverlay,
  loadTaskPackage,
  materializeTask,
  scoreTaskAttempt,
  taskPackageDigests,
} from './tasks/package.mjs'

const repositoryRoot = resolve(fileURLToPath(new URL('../../..', import.meta.url)))
const taskRoots = [
  'agent_local_logic_fix',
  'agent_cross_file_contract',
  'agent_input_validation',
].map((id) => resolve(repositoryRoot, 'qa/benchmarks/agent/tasks', id))
const tempRoot = await mkdtemp(join(tmpdir(), 'camelid-agent-task-'))

try {
  for (const taskRoot of taskRoots) await verifyControls(taskRoot)
  await verifyPackageTampering()
  await verifySymlinkRejected()
  await verifyScorerSelfMutation()
  await verifyTaskDefinitionMutation()
} finally {
  await rm(tempRoot, { recursive: true, force: true })
}

console.log('benchmark Phase 2 task foundation: PASS')

async function verifyControls(taskRoot) {
  const id = taskRoot.split(/[\\/]/).at(-1)
  const taskPackage = await loadTaskPackage(taskRoot)
  const untouched = await materializeTask(taskPackage, join(tempRoot, id, 'untouched'))
  assert.equal((await scoreTaskAttempt(taskRoot, untouched.workspaceRoot)).outcome, 'FAIL_BEHAVIOR')
  assert.equal(await pathExists(join(untouched.attemptRoot, 'scorer')), false)

  const solution = await materializeTask(taskPackage, join(tempRoot, id, 'solution'))
  await applyTaskOverlay(taskRoot, 'expected/solution', solution.attemptRoot)
  assert.equal((await scoreTaskAttempt(taskRoot, solution.workspaceRoot)).outcome, 'PASS_COMPARABLE')

  await applyTaskOverlay(taskRoot, 'expected/ablation', solution.attemptRoot)
  assert.equal((await scoreTaskAttempt(taskRoot, solution.workspaceRoot)).outcome, 'FAIL_BEHAVIOR')

  const wrong = await materializeTask(taskPackage, join(tempRoot, id, 'wrong'))
  await applyTaskOverlay(taskRoot, 'expected/wrong', wrong.attemptRoot)
  const wrongScore = await scoreTaskAttempt(taskRoot, wrong.workspaceRoot)
  assert.equal(wrongScore.passed_checks, wrongScore.required_checks)
  assert.equal(wrongScore.outcome, 'FAIL_BEHAVIOR', `${id} plausible wrong fix escaped the scorer`)

  const deletedTest = await materializeTask(taskPackage, join(tempRoot, id, 'deleted-test'))
  await unlink(join(deletedTest.attemptRoot, 'tests', 'test.cjs'))
  assert.equal((await scoreTaskAttempt(taskRoot, deletedTest.workspaceRoot)).outcome, 'FAIL_FORBIDDEN_MUTATION')

  const unrelated = await materializeTask(taskPackage, join(tempRoot, id, 'unrelated'))
  await writeFile(join(unrelated.attemptRoot, 'notes.txt'), 'not allowed\n')
  assert.equal((await scoreTaskAttempt(taskRoot, unrelated.workspaceRoot)).outcome, 'FAIL_FORBIDDEN_MUTATION')

  const binary = await materializeTask(taskPackage, join(tempRoot, id, 'binary'))
  await writeFile(join(binary.attemptRoot, 'src', 'payload.obj'), 'not a real object file\n')
  const binaryScore = await scoreTaskAttempt(taskRoot, binary.workspaceRoot)
  assert.equal(binaryScore.outcome, 'FAIL_FORBIDDEN_MUTATION')
  assert.match(binaryScore.errors.join('\n'), /unexpected binary artifact/)

  if (process.platform !== 'win32') {
    const executable = await materializeTask(taskPackage, join(tempRoot, id, 'executable'))
    const executablePath = join(executable.attemptRoot, 'src', 'pricing-helper')
    await writeFile(executablePath, 'not a script\n')
    await chmod(executablePath, 0o755)
    const executableScore = await scoreTaskAttempt(taskRoot, executable.workspaceRoot)
    assert.equal(executableScore.outcome, 'FAIL_FORBIDDEN_MUTATION')
    assert.match(executableScore.errors.join('\n'), /unexpected executable artifact/)
  }

  const outside = await materializeTask(taskPackage, join(tempRoot, id, 'outside'))
  await writeFile(join(outside.workspaceRoot, 'canary', 'outside.txt'), 'changed\n')
  assert.equal((await scoreTaskAttempt(taskRoot, outside.workspaceRoot)).outcome, 'FAIL_FORBIDDEN_MUTATION')

  const shadowScorer = await materializeTask(taskPackage, join(tempRoot, id, 'shadow-scorer'))
  await writeFile(join(shadowScorer.attemptRoot, 'scorer.mjs'), 'process.exit(0)\n')
  assert.equal((await scoreTaskAttempt(taskRoot, shadowScorer.workspaceRoot)).outcome, 'FAIL_FORBIDDEN_MUTATION')
}

async function pathExists(path) {
  try {
    await import('node:fs/promises').then(({ lstat }) => lstat(path))
    return true
  } catch (error) {
    if (error.code === 'ENOENT') return false
    throw error
  }
}

async function verifyPackageTampering() {
  const packageRoot = join(tempRoot, 'tampered-package')
  await cp(taskRoots[0], packageRoot, { recursive: true })
  await writeFile(join(packageRoot, 'fixture', 'src', 'pricing.cjs'), 'module.exports = {}\n')
  await assert.rejects(loadTaskPackage(packageRoot), /fixture manifest is/)
}

async function verifyScorerSelfMutation() {
  const packageRoot = join(tempRoot, 'self-mutating-package')
  await cp(taskRoots[0], packageRoot, { recursive: true })
  const scorerPath = join(packageRoot, 'scorer', 'score.mjs')
  const scorer = await readFile(scorerPath, 'utf8')
  await writeFile(scorerPath, [
    "import { appendFileSync } from 'node:fs'",
    "import { fileURLToPath } from 'node:url'",
    "appendFileSync(fileURLToPath(import.meta.url), '\\n// self mutation\\n')",
    scorer,
  ].join('\n'))
  const digests = await taskPackageDigests(packageRoot)
  const taskPath = join(packageRoot, 'task.json')
  const task = JSON.parse(await readFile(taskPath, 'utf8'))
  task.scorer_manifest_sha256 = digests.scorer.sha256
  await writeFile(taskPath, `${JSON.stringify(task, null, 2)}\n`)
  const taskPackage = await loadTaskPackage(packageRoot)
  const workspace = await materializeTask(taskPackage, join(tempRoot, 'self-mutating-workspace'))
  await applyTaskOverlay(packageRoot, 'expected/solution', workspace.attemptRoot)
  const result = await scoreTaskAttempt(packageRoot, workspace.workspaceRoot)
  assert.equal(result.outcome, 'INVALID_SCORER')
  assert.match(result.errors.join('\n'), /task package changed while the scorer ran|scorer manifest is/)
}

async function verifyTaskDefinitionMutation() {
  const packageRoot = join(tempRoot, 'task-mutating-package')
  await cp(taskRoots[0], packageRoot, { recursive: true })
  const scorerPath = join(packageRoot, 'scorer', 'score.mjs')
  const scorer = await readFile(scorerPath, 'utf8')
  await writeFile(scorerPath, [
    "import { appendFileSync } from 'node:fs'",
    "appendFileSync(new URL('../task.json', import.meta.url), ' ')",
    scorer,
  ].join('\n'))
  const digests = await taskPackageDigests(packageRoot)
  const taskPath = join(packageRoot, 'task.json')
  const task = JSON.parse(await readFile(taskPath, 'utf8'))
  task.scorer_manifest_sha256 = digests.scorer.sha256
  await writeFile(taskPath, `${JSON.stringify(task, null, 2)}\n`)
  const taskPackage = await loadTaskPackage(packageRoot)
  const workspace = await materializeTask(taskPackage, join(tempRoot, 'task-mutating-workspace'))
  await applyTaskOverlay(packageRoot, 'expected/solution', workspace.attemptRoot)
  const result = await scoreTaskAttempt(packageRoot, workspace.workspaceRoot)
  assert.equal(result.outcome, 'INVALID_SCORER')
  assert.match(result.errors.join('\n'), /task package changed while the scorer ran/)
}

async function verifySymlinkRejected() {
  const packageRoot = join(tempRoot, 'symlink-package')
  const outsideRoot = join(tempRoot, 'symlink-outside')
  await cp(taskRoots[0], packageRoot, { recursive: true })
  await mkdir(outsideRoot)
  await writeFile(join(outsideRoot, 'outside.txt'), 'outside\n')
  await symlink(
    outsideRoot,
    join(packageRoot, 'fixture', 'escape'),
    process.platform === 'win32' ? 'junction' : 'dir',
  )
  await assert.rejects(loadTaskPackage(packageRoot), /symbolic links are not allowed/)
}