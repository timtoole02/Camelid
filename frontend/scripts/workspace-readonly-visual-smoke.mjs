#!/usr/bin/env node

import { mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join, resolve } from 'node:path'
import { launchBrowser } from './lib/launch-browser.mjs'

const baseUrl = process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const outputDir = process.env.CAMELID_CAPTURE_DIR
  ? resolve(process.env.CAMELID_CAPTURE_DIR)
  : fileURLToPath(new URL('../../target/', import.meta.url))
await mkdir(outputDir, { recursive: true })
const browser = await launchBrowser({ purpose: 'the Workspace visual smoke', headless: 'new' })
const markdownFiles = [
  'CONFIGURATION.md',
  'CONFORMANCE.md',
  'CONTEXT.md',
  'CONTRIBUTOR_QUICKSTART.md',
  'TELEMETRY.md',
  'VALIDATION_MATRIX.md',
  'WAR_ROOM_EVIDENCE_INDEX.md',
  'gemma4-cuda-port-plan.md',
  'gemma4-cuda-q4_0-plan.md',
  'gemma4-engine-status.md',
  'gemma4-gpu-port-plan.md',
  'gemma4-row-audit-2026-06-09.md',
  'gemma4-two-mac-cluster.md',
  'housekeeping-check.md',
]
const inventory = [
  `Found ${markdownFiles.length} Markdown files in the selected folder:`,
  '',
  ...markdownFiles.map((file) => `- \`${file}\``),
  '',
  'Directories and non-matching files were excluded. Nested folders were not searched.',
  '',
  'Verification order:',
  '',
  ...markdownFiles.map((file, index) => `${index + 1}. \`${file}\``),
].join('\n')

const health = {
  ok: true, engine: 'camelid', loaded_now: true, generation_ready: true,
  active_model_id: 'qwen3_4b_q4_k_m', backend: 'qwen3', model_family: 'qwen3',
  q8_runtime: {}, execution_plan: null, engine_queue_depth: 0,
}
const models = {
  object: 'list',
  data: [{
    id: 'qwen3_4b_q4_k_m', object: 'model', created: 0, owned_by: 'camelid',
    meta: { n_ctx_train: 32768, n_params: 4_000_000_000, size: 2_497_280_256 },
  }],
}
const currentModel = {
  id: 'qwen3_4b_q4_k_m', path: 'C:/models/Qwen3-4B-Q4_K_M.gguf',
  gguf: { metadata: { general: { file_type: 15 } } }, tokenizer: { status: 'available' },
}
const localModels = {
  models_dir: 'C:/models',
  models: [{
    filename: 'Qwen3-4B-Q4_K_M.gguf', size_bytes: 2_497_280_256,
    architecture: 'qwen3', quantization: 'Q4_K_M', tokenizer_kind: 'gpt2_bpe',
    admitted: true, oracle_qualified: true, chat_capable: true,
    context_length: 32768, lane_class: 'supported',
  }],
}
const capabilities = {
  model_compatibility: [{
    id: 'qwen3_4b_q4_k_m', family: 'qwen3', quantization: 'Q4_K_M',
    status: 'supported_exact_row_smoke', tool_capable: true,
  }],
  planned_model_families: [], api_features: [], support_contract: {},
}

async function respondJson(request, value, status = 200) {
  await request.respond({
    status,
    contentType: 'application/json',
    headers: { 'Access-Control-Allow-Origin': '*' },
    body: JSON.stringify(value),
  })
}

try {
  for (const viewport of [
    { name: 'desktop', width: 1280, height: 800 },
    { name: 'mobile', width: 390, height: 844 },
  ]) {
    const page = await browser.newPage()
    await page.setViewport(viewport)
    await page.evaluateOnNewDocument((answer, files) => {
      localStorage.setItem('camelid-theme', 'dark')
      class MockEventSource {
        constructor() {
          this.listeners = new Map()
          this.closed = false
        }
        addEventListener(type, callback) {
          this.listeners.set(type, callback)
          if (type !== 'workspace') return
          const emit = (delay, payload) => setTimeout(() => {
            if (!this.closed) callback({ data: JSON.stringify(payload) })
          }, delay)
          emit(20, { sequence: 1, event: 'session.started', model_id: 'qwen3_4b_q4_k_m' })
          emit(40, {
            sequence: 2, event: 'memory.updated', prompt_tokens: 2560,
            generation_tokens: 512, budget_total: 4096,
            system_tokens_estimate: 140, tool_definition_tokens_estimate: 280,
            message_tokens_estimate: 180, recent_memory_tokens_estimate: 1500,
            retrieved_memory_tokens_estimate: 200, evidence_memory_tokens_estimate: 180,
            tool_result_tokens_estimate: 80,
          })
          emit(60, { sequence: 3, event: 'tool.call', detail: 'list_dir(., offset=0, limit=all)' })
          emit(80, {
            sequence: 4, event: 'tool.result', tool: 'list_dir', outcome: 'ok',
            content: [...files, 'architecture/', 'archive/', 'notes.txt'].join('\n'),
          })
          emit(100, { sequence: 5, event: 'model.answer', content: answer })
          emit(120, {
            sequence: 6, event: 'memory.compacted', compacted_through_turn: 3,
            archived_turns: 4, compaction_count: 1, trigger_tokens: 3072,
            budget_total: 4096,
          })
          emit(140, { sequence: 7, event: 'session.finished', outcome: 'answered' })
        }
        close() { this.closed = true }
      }
      globalThis.EventSource = MockEventSource
    }, inventory, markdownFiles)

    const sessionBodies = []
    await page.setRequestInterception(true)
    page.on('request', async (request) => {
      const url = request.url()
      if (request.method() === 'OPTIONS') {
        return request.respond({
          status: 204,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
          },
          body: '',
        })
      }
      if (url.endsWith('/v1/health')) return respondJson(request, health)
      if (url.endsWith('/v1/models')) return respondJson(request, models)
      if (url.endsWith('/api/capabilities')) return respondJson(request, capabilities)
      if (url.endsWith('/api/models/catalog/downloads')) return respondJson(request, [])
      if (url.endsWith('/api/models/current')) return respondJson(request, currentModel)
      if (url.endsWith('/api/models/local')) return respondJson(request, localModels)
      if (url.endsWith('/api/agent/workspace/models')) {
        return respondJson(request, { models: [{
          row_id: 'qwen3_4b_q4_k_m', name: 'Qwen3 4B Q4_K_M',
          filename: 'Qwen3-4B-Q4_K_M.gguf', quantization: 'Q4_K_M',
          installed: true, catalog_id: null, fit: 'fits_resident', fit_confidence: 'exact',
        }] })
      }
      if (url.includes('/api/agent/workspace/threads?')) return respondJson(request, { threads: [] })
      if (url.endsWith('/api/agent/workspace/sessions') && request.method() === 'POST') {
        sessionBodies.push(JSON.parse(request.postData() || '{}'))
        return respondJson(request, {
          id: 'workspace-format', workspace: 'C:/camelid-agent-workspace/docs',
          model_id: 'qwen3_4b_q4_k_m', state: 'waiting_for_events',
          max_steps: 12, max_tokens: 512, allow_writes: false,
        }, 201)
      }
      if (url.endsWith('/api/agent/workspace/sessions/workspace-format')) {
        return respondJson(request, {
          id: 'workspace-format', workspace: 'C:/camelid-agent-workspace/docs',
          model_id: 'qwen3_4b_q4_k_m', state: 'idle', context_budget_tokens: 4096,
          resident_cuda: { max_positions: 29946, filled_positions: 3072, offloaded: false },
          allow_writes: false,
        })
      }
      if (url.includes('/api/agent/workspace/threads/workspace-format/compact?')) {
        return respondJson(request, {
          compacted_through_turn: null, archived_turns: 4, compaction_count: 0,
        })
      }
      return request.continue()
    })

    await page.goto(`${baseUrl}/#workspace`, { waitUntil: 'networkidle2', timeout: 30000 })
    await page.waitForSelector('.workspace-view')
    await page.$eval('.workspace-field input', (input) => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
      setter.call(input, 'C:/camelid-agent-workspace/docs')
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await page.type('.workspace-field--goal textarea', 'check all the md files in this folder')
    const startButton = await page.$('.workspace-setup__actions .cx-btn--primary')
    if (!startButton || await startButton.evaluate((node) => node.disabled)) {
      const diagnostics = await page.evaluate(() => ({
        status: document.querySelector('.workspace-status')?.innerText,
        model: document.querySelector('.workspace-model-line')?.innerText,
        prerequisite: document.querySelector('.workspace-prerequisite')?.innerText,
      }))
      throw new Error(`${viewport.name}: Workspace did not unlock ${JSON.stringify(diagnostics)}`)
    }
    await page.evaluate(() => {
      const button = document.querySelector('.workspace-setup__actions .cx-btn--primary')
      button.click()
      button.click()
    })
    await page.waitForFunction(
      () => document.querySelector('.workspace-status')?.textContent === 'Complete',
      { timeout: 5000 },
    )
    await page.waitForFunction(
      () => document.body.textContent.includes('Conversation compacted'),
      { timeout: 5000 },
    )

    const result = await page.evaluate((expectedFiles) => {
      const root = document.documentElement
      const answer = document.querySelector('.workspace-answer__body')
      const bullets = [...answer.querySelectorAll('ul li')].map((node) => node.textContent.trim())
      const ordered = [...answer.querySelectorAll('ol li')].map((node) => node.textContent.trim())
      const orderedList = answer.querySelector('ol')
      const answerRect = answer.getBoundingClientRect()
      const firstOrderedRect = orderedList?.querySelector('li')?.getBoundingClientRect()
      const setupRect = document.querySelector('.workspace-setup').getBoundingClientRect()
      const activityRect = document.querySelector('.workspace-activity').getBoundingClientRect()
      const goalRect = document.querySelector('.workspace-field--goal').getBoundingClientRect()
      const actionsRect = document.querySelector('.workspace-setup__actions').getBoundingClientRect()
      const questionRect = document.querySelector('.workspace-answer__question').getBoundingClientRect()
      const answerBarRect = document.querySelector('.workspace-answer__bar').getBoundingClientRect()
      const answerBodyRect = answer.getBoundingClientRect()
      return {
        bullets,
        ordered,
        expectedFiles,
        writeCheckboxes: document.querySelectorAll('.workspace-view input[type="checkbox"]').length,
        writeTextPresent: document.body.innerText.includes('Allow proposed file changes'),
        readOnlyTextPresent: document.body.innerText.includes('files are never changed'),
        compactedEventPresent: document.body.textContent.includes('Conversation compacted'),
        undoPresent: [...document.querySelectorAll('button')].some((button) => button.textContent.includes('Undo last')),
        inspectorPresent: Boolean(document.querySelector('.workspace-context-inspector')),
        inspectorText: document.querySelector('.workspace-context-inspector')?.textContent,
        folderLocked: document.querySelector('.workspace-field input')?.disabled,
        goalLocked: document.querySelector('.workspace-field--goal textarea')?.disabled,
        statusRole: document.querySelector('.workspace-status')?.getAttribute('role'),
        statusLive: document.querySelector('.workspace-status')?.getAttribute('aria-live'),
        buttonTexts: [...document.querySelectorAll('button')].map((button) => button.textContent.trim()),
        orderedPadding: orderedList ? Number.parseFloat(getComputedStyle(orderedList).paddingInlineStart) : 0,
        orderedContentInset: firstOrderedRect ? firstOrderedRect.left - answerRect.left : 0,
        vertical: {
          setupBottom: setupRect.bottom,
          activityTop: activityRect.top,
          goalBottom: goalRect.bottom,
          actionsBottom: actionsRect.bottom,
          questionBottom: questionRect.bottom,
          answerBarTop: answerBarRect.top,
          answerBarBottom: answerBarRect.bottom,
          answerBodyTop: answerBodyRect.top,
        },
        documentWidth: [root.clientWidth, root.scrollWidth],
        answerWidth: [answer.clientWidth, answer.scrollWidth],
      }
    }, markdownFiles)

    if (JSON.stringify(result.bullets) !== JSON.stringify(markdownFiles)) throw new Error(`${viewport.name}: grounded bullet inventory mismatch ${JSON.stringify(result)}`)
    if (JSON.stringify(result.ordered) !== JSON.stringify(markdownFiles)) throw new Error(`${viewport.name}: long ordered list mismatch ${JSON.stringify(result)}`)
    if (result.bullets.some((entry) => entry.endsWith('/'))) throw new Error(`${viewport.name}: directory leaked into inventory`)
    if (result.writeCheckboxes !== 0 || result.writeTextPresent) throw new Error(`${viewport.name}: write UI leaked ${JSON.stringify(result)}`)
    if (!result.readOnlyTextPresent) throw new Error(`${viewport.name}: read-only contract missing`)
    if (!result.compactedEventPresent || !result.undoPresent) throw new Error(`${viewport.name}: automatic compaction or undo missing ${JSON.stringify(result)}`)
    if (!result.folderLocked || !result.goalLocked) throw new Error(`${viewport.name}: active session identity was not locked ${JSON.stringify(result)}`)
    if (result.statusRole !== 'status' || result.statusLive !== 'polite') throw new Error(`${viewport.name}: status is not announced accessibly ${JSON.stringify(result)}`)
    if (result.orderedPadding < 40 || result.orderedContentInset < 40) throw new Error(`${viewport.name}: ordered markers lack stable inset ${JSON.stringify(result)}`)
    if (result.vertical.goalBottom > result.vertical.setupBottom || result.vertical.actionsBottom > result.vertical.setupBottom) throw new Error(`${viewport.name}: setup content escaped its grid row ${JSON.stringify(result)}`)
    if (viewport.name === 'mobile' && result.vertical.setupBottom > result.vertical.activityTop + 1) throw new Error(`${viewport.name}: setup and activity panes overlap ${JSON.stringify(result)}`)
    if (result.vertical.questionBottom > result.vertical.answerBarTop || result.vertical.answerBarBottom > result.vertical.answerBodyTop) throw new Error(`${viewport.name}: answer sections overlap ${JSON.stringify(result)}`)
    if (result.documentWidth[0] !== result.documentWidth[1] || result.answerWidth[0] !== result.answerWidth[1]) throw new Error(`${viewport.name}: horizontal overflow ${JSON.stringify(result)}`)
    if (sessionBodies.length !== 1 || sessionBodies[0].allow_writes !== false) throw new Error(`${viewport.name}: session was not explicitly read-only ${JSON.stringify(sessionBodies)}`)

    await page.screenshot({ path: join(outputDir, `workspace-readonly-format-${viewport.name}.png`), fullPage: true })
    if (viewport.name === 'desktop') {
      await page.click('.workspace-context-inspector > summary')
      await page.waitForSelector('.workspace-context-inspector[open] .workspace-context-inspector__panel')
      await page.screenshot({ path: join(outputDir, 'workspace-readonly-context-desktop.png'), fullPage: true })
    }
    console.log(`${viewport.name}: PASS ${JSON.stringify(result)}`)
    await page.close()
  }

  const resumePage = await browser.newPage()
  await resumePage.setViewport({ width: 1280, height: 800 })
  await resumePage.evaluateOnNewDocument(() => {
    localStorage.setItem('camelid-theme', 'dark')
    localStorage.setItem('camelid.workspacePath', 'C:/workspace-preview')
  })
  let previewDeleteCount = 0
  await resumePage.setRequestInterception(true)
  resumePage.on('request', async (request) => {
    const url = request.url()
    if (request.method() === 'OPTIONS') {
      return request.respond({ status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' }, body: '' })
    }
    if (url.endsWith('/v1/health')) return respondJson(request, health)
    if (url.endsWith('/v1/models')) return respondJson(request, models)
    if (url.endsWith('/api/capabilities')) return respondJson(request, capabilities)
    if (url.endsWith('/api/models/catalog/downloads')) return respondJson(request, [])
    if (url.endsWith('/api/models/current')) return respondJson(request, currentModel)
    if (url.endsWith('/api/models/local')) return respondJson(request, localModels)
    if (url.endsWith('/api/agent/workspace/models')) return respondJson(request, { models: [] })
    if (url.includes('/api/agent/workspace/threads/workspace-preview?') && request.method() === 'DELETE') {
      previewDeleteCount += 1
      return request.respond({ status: 204, headers: { 'Access-Control-Allow-Origin': '*' }, body: '' })
    }
    if (url.includes('/api/agent/workspace/threads/workspace-preview?')) {
      return respondJson(request, {
        thread: {
          id: 'workspace-preview', title: 'Which file handles authentication?', canonical_root: 'C:/workspace-preview', model_id: 'qwen3_4b_q4_k_m',
          model_sha256: 'test-sha', compacted_through_turn: null, compaction_count: 0,
          updated_at: 1_784_733_939, turn_count: 2,
        },
        turns: [
          { user_text: 'Which file handles authentication?', assistant_text: 'Authentication is handled by auth.rs.', terminal_outcome: 'answered' },
          { user_text: 'Read every file before I stopped you.', assistant_text: '', terminal_outcome: 'aborted' },
        ],
      })
    }
    if (url.includes('/api/agent/workspace/threads?')) {
      return respondJson(request, { threads: [{
        id: 'workspace-preview', title: 'Which file handles authentication?', canonical_root: 'C:/workspace-preview', model_id: 'qwen3_4b_q4_k_m',
        model_sha256: 'test-sha', compacted_through_turn: null, compaction_count: 0,
        updated_at: 1_784_733_939, turn_count: 2,
      }] })
    }
    return request.continue()
  })
  await resumePage.goto(`${baseUrl}/#workspace`, { waitUntil: 'networkidle2', timeout: 30000 })
  await resumePage.waitForSelector('.workspace-thread-picker option[value="workspace-preview"]')
  await resumePage.select('.workspace-thread-picker select', 'workspace-preview')
  await resumePage.waitForFunction(() => document.body.textContent.includes('Authentication is handled by auth.rs.'), { timeout: 5000 })
  const resumeState = await resumePage.evaluate(() => ({
    selectedLabel: document.querySelector('.workspace-thread-picker select')?.selectedOptions[0]?.textContent.trim(),
    questions: [...document.querySelectorAll('.workspace-answer__question')].map((node) => node.textContent.trim()),
    answerText: document.querySelector('.workspace-result')?.textContent,
    followUpPresent: Boolean(document.querySelector('.workspace-follow-up')),
    goalLabel: document.querySelector('.workspace-field--goal > span')?.textContent,
    primaryText: document.querySelector('.workspace-setup__actions .cx-btn--primary')?.textContent.trim(),
  }))
  if (JSON.stringify(resumeState.questions) !== JSON.stringify(['Which file handles authentication?', 'Read every file before I stopped you.'])) {
    throw new Error(`saved conversation questions did not hydrate on selection: ${JSON.stringify(resumeState)}`)
  }
  if (!resumeState.selectedLabel.startsWith('Which file handles authentication? · 2 turns ·')) {
    throw new Error(`saved conversation title was not rendered first: ${JSON.stringify(resumeState)}`)
  }
  if (!resumeState.answerText.includes('Authentication is handled by auth.rs.') || !resumeState.answerText.includes('Stopped')) {
    throw new Error(`saved conversation answers/outcomes did not hydrate on selection: ${JSON.stringify(resumeState)}`)
  }
  if (resumeState.followUpPresent || resumeState.goalLabel !== 'Next goal' || resumeState.primaryText !== 'Resume & send') {
    throw new Error(`saved conversation preview exposed incorrect controls: ${JSON.stringify(resumeState)}`)
  }
  let dismissedDialog = ''
  resumePage.once('dialog', async (dialog) => {
    dismissedDialog = dialog.message()
    await dialog.dismiss()
  })
  await resumePage.click('.workspace-thread-picker button')
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 50))
  if (!dismissedDialog.includes('Which file handles authentication?') || !dismissedDialog.includes('cannot be undone')) {
    throw new Error(`saved-thread deletion did not explain the destructive action: ${dismissedDialog}`)
  }
  if (previewDeleteCount !== 0) throw new Error('dismissed saved-thread deletion still reached the backend')

  resumePage.once('dialog', async (dialog) => { await dialog.accept() })
  await resumePage.click('.workspace-thread-picker button')
  await resumePage.waitForFunction(() => !document.querySelector('.workspace-thread-picker option[value="workspace-preview"]'), { timeout: 5000 })
  if (previewDeleteCount !== 1) throw new Error(`confirmed saved-thread deletion count was ${previewDeleteCount}`)
  resumeState.deleteConfirmation = 'PASS'
  console.log(`resume-preview: PASS ${JSON.stringify(resumeState)}`)
  await resumePage.close()

  const cancelPage = await browser.newPage()
  await cancelPage.setViewport({ width: 1280, height: 800 })
  await cancelPage.evaluateOnNewDocument(() => {
    localStorage.setItem('camelid-theme', 'dark')
    localStorage.removeItem('camelid.workspacePath')
    class HangingEventSource {
      constructor() { this.listeners = new Map(); this.closed = false }
      addEventListener(type, callback) {
        this.listeners.set(type, callback)
        if (type !== 'workspace') return
        setTimeout(() => callback({ data: JSON.stringify({ sequence: 1, event: 'session.started', model_id: 'qwen3_4b_q4_k_m' }) }), 20)
        setTimeout(() => callback({ data: JSON.stringify({ sequence: 2, event: 'turn.started', turn_index: 0 }) }), 40)
      }
      close() { this.closed = true }
    }
    globalThis.__emitWorkspace = (payload) => {
      const source = globalThis.__workspaceSource
      const callback = source?.listeners.get('workspace')
      if (callback && !source.closed) callback({ data: JSON.stringify(payload) })
    }
    const OriginalEventSource = HangingEventSource
    globalThis.EventSource = class extends OriginalEventSource {
      constructor(...args) {
        super(...args)
        globalThis.__workspaceSource = this
      }
    }
  })
  await cancelPage.setRequestInterception(true)
  let cancelAttempts = 0
  let cancelStatusReads = 0
  cancelPage.on('request', async (request) => {
    const url = request.url()
    if (request.method() === 'OPTIONS') {
      return request.respond({ status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' }, body: '' })
    }
    if (url.endsWith('/v1/health')) return respondJson(request, health)
    if (url.endsWith('/v1/models')) return respondJson(request, models)
    if (url.endsWith('/api/capabilities')) return respondJson(request, capabilities)
    if (url.endsWith('/api/models/catalog/downloads')) return respondJson(request, [])
    if (url.endsWith('/api/models/current')) return respondJson(request, currentModel)
    if (url.endsWith('/api/models/local')) return respondJson(request, localModels)
    if (url.endsWith('/api/agent/workspace/models')) return respondJson(request, { models: [] })
    if (url.includes('/api/agent/workspace/threads?')) return respondJson(request, { threads: [] })
    if (url.endsWith('/api/agent/workspace/sessions') && request.method() === 'POST') {
      return respondJson(request, { id: 'workspace-cancel-failure', workspace: 'C:/workspace', model_id: 'qwen3_4b_q4_k_m', state: 'waiting_for_events', max_steps: 12, max_tokens: 512, allow_writes: false }, 201)
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-cancel-failure') && request.method() === 'DELETE') {
      cancelAttempts += 1
      if (cancelAttempts === 1) return respondJson(request, { error: { message: 'simulated cancel failure' } }, 500)
      return request.respond({ status: 204, body: '' })
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-cancel-failure') && request.method() === 'GET') {
      cancelStatusReads += 1
      return respondJson(request, {
        id: 'workspace-cancel-failure', workspace: 'C:/workspace', model_id: 'qwen3_4b_q4_k_m',
        state: cancelStatusReads < 4 ? 'cancelling' : 'cancelled', context_budget_tokens: 4096,
        resident_cuda: null, allow_writes: false,
      })
    }
    return request.continue()
  })
  await cancelPage.goto(`${baseUrl}/#workspace`, { waitUntil: 'networkidle2', timeout: 30000 })
  await cancelPage.waitForSelector('.workspace-view')
  await cancelPage.$eval('.workspace-field input', (input) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
    setter.call(input, 'C:/workspace')
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await cancelPage.type('.workspace-field--goal textarea', 'inspect files')
  await cancelPage.click('.workspace-setup__actions .cx-btn--primary')
  await cancelPage.waitForSelector('.workspace-setup__actions .cx-btn--outline')
  await cancelPage.click('.workspace-setup__actions .cx-btn--outline')
  try {
    await cancelPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Stop failed', { timeout: 5000 })
  } catch {
    const diagnostic = await cancelPage.evaluate(() => ({
      status: document.querySelector('.workspace-status')?.textContent,
      text: document.querySelector('.workspace-view')?.textContent,
    }))
    throw new Error(`cancel failure did not reach Stop failed state: ${JSON.stringify(diagnostic)}`)
  }
  const cancelState = await cancelPage.evaluate(() => ({
    status: document.querySelector('.workspace-status')?.textContent,
    stoppedText: document.body.textContent.includes('Session stopped'),
    errorText: document.body.textContent.includes('simulated cancel failure'),
    followUpPresent: Boolean(document.querySelector('.workspace-follow-up')),
  }))
  if (cancelState.status !== 'Stop failed' || cancelState.stoppedText || !cancelState.errorText || cancelState.followUpPresent) {
    throw new Error(`cancel failure was misreported: ${JSON.stringify(cancelState)}`)
  }
  console.log(`cancel-failure: PASS ${JSON.stringify(cancelState)}`)

  await cancelPage.click('.workspace-setup__actions .cx-btn--outline')
  await cancelPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Stopping', { timeout: 5000 })
  await cancelPage.evaluate(() => globalThis.__emitWorkspace({ sequence: 3, event: 'session.finished', outcome: 'aborted' }))
  const stoppingState = await cancelPage.evaluate(() => ({
    status: document.querySelector('.workspace-status')?.textContent,
    followUpPresent: Boolean(document.querySelector('.workspace-follow-up')),
  }))
  if (stoppingState.followUpPresent) throw new Error(`follow-up appeared while cancellation was settling: ${JSON.stringify(stoppingState)}`)
  await cancelPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Stopped', { timeout: 5000 })
  await cancelPage.waitForSelector('.workspace-follow-up', { timeout: 5000 })
  const settledState = await cancelPage.evaluate(() => ({
    status: document.querySelector('.workspace-status')?.textContent,
    followUpPresent: Boolean(document.querySelector('.workspace-follow-up')),
  }))
  if (!settledState.followUpPresent) throw new Error(`follow-up did not appear after cancellation settled: ${JSON.stringify(settledState)}`)
  console.log(`cancel-settled: PASS ${JSON.stringify({ ...settledState, cancelAttempts, cancelStatusReads })}`)
  await cancelPage.close()

  const startStopPage = await browser.newPage()
  await startStopPage.setViewport({ width: 1280, height: 800 })
  await startStopPage.evaluateOnNewDocument(() => {
    localStorage.setItem('camelid-theme', 'dark')
    localStorage.removeItem('camelid.workspacePath')
    globalThis.EventSource = class {
      constructor() { this.closed = false }
      addEventListener(type, callback) {
        if (type !== 'workspace') return
        const emit = (delay, payload) => setTimeout(() => {
          if (!this.closed) callback({ data: JSON.stringify(payload) })
        }, delay)
        emit(15, { sequence: 1, event: 'session.started', model_id: 'qwen3_4b_q4_k_m' })
        emit(25, { sequence: 2, event: 'model.answer', content: 'Restart completed without an orphaned session.' })
        emit(35, { sequence: 3, event: 'session.finished', outcome: 'answered' })
      }
      close() { this.closed = true }
    }
  })
  let releasePublishedCreate
  const publishedCreateRelease = new Promise((resolvePromise) => { releasePublishedCreate = resolvePromise })
  let createPosts = 0
  let firstCreatePublished = false
  const startStopActions = []
  let startStopDeletes = 0
  await startStopPage.setRequestInterception(true)
  startStopPage.on('request', async (request) => {
    const url = request.url()
    if (request.method() === 'OPTIONS') {
      return request.respond({ status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' }, body: '' })
    }
    if (url.endsWith('/v1/health')) return respondJson(request, health)
    if (url.endsWith('/v1/models')) return respondJson(request, models)
    if (url.endsWith('/api/capabilities')) return respondJson(request, capabilities)
    if (url.endsWith('/api/models/catalog/downloads')) return respondJson(request, [])
    if (url.endsWith('/api/models/current')) return respondJson(request, currentModel)
    if (url.endsWith('/api/models/local')) return respondJson(request, localModels)
    if (url.endsWith('/api/agent/workspace/models')) return respondJson(request, { models: [] })
    if (url.includes('/api/agent/workspace/threads?')) return respondJson(request, { threads: [] })
    if (url.endsWith('/api/agent/workspace/sessions') && request.method() === 'POST') {
      createPosts += 1
      if (createPosts === 1) {
        firstCreatePublished = true
        startStopActions.push('first-published')
        await publishedCreateRelease
        startStopActions.push('first-response')
        return respondJson(request, {
          id: 'workspace-start-race-1', workspace: 'C:/workspace-start-race', model_id: 'qwen3_4b_q4_k_m',
          state: 'waiting_for_events', max_steps: 12, max_tokens: 512, allow_writes: false,
        }, 201)
      }
      startStopActions.push('second-create')
      return respondJson(request, {
        id: 'workspace-start-race-2', workspace: 'C:/workspace-start-race', model_id: 'qwen3_4b_q4_k_m',
        state: 'waiting_for_events', max_steps: 12, max_tokens: 512, allow_writes: false,
      }, 201)
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-start-race-1') && request.method() === 'DELETE') {
      startStopDeletes += 1
      startStopActions.push('first-delete')
      return request.respond({ status: 204, body: '' })
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-start-race-1') && request.method() === 'GET') {
      return respondJson(request, {
        id: 'workspace-start-race-1', workspace: 'C:/workspace-start-race', model_id: 'qwen3_4b_q4_k_m',
        state: 'cancelled', context_budget_tokens: 4096, resident_cuda: null, allow_writes: false,
      })
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-start-race-2') && request.method() === 'GET') {
      return respondJson(request, {
        id: 'workspace-start-race-2', workspace: 'C:/workspace-start-race', model_id: 'qwen3_4b_q4_k_m',
        state: 'idle', context_budget_tokens: 4096, resident_cuda: null, allow_writes: false,
      })
    }
    return request.continue()
  })
  await startStopPage.goto(`${baseUrl}/#workspace`, { waitUntil: 'networkidle2', timeout: 30000 })
  await startStopPage.waitForSelector('.workspace-view')
  await startStopPage.$eval('.workspace-field input', (input) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
    setter.call(input, 'C:/workspace-start-race')
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await startStopPage.type('.workspace-field--goal textarea', 'exercise stop while Workspace is starting')
  await startStopPage.click('.workspace-setup__actions .cx-btn--primary')
  for (let attempt = 0; attempt < 100 && !firstCreatePublished; attempt += 1) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 10))
  }
  if (!firstCreatePublished) throw new Error('start/stop race never reached the backend-published state')
  await startStopPage.waitForSelector('.workspace-setup__actions .cx-btn--outline')
  await startStopPage.click('.workspace-setup__actions .cx-btn--outline')
  await startStopPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Stopping', { timeout: 5000 })
  if (startStopDeletes !== 0) throw new Error('start/stop race tried to cancel before the published session ID arrived')
  releasePublishedCreate()
  await startStopPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Stopped', { timeout: 5000 })
  if (startStopDeletes !== 1 || startStopActions.join(',') !== 'first-published,first-response,first-delete') {
    throw new Error(`start/stop race did not cancel the published session exactly once: ${JSON.stringify({ startStopDeletes, startStopActions })}`)
  }
  await startStopPage.click('.workspace-setup__actions .cx-btn--primary')
  await startStopPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Complete', { timeout: 5000 })
  if (createPosts !== 2 || startStopActions.at(-1) !== 'second-create') {
    throw new Error(`immediate restart did not create a fresh session: ${JSON.stringify({ createPosts, startStopActions })}`)
  }
  console.log(`start-stop-race: PASS ${JSON.stringify({ createPosts, startStopDeletes, startStopActions })}`)
  await startStopPage.close()

  const retryPage = await browser.newPage()
  await retryPage.setViewport({ width: 1280, height: 800 })
  await retryPage.evaluateOnNewDocument(() => {
    localStorage.setItem('camelid-theme', 'dark')
    localStorage.removeItem('camelid.workspacePath')
    globalThis.__workspaceEventSources = 0
    globalThis.EventSource = class {
      constructor(url) {
        this.closed = false
        this.workspace = String(url).includes('/api/agent/workspace/')
        if (this.workspace) globalThis.__workspaceEventSources += 1
      }
      addEventListener(type, callback) {
        if (type !== 'workspace' || !this.workspace) return
        const emit = (delay, payload) => setTimeout(() => {
          if (!this.closed) callback({ data: JSON.stringify(payload) })
        }, delay)
        emit(15, { sequence: 1, event: 'session.started', model_id: 'qwen3_4b_q4_k_m' })
        emit(25, { sequence: 2, event: 'model.answer', content: 'Initial durable answer.' })
        emit(35, { sequence: 3, event: 'session.finished', outcome: 'answered' })
      }
      close() { this.closed = true }
    }
  })
  const followUpBodies = []
  let followUpPosts = 0
  await retryPage.setRequestInterception(true)
  retryPage.on('request', async (request) => {
    const url = request.url()
    if (request.method() === 'OPTIONS') {
      return request.respond({ status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' }, body: '' })
    }
    if (url.endsWith('/v1/health')) return respondJson(request, health)
    if (url.endsWith('/v1/models')) return respondJson(request, models)
    if (url.endsWith('/api/capabilities')) return respondJson(request, capabilities)
    if (url.endsWith('/api/models/catalog/downloads')) return respondJson(request, [])
    if (url.endsWith('/api/models/current')) return respondJson(request, currentModel)
    if (url.endsWith('/api/models/local')) return respondJson(request, localModels)
    if (url.endsWith('/api/agent/workspace/models')) return respondJson(request, { models: [] })
    if (url.includes('/api/agent/workspace/threads/workspace-retry?')) {
      return respondJson(request, {
        thread: {
          id: 'workspace-retry', title: 'inspect retry behavior', canonical_root: 'C:/workspace-retry',
          model_id: 'qwen3_4b_q4_k_m', model_sha256: 'test-sha', compacted_through_turn: null,
          compaction_count: 0, updated_at: 1_784_733_939, turn_count: 2,
        },
        turns: [
          { user_text: 'inspect retry behavior', assistant_text: 'Initial durable answer.', terminal_outcome: 'answered' },
          { user_text: 'show the durable retry', assistant_text: 'Durable follow-up answer.', terminal_outcome: 'answered' },
        ],
      })
    }
    if (url.includes('/api/agent/workspace/threads?')) return respondJson(request, { threads: [] })
    if (url.endsWith('/api/agent/workspace/sessions') && request.method() === 'POST') {
      return respondJson(request, {
        id: 'workspace-retry', workspace: 'C:/workspace-retry', model_id: 'qwen3_4b_q4_k_m',
        state: 'waiting_for_events', max_steps: 12, max_tokens: 512, allow_writes: false,
      }, 201)
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-retry/messages') && request.method() === 'POST') {
      followUpPosts += 1
      followUpBodies.push(JSON.parse(request.postData() || '{}'))
      if (followUpPosts === 1) return request.abort('failed')
      return respondJson(request, {
        session_id: 'workspace-retry', turn_index: 1, state: 'idle', duplicate: true,
      })
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-retry') && request.method() === 'GET') {
      return respondJson(request, {
        id: 'workspace-retry', workspace: 'C:/workspace-retry', model_id: 'qwen3_4b_q4_k_m',
        state: 'idle', context_budget_tokens: 4096, resident_cuda: null, allow_writes: false,
      })
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-retry') && request.method() === 'DELETE') {
      return request.respond({ status: 204, body: '' })
    }
    return request.continue()
  })
  await retryPage.goto(`${baseUrl}/#workspace`, { waitUntil: 'networkidle2', timeout: 30000 })
  await retryPage.waitForSelector('.workspace-view')
  await retryPage.$eval('.workspace-field input', (input) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
    setter.call(input, 'C:/workspace-retry')
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await retryPage.type('.workspace-field--goal textarea', 'inspect retry behavior')
  await retryPage.evaluate(() => {
    const button = document.querySelector('.workspace-setup__actions .cx-btn--primary')
    button.click()
    button.click()
  })
  await retryPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Complete', { timeout: 5000 })
  await retryPage.type('.workspace-follow-up textarea', 'show the durable retry')
  await retryPage.click('.workspace-follow-up button[type="submit"]')
  await retryPage.waitForFunction(() => (
    document.querySelector('.workspace-status')?.textContent === 'Error'
    && document.querySelector('.workspace-follow-up textarea')?.value === 'show the durable retry'
  ), { timeout: 5000 })
  await retryPage.click('.workspace-follow-up button[type="submit"]')
  await retryPage.waitForFunction(() => document.body.textContent.includes('Durable follow-up answer.'), { timeout: 5000 })
  const retryState = await retryPage.evaluate(() => ({
    eventSources: globalThis.__workspaceEventSources,
    folderLocked: document.querySelector('.workspace-field input')?.disabled,
    folderValue: document.querySelector('.workspace-field input')?.value,
    clearPresent: [...document.querySelectorAll('.workspace-setup__actions button')].some((button) => button.textContent.includes('Clear activity')),
    questions: [...document.querySelectorAll('.workspace-answer__question')].map((node) => node.textContent.trim()),
  }))
  if (followUpBodies.length !== 2 || followUpBodies[0].client_message_id !== followUpBodies[1].client_message_id) {
    throw new Error(`ambiguous follow-up retry changed its client_message_id: ${JSON.stringify(followUpBodies)}`)
  }
  if (retryState.eventSources !== 1) throw new Error(`persisted duplicate opened an unavailable second event stream: ${JSON.stringify(retryState)}`)
  if (!retryState.folderLocked || retryState.folderValue !== 'C:/workspace-retry') throw new Error(`session folder identity drifted: ${JSON.stringify(retryState)}`)
  if (!retryState.clearPresent) throw new Error(`restored duplicate left the bound session impossible to clear: ${JSON.stringify(retryState)}`)
  if (JSON.stringify(retryState.questions) !== JSON.stringify(['inspect retry behavior', 'show the durable retry'])) {
    throw new Error(`duplicate retry did not hydrate durable turns: ${JSON.stringify(retryState)}`)
  }
  console.log(`follow-up-retry: PASS ${JSON.stringify({ ...retryState, clientMessageId: followUpBodies[0].client_message_id })}`)
  await retryPage.close()

  const recoveryPage = await browser.newPage()
  await recoveryPage.setViewport({ width: 1280, height: 800 })
  await recoveryPage.evaluateOnNewDocument(() => {
    localStorage.setItem('camelid-theme', 'dark')
    localStorage.removeItem('camelid.workspacePath')
    globalThis.EventSource = class {
      constructor(url) {
        this.closed = false
        this.workspace = String(url).includes('/api/agent/workspace/')
      }
      addEventListener(type, callback) {
        if (type !== 'workspace' || !this.workspace) return
        setTimeout(() => {
          if (!this.closed) callback({ data: JSON.stringify({ sequence: 1, event: 'session.started', model_id: 'qwen3_4b_q4_k_m' }) })
        }, 15)
        setTimeout(() => {
          if (!this.closed) callback({ data: JSON.stringify({ sequence: 2, event: 'approval.required', approval_id: 'should-never-happen' }) })
        }, 30)
      }
      close() { this.closed = true }
    }
  })
  let releaseRecovery
  const recoveryGate = new Promise((resolveRecovery) => { releaseRecovery = resolveRecovery })
  let recoveryDeletes = 0
  await recoveryPage.setRequestInterception(true)
  recoveryPage.on('request', async (request) => {
    const url = request.url()
    if (request.method() === 'OPTIONS') {
      return request.respond({ status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' }, body: '' })
    }
    if (url.endsWith('/v1/health')) return respondJson(request, health)
    if (url.endsWith('/v1/models')) return respondJson(request, models)
    if (url.endsWith('/api/capabilities')) return respondJson(request, capabilities)
    if (url.endsWith('/api/models/catalog/downloads')) return respondJson(request, [])
    if (url.endsWith('/api/models/current')) return respondJson(request, currentModel)
    if (url.endsWith('/api/models/local')) return respondJson(request, localModels)
    if (url.endsWith('/api/agent/workspace/models')) return respondJson(request, { models: [] })
    if (url.includes('/api/agent/workspace/threads/workspace-recovery?')) {
      return respondJson(request, {
        thread: {
          id: 'workspace-recovery', title: 'exercise recovery', canonical_root: 'C:/workspace-recovery',
          model_id: 'qwen3_4b_q4_k_m', model_sha256: 'test-sha', compacted_through_turn: null,
          compaction_count: 0, updated_at: 1_784_733_939, turn_count: 1,
        },
        turns: [{ user_text: 'exercise recovery', assistant_text: '', terminal_outcome: 'aborted' }],
      })
    }
    if (url.includes('/api/agent/workspace/threads?')) return respondJson(request, { threads: [] })
    if (url.endsWith('/api/agent/workspace/sessions') && request.method() === 'POST') {
      return respondJson(request, {
        id: 'workspace-recovery', workspace: 'C:/workspace-recovery', model_id: 'qwen3_4b_q4_k_m',
        state: 'waiting_for_events', max_steps: 12, max_tokens: 512, allow_writes: false,
      }, 201)
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-recovery') && request.method() === 'DELETE') {
      recoveryDeletes += 1
      if (recoveryDeletes === 1) await recoveryGate
      return request.respond({ status: 204, body: '' })
    }
    if (url.endsWith('/api/agent/workspace/sessions/workspace-recovery') && request.method() === 'GET') {
      return respondJson(request, {
        id: 'workspace-recovery', workspace: 'C:/workspace-recovery', model_id: 'qwen3_4b_q4_k_m',
        state: 'cancelled', context_budget_tokens: 4096, resident_cuda: null, allow_writes: false,
      })
    }
    return request.continue()
  })
  await recoveryPage.goto(`${baseUrl}/#workspace`, { waitUntil: 'networkidle2', timeout: 30000 })
  await recoveryPage.waitForSelector('.workspace-view')
  await recoveryPage.$eval('.workspace-field input', (input) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
    setter.call(input, 'C:/workspace-recovery')
    input.dispatchEvent(new Event('input', { bubbles: true }))
  })
  await recoveryPage.type('.workspace-field--goal textarea', 'exercise recovery')
  await recoveryPage.click('.workspace-setup__actions .cx-btn--primary')
  await recoveryPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Recovering', { timeout: 5000 })
  const heldRecovery = await recoveryPage.evaluate(() => ({
    followUpPresent: Boolean(document.querySelector('.workspace-follow-up')),
    stopPresent: [...document.querySelectorAll('.workspace-setup__actions button')].some((button) => button.textContent.includes('Stop')),
  }))
  if (heldRecovery.followUpPresent || !heldRecovery.stopPresent) {
    throw new Error(`approval recovery did not fail closed: ${JSON.stringify(heldRecovery)}`)
  }
  releaseRecovery()
  await recoveryPage.waitForFunction(() => document.querySelector('.workspace-status')?.textContent === 'Error', { timeout: 5000 })
  await recoveryPage.waitForSelector('.workspace-follow-up', { timeout: 5000 })
  const settledRecovery = await recoveryPage.evaluate(() => ({
    errorText: document.querySelector('.workspace-result')?.textContent,
    stoppedTurn: document.querySelector('.workspace-answer__body')?.textContent,
    followUpPresent: Boolean(document.querySelector('.workspace-follow-up')),
  }))
  if (!settledRecovery.errorText.includes('unexpected approval request') || settledRecovery.stoppedTurn !== 'Stopped' || !settledRecovery.followUpPresent) {
    throw new Error(`approval recovery did not reconcile durable state: ${JSON.stringify(settledRecovery)}`)
  }
  console.log(`approval-recovery: PASS ${JSON.stringify({ ...settledRecovery, recoveryDeletes })}`)
  await recoveryPage.close()
} finally {
  await browser.close()
}

console.log('workspace-readonly-visual-smoke: PASS')
