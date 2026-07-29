#!/usr/bin/env node

import { existsSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join, resolve } from 'node:path'
import puppeteer from 'puppeteer-core'

const executablePath = [
  process.env.PUPPETEER_EXECUTABLE_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter(Boolean).find(existsSync)
if (!executablePath) throw new Error('Chrome or Edge is required for Workspace visual smoke')

const baseUrl = process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const outputDir = process.env.CAMELID_CAPTURE_DIR
  ? resolve(process.env.CAMELID_CAPTURE_DIR)
  : fileURLToPath(new URL('../../target/', import.meta.url))
await mkdir(outputDir, { recursive: true })
const browser = await puppeteer.launch({ executablePath, headless: 'new' })
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
    await startButton.click()
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
} finally {
  await browser.close()
}

console.log('workspace-readonly-visual-smoke: PASS')
