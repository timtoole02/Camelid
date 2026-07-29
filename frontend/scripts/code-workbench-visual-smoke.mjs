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
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
].filter(Boolean).find(existsSync)
if (!executablePath) throw new Error('Chrome or Edge is required for Code workbench visual smoke')

const baseUrl = process.env.CAMELID_CAPTURE_URL || 'http://127.0.0.1:4175'
const outputDir = process.env.CAMELID_CAPTURE_DIR
  ? resolve(process.env.CAMELID_CAPTURE_DIR)
  : fileURLToPath(new URL('../../target/code-workbench-smoke/', import.meta.url))
await mkdir(outputDir, { recursive: true })

const health = {
  ok: true,
  engine: 'camelid',
  loaded_now: true,
  generation_ready: true,
  active_model_id: 'Qwen3-4B-Q4_K_M.gguf',
  backend: 'llama',
  model_family: 'qwen3',
  execution_plan: { selected_backend: 'cuda_resident_kquant_runtime', cuda_resident_active: true },
}
const models = {
  object: 'list',
  data: [{ id: 'Qwen3-4B-Q4_K_M.gguf', object: 'model', created: 0, owned_by: 'camelid', meta: { size: 2_497_280_256 } }],
}
const currentModel = {
  id: 'Qwen3-4B-Q4_K_M.gguf',
  path: 'models/Qwen3-4B-Q4_K_M.gguf',
  gguf: { metadata: { 'general.file_type': 15 } },
  tokenizer: { status: 'available' },
}
const localModels = {
  models_dir: 'C:/models',
  models: [{
    filename: 'Qwen3-4B-Q4_K_M.gguf',
    size_bytes: 2_497_280_256,
    architecture: 'qwen3',
    quantization: 'Q4_K_M',
    tokenizer_kind: 'gpt2_bpe',
    admitted: true,
    chat_capable: true,
    context_length: 40960,
    lane_class: 'supported',
  }],
}
const capabilities = {
  model_compatibility: [{
    id: 'Qwen3-4B-Q4_K_M.gguf',
    family: 'qwen3',
    quantization: 'Q4_K_M',
    status: 'supported_exact_row_smoke',
    tool_capable: true,
  }],
  planned_model_families: [],
  api_features: [],
  support_contract: {},
}

async function respondJson(request, value, status = 200) {
  await request.respond({
    status,
    contentType: 'application/json',
    headers: { 'Access-Control-Allow-Origin': '*' },
    body: JSON.stringify(value),
  })
}

const browser = await puppeteer.launch({ executablePath, headless: 'new' })
try {
  const page = await browser.newPage()
  // Match the default Tauri Desktop window rather than validating only a wide browser.
  await page.setViewport({ width: 1180, height: 820, deviceScaleFactor: 1 })
  await page.evaluateOnNewDocument(() => {
    localStorage.setItem('camelid-theme', 'dark')
    localStorage.setItem('camelid.codeWorkspacePath', 'C:/projects/camelid-demo')
    localStorage.setItem('camelid.codeInspectorOpen', 'true')

    class MockEventSource {
      constructor() {
        this.listeners = new Map()
        this.closed = false
        globalThis.__codeEventSource = this
      }
      addEventListener(type, callback) {
        this.listeners.set(type, callback)
        if (type !== 'workspace') return
        this.emitAfter(20, { sequence: 1, event: 'session.started', workspace: 'C:/projects/camelid-demo', model_id: 'Qwen3-4B-Q4_K_M.gguf' })
        this.emitAfter(40, { sequence: 2, event: 'turn.started', turn_index: 0 })
        this.emitAfter(70, { sequence: 3, event: 'model.delta', content: 'I will inspect the existing component before changing it.' })
        this.emitAfter(90, { sequence: 4, event: 'tool.call', detail: 'update_plan(3 steps)' })
        this.emitAfter(110, { sequence: 5, event: 'tool.result', tool: 'update_plan', outcome: 'ok', content: 'plan updated\n[x] Inspect the existing Code workspace\n[~] Build the interactive agent component\n[ ] Run focused regression tests' })
        this.emitAfter(140, { sequence: 6, event: 'tool.call', detail: 'read_file(frontend/src/App.jsx, offset=0, limit=220)' })
        this.emitAfter(170, { sequence: 7, event: 'tool.result', tool: 'read_file', outcome: 'ok', content: 'import App from \"./App\"\\n// existing application shell\\n' })
        this.emitAfter(200, { sequence: 8, event: 'tool.call', detail: 'write_file(frontend/src/components/InteractiveAgent.jsx, 1480 bytes)' })
        this.emitAfter(230, {
          sequence: 9,
          event: 'approval.required',
          approval_id: 'approval-1',
          tool: 'write_file',
          risk: 'write',
          detail: 'Create frontend/src/components/InteractiveAgent.jsx inside C:/projects/camelid-demo',
        })
      }
      emitAfter(delay, payload) { setTimeout(() => this.emit(payload), delay) }
      emit(payload) {
        const callback = this.listeners.get('workspace')
        if (callback && !this.closed) callback({ data: JSON.stringify(payload) })
      }
      close() { this.closed = true }
    }
    globalThis.EventSource = MockEventSource
    globalThis.__finishCodeTurn = () => {
      const source = globalThis.__codeEventSource
      source?.emit({ sequence: 10, event: 'tool.result', tool: 'write_file', outcome: 'ok', content: 'Created frontend/src/components/InteractiveAgent.jsx' })
      source?.emit({ sequence: 11, event: 'model.answer', content: 'Implemented the interactive agent component and kept the change inside the selected workspace. The new component is ready for review.' })
      source?.emit({ sequence: 12, event: 'model.timing', total_ms: 2480, ttft_ms: 165, output_tokens: 92 })
      source?.emit({ sequence: 13, event: 'session.finished', outcome: 'answered' })
    }
  })

  const decisions = []
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
    if (url.includes('/api/agent/workspace/threads/recent?')) return respondJson(request, { threads: [] })
    if (url.includes('/api/agent/workspace/threads?')) return respondJson(request, { threads: [] })
    if (url.endsWith('/api/agent/workspace/sessions') && request.method() === 'POST') {
      const body = JSON.parse(request.postData() || '{}')
      sessionBodies.push(body)
      return respondJson(request, {
        id: 'code-workbench-smoke',
        workspace: 'C:/projects/camelid-demo',
        model_id: 'Qwen3-4B-Q4_K_M.gguf',
        state: 'waiting_for_events',
        max_steps: 0,
        max_tokens: 768,
        allow_writes: true,
        approval_mode: body.approval_mode,
        allow_network: body.allow_network,
        mode: 'code',
      }, 201)
    }
    if (url.endsWith('/api/agent/workspace/sessions/code-workbench-smoke/decisions')) {
      decisions.push(JSON.parse(request.postData() || '{}'))
      return request.respond({ status: 204, body: '' })
    }
    if (url.endsWith('/api/agent/workspace/sessions/code-workbench-smoke/changes')) {
      const completed = decisions.length > 0
      return respondJson(request, completed ? {
        summary: '1 file changed',
        diff: '+++ frontend/src/components/InteractiveAgent.jsx\\n+export function InteractiveAgent() {\\n+  return <section>Interactive agent</section>\\n+}',
        files: ['frontend/src/components/InteractiveAgent.jsx'],
      } : { summary: 'No checkpoints this session', diff: 'No changes this session', files: [] })
    }
    return request.continue()
  })

  await page.goto(`${baseUrl}/#code`, { waitUntil: 'networkidle2', timeout: 30000 })
  await page.waitForSelector('.code-workbench')
  await page.type('.code-composer textarea', 'Inspect the WebUI and build an interactive coding agent experience.')
  await page.click('.code-composer__send')
  await page.waitForSelector('.code-inline-approval.is-pending', { timeout: 5000 })

  const pendingState = await page.evaluate(() => ({
    href: location.hash,
    workspace: document.querySelector('.code-workspace-picker input')?.value,
    hasThread: Boolean(document.querySelector('.code-feed')),
    hasToolCard: Boolean(document.querySelector('.code-tool-card')),
    planText: document.querySelector('.code-plan-update')?.textContent.replace(/\s+/g, ' ').trim(),
    hasStepChip: [...document.querySelectorAll('.code-composer__chips > span')].some((node) => node.textContent.includes('steps')),
    hasApproval: Boolean(document.querySelector('.code-inline-approval.is-pending')),
    hasInspector: Boolean(document.querySelector('.code-inspector')),
    hasComposer: Boolean(document.querySelector('.code-composer')),
    rects: Object.fromEntries(['.code-workbench', '.code-stage', '.code-thread', '.code-composer-shell', '.code-inspector', '.code-inline-approval'].map((selector) => {
      const node = document.querySelector(selector)
      const rect = node?.getBoundingClientRect()
      const style = node ? getComputedStyle(node) : null
      return [selector, rect ? {
        left: rect.left,
        right: rect.right,
        width: rect.width,
        cssWidth: style.width,
        minWidth: style.minWidth,
        maxWidth: style.maxWidth,
        position: style.position,
        display: style.display,
      } : null]
    })),
    bodyWidth: [document.documentElement.clientWidth, document.documentElement.scrollWidth],
  }))
  if (pendingState.href !== '#code'
    || !pendingState.hasThread
    || !pendingState.hasToolCard
    || !pendingState.hasApproval
    || !pendingState.hasInspector
    || !pendingState.hasComposer
    || !pendingState.planText?.includes('Working on: Build the interactive agent component')
    || !pendingState.planText?.includes('Run focused regression tests')
    || pendingState.hasStepChip) {
    throw new Error(`interactive workbench did not render: ${JSON.stringify(pendingState)}`)
  }
  if (pendingState.bodyWidth[0] !== pendingState.bodyWidth[1]) throw new Error(`horizontal overflow: ${JSON.stringify(pendingState)}`)
  if (pendingState.rects['.code-composer-shell'].right > pendingState.rects['.code-stage'].right + 1
    || pendingState.rects['.code-inline-approval'].right > pendingState.rects['.code-stage'].right + 1) {
    throw new Error(`thread content escaped under inspector: ${JSON.stringify(pendingState)}`)
  }
  if (sessionBodies.length !== 1
    || sessionBodies[0].allow_writes !== true
    || sessionBodies[0].mode !== 'code'
    || Object.hasOwn(sessionBodies[0], 'max_steps')
    || sessionBodies[0].approval_mode !== 'approval_gated'
    || sessionBodies[0].allow_network !== false) {
    throw new Error(`Code session contract mismatch: ${JSON.stringify(sessionBodies)}`)
  }
  await page.screenshot({ path: join(outputDir, 'code-workbench-approval.png'), fullPage: true })

  // Switching to ordinary Chat is a view change, not a cancellation command.
  // The live Code component must remain mounted and resume exactly where it was.
  await page.click('.topbar__mode-switch button:first-child')
  await page.waitForFunction(() => document.querySelector('.topbar__mode-switch button[aria-pressed="true"]')?.textContent === 'Chat')
  await page.click('.topbar__mode-switch button:last-child')
  await page.waitForSelector('.code-inline-approval.is-pending', { timeout: 5000 })
  const backgroundSwitchState = await page.evaluate(() => ({
    sourceClosed: Boolean(globalThis.__codeEventSource?.closed),
    approvalStillPending: Boolean(document.querySelector('.code-inline-approval.is-pending')),
    taskText: document.querySelector('.code-message--user')?.textContent,
  }))
  if (backgroundSwitchState.sourceClosed
    || !backgroundSwitchState.approvalStillPending
    || !backgroundSwitchState.taskText?.includes('interactive coding agent experience')) {
    throw new Error(`Code run did not survive a Chat view switch: ${JSON.stringify(backgroundSwitchState)}`)
  }

  await page.click('.code-inline-approval__actions .cx-btn--primary')
  await page.waitForFunction(() => document.body.textContent.includes('Decision sent'), { timeout: 5000 })
  await page.evaluate(() => globalThis.__finishCodeTurn())
  await page.waitForFunction(() => document.body.textContent.includes('Implemented the interactive agent component'), { timeout: 5000 })
  await page.waitForFunction(() => document.body.textContent.includes('InteractiveAgent.jsx'), { timeout: 5000 })
  await page.type('.code-composer textarea', 'Also add focused tests for this component.')
  if (decisions.length !== 1 || decisions[0].decision !== 'allow_once') {
    throw new Error(`Approval decision mismatch: ${JSON.stringify(decisions)}`)
  }

  const completedState = await page.evaluate(() => ({
    status: document.querySelector('.code-stage__status')?.textContent.trim(),
    assistant: document.querySelector('.code-message--assistant')?.textContent.trim(),
    changedFiles: [...document.querySelectorAll('.code-file-list li')].map((node) => node.textContent.trim()),
    activeWork: document.querySelector('.code-process-row')?.textContent.trim(),
    sendEnabled: !document.querySelector('.code-composer__send')?.disabled,
    bodyWidth: [document.documentElement.clientWidth, document.documentElement.scrollWidth],
  }))
  if (completedState.status !== 'Complete' || !completedState.assistant?.includes('interactive agent component')) {
    throw new Error(`completed workbench state mismatch: ${JSON.stringify(completedState)}`)
  }
  if (!completedState.changedFiles.some((file) => file.includes('InteractiveAgent.jsx'))) {
    throw new Error(`changes inspector did not refresh: ${JSON.stringify(completedState)}`)
  }
  if (!completedState.sendEnabled || completedState.bodyWidth[0] !== completedState.bodyWidth[1]) {
    throw new Error(`follow-up or layout state mismatch: ${JSON.stringify(completedState)}`)
  }
  await page.screenshot({ path: join(outputDir, 'code-workbench-complete.png'), fullPage: true })

  await page.click('button[aria-label="New coding session"]')
  await page.click('.code-access-chip')
  await page.waitForSelector('.code-access-menu')
  const menuState = await page.evaluate(() => ({
    text: document.querySelector('.code-access-menu')?.textContent,
    checked: [...document.querySelectorAll('.code-access-menu > button')].map((node) => node.getAttribute('aria-checked')),
    bodyWidth: [document.documentElement.clientWidth, document.documentElement.scrollWidth],
  }))
  if (!menuState.text?.includes('Today is a good day to die')
    || !menuState.text?.includes('Network and web search')
    || menuState.checked.at(-1) !== 'false'
    || menuState.bodyWidth[0] !== menuState.bodyWidth[1]) {
    throw new Error(`access menu mismatch: ${JSON.stringify(menuState)}`)
  }
  await page.screenshot({ path: join(outputDir, 'code-workbench-access-menu.png'), fullPage: true })
  await page.click('.code-access-menu > button.is-danger')
  await page.waitForSelector('.cx-modal')
  const confirmation = await page.$eval('.cx-modal', (node) => node.textContent)
  if (!confirmation.includes('edit files and run shell commands without asking')
    || !confirmation.includes('not filesystem- or network-isolated')) {
    throw new Error(`full-auto warning mismatch: ${confirmation}`)
  }
  await page.click('.cx-modal .cx-btn--danger')
  await page.waitForFunction(() => document.querySelector('.code-access-chip')?.textContent.includes('Full auto'))
  await page.click('.code-access-chip')
  await page.waitForSelector('.code-access-menu button[role="menuitemcheckbox"]')
  await page.click('.code-access-menu button[role="menuitemcheckbox"]')
  await page.waitForFunction(() => document.querySelector('.code-access-menu button[role="menuitemcheckbox"]')?.getAttribute('aria-checked') === 'true')
  await page.keyboard.press('Escape')
  await page.waitForSelector('.code-access-menu', { hidden: true })
  await page.click('.code-composer textarea')
  await page.type('.code-composer textarea', 'Run the complete test suite and research any failing dependency.')
  await page.click('.code-composer__send')
  await new Promise((resolve) => setTimeout(resolve, 120))
  if (sessionBodies.length !== 2
    || sessionBodies[1].approval_mode !== 'full_auto'
    || sessionBodies[1].allow_network !== true) {
    throw new Error(`full-auto session contract mismatch: ${JSON.stringify(sessionBodies)}`)
  }

  console.log(`code-workbench: PASS ${JSON.stringify({ pendingState, backgroundSwitchState, completedState, menuState, fullAutoBody: sessionBodies[1] })}`)
} finally {
  await browser.close()
}
