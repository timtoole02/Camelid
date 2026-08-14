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

    // Counted per Code session stream only — the app also opens an observatory
    // telemetry EventSource at startup, which must not consume a turn script.
    let streamsOpened = 0
    globalThis.__codeEventSourceUrls = []
    globalThis.__staleCodeTerminalReplayed = false
    class MockEventSource {
      constructor(url) {
        this.listeners = new Map()
        this.closed = false
        this.stream = 0
        this.url = String(url || '')
        if (this.url.includes('/api/agent/workspace/sessions/')) {
          streamsOpened += 1
          this.stream = streamsOpened
          globalThis.__codeStreamsOpened = streamsOpened
          globalThis.__codeEventSource = this
          globalThis.__codeEventSourceUrls.push(this.url)
        }
      }
      addEventListener(type, callback) {
        this.listeners.set(type, callback)
        if (type !== 'workspace' || this.stream === 0) return
        if (this.stream > 1) return this.followUpTurn()
        this.emitAfter(20, { sequence: 1, event: 'session.started', workspace: 'C:/projects/camelid-demo', model_id: 'Qwen3-4B-Q4_K_M.gguf' })
        this.emitAfter(40, { sequence: 2, event: 'turn.started', turn_index: 0 })
        this.emitAfter(45, { sequence: 3, event: 'agent.updated', agent_id: 'main', parent_id: null, label: 'Camelid', status: 'running', task: 'Build an interactive coding agent experience', detail: 'Inspecting the workspace' })
        this.emitAfter(50, { sequence: 4, event: 'agent.updated', agent_id: 'child-ui', parent_id: 'main', label: 'ui-specialist', status: 'running', task: 'Implement the right-side agent activity panel', detail: 'Delegated agent is working' })
        this.emitAfter(70, { sequence: 5, event: 'model.delta', content: 'I will inspect the existing component before changing it.' })
        // Qwen/Hermes models stream their tool call as ordinary tokens. It is
        // syntax, not prose, and must never reach the visible transcript.
        this.emitAfter(72, { sequence: 6, event: 'model.delta', content: '\n<tool_call>\n{"name":"list_dir","arguments":{"path":"/x","offset":0,"limit":200}}\n</tool_call>' })
        // The other shape seen live: no wrapper tag, just the call itself.
        this.emitAfter(74, { sequence: 7, event: 'model.delta', content: '\nlist_dir({"path": "/x/workspace", "limit": 200, "offset": 0})' })
        this.emitAfter(90, { sequence: 8, event: 'tool.call', detail: 'update_plan(3 steps)' })
        this.emitAfter(110, { sequence: 9, event: 'tool.result', tool: 'update_plan', outcome: 'ok', content: 'plan updated\n[x] Inspect the existing Code workspace\n[~] Build the interactive agent component\n[ ] Run focused regression tests' })
        this.emitAfter(140, { sequence: 10, event: 'tool.call', detail: 'read_file(frontend/src/App.jsx, offset=0, limit=220)' })
        this.emitAfter(170, { sequence: 11, event: 'tool.result', tool: 'read_file', outcome: 'ok', content: 'import App from \"./App\"\\n// existing application shell\\n' })
        this.emitAfter(200, { sequence: 12, event: 'tool.call', detail: 'write_file(frontend/src/components/InteractiveAgent.jsx, 1480 bytes)' })
        this.emitAfter(230, {
          sequence: 13,
          event: 'approval.required',
          approval_id: 'approval-1',
          tool: 'write_file',
          risk: 'write',
          detail: 'Create frontend/src/components/InteractiveAgent.jsx inside C:/projects/camelid-demo',
        })
      }
      // A follow-up opens a SECOND stream, and the server counts sequences per
      // stream — so these low numbers collide with turn one's, which is exactly
      // the state that used to give two rendered cards the same React key.
      followUpTurn() {
        // Model the retained server feed closely enough to catch the `after=0`
        // regression. An explicit zero asks for the whole session, including
        // turn one's terminal. Omitting the cursor asks the server for the
        // current turn only, so this stale terminal is never replayed.
        if (new URL(this.url, location.href).searchParams.get('after') === '0') {
          globalThis.__staleCodeTerminalReplayed = true
          this.emitAfter(5, { sequence: 17, event: 'session.finished', outcome: 'answered' })
        }
        this.emitAfter(20, { sequence: 1, event: 'session.started', workspace: 'C:/projects/camelid-demo', model_id: 'Qwen3-4B-Q4_K_M.gguf' })
        this.emitAfter(40, { sequence: 2, event: 'tool.call', detail: 'read_file(frontend/src/components/InteractiveAgent.jsx, offset=0, limit=80)' })
        this.emitAfter(60, { sequence: 3, event: 'tool.result', tool: 'read_file', outcome: 'ok', content: 'export function InteractiveAgent() { return null }' })
        this.emitAfter(80, { sequence: 4, event: 'model.delta', content: 'Writing focused tests for the component now.' })
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
      source?.emit({ sequence: 14, event: 'tool.result', tool: 'write_file', outcome: 'ok', content: 'Created frontend/src/components/InteractiveAgent.jsx' })
      source?.emit({ sequence: 15, event: 'model.answer', content: 'Implemented the interactive agent component and kept the change inside the selected workspace. The new component is ready for review.' })
      source?.emit({ sequence: 16, event: 'model.timing', total_ms: 2480, ttft_ms: 165, output_tokens: 92 })
      source?.emit({ sequence: 17, event: 'session.finished', outcome: 'answered' })
    }
    // The terminal event a Stop really produces: the still-open stream delivers
    // the server's own `aborted` before the DELETE poll settles.
    globalThis.__abortCodeTurn = () => {
      // Sequences are session-scoped and monotonic now, and the client drops
      // anything at or below what it has already applied — so a terminal must
      // be numbered ABOVE the flood below (100..324) or it is deduped away and
      // the turn never visibly ends.
      globalThis.__codeEventSource?.emit({ sequence: 1_000, event: 'session.finished', outcome: 'aborted' })
    }
    // Pushes the oldest entries out of the client's 240-entry activity ring.
    globalThis.__floodCodeEvents = (count) => {
      const source = globalThis.__codeEventSource
      for (let index = 0; index < count; index += 1) {
        source?.emit({ sequence: 100 + index, event: 'session.notice', content: `Checkpoint ${index}` })
      }
    }
  })

  const decisions = []
  const sessionBodies = []
  const followUps = []
  const cancels = []
  const railThreads = [{
    id: 'saved-code-thread',
    title: 'Earlier coding session',
    canonical_root: 'C:/projects/camelid-demo',
    turn_count: 3,
    updated_at: Date.now(),
  }]
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
    if (url.endsWith('/api/agent/workspace/activity')) return respondJson(request, { activity: null })
    if (url.includes('/api/agent/workspace/threads/recent?')) return respondJson(request, { threads: railThreads })
    if (url.includes('/api/agent/workspace/threads?')) return respondJson(request, { threads: [] })
    if (url.endsWith('/api/agent/workspace/sessions/code-workbench-smoke/messages')) {
      followUps.push(JSON.parse(request.postData() || '{}'))
      return respondJson(request, { session_id: 'code-workbench-smoke', turn_index: 1, state: 'waiting_for_events', duplicate: false })
    }
    if (url.endsWith('/api/agent/workspace/sessions/code-workbench-smoke')) {
      if (request.method() === 'DELETE') {
        cancels.push(Date.now())
        return request.respond({ status: 204, body: '' })
      }
      return respondJson(request, { id: 'code-workbench-smoke', state: cancels.length ? 'cancelled' : 'running' })
    }
    if (url.endsWith('/api/agent/workspace/sessions') && request.method() === 'POST') {
      const body = JSON.parse(request.postData() || '{}')
      sessionBodies.push(body)
      // The third start answers slowly on purpose: the composer offers Stop from
      // the moment it is submitted, which is before any session id exists.
      if (sessionBodies.length === 3) await new Promise((resolve) => setTimeout(resolve, 500))
      return respondJson(request, {
        id: 'code-workbench-smoke',
        workspace: 'C:/projects/camelid-demo',
        model_id: 'Qwen3-4B-Q4_K_M.gguf',
        state: 'waiting_for_events',
        max_steps: 48,
        max_tokens: 768,
        context_window: {
          mode: 'auto',
          effective_tokens: 16_384,
          safe_max_tokens: 16_384,
          model_max_tokens: 40_960,
          paged_target_tokens: 16_384,
          paged_working_set_tokens: 8_000,
          limiting_factor: 'paged_model_target',
        },
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
  await page.evaluate(() => {
    const sessionSummary = [...document.querySelectorAll('.ci-fold > summary')]
      .find((node) => node.querySelector('span')?.textContent.trim() === 'Session')
    sessionSummary?.click()
  })
  await page.waitForSelector('.ci-kv--wide')

  const pendingState = await page.evaluate(() => ({
    href: location.hash,
    workspace: document.querySelector('.code-workspace-picker input')?.value,
    hasThread: Boolean(document.querySelector('.code-feed')),
    hasToolCard: Boolean(document.querySelector('.code-tool-card')),
    planText: document.querySelector('.code-plan-update')?.textContent.replace(/\s+/g, ' ').trim(),
    hasStepChip: [...document.querySelectorAll('.code-composer__chips > span')].some((node) => node.textContent.includes('steps')),
    contextChip: document.querySelector('.code-context-chip')?.textContent.replace(/\s+/g, ' ').trim(),
    sessionDetails: Object.fromEntries([...document.querySelectorAll('.ci-kv--wide > div')].map((row) => [
      row.querySelector('dt')?.textContent.trim(),
      row.querySelector('dd')?.textContent.replace(/\s+/g, ' ').trim(),
    ])),
    hasApproval: Boolean(document.querySelector('.code-inline-approval.is-pending')),
    hasInspector: Boolean(document.querySelector('.code-inspector')),
    agents: [...document.querySelectorAll('.code-agent-list li')].map((node) => node.textContent.replace(/\s+/g, ' ').trim()),
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
    || pendingState.agents.length !== 2
    || !pendingState.agents.some((agent) => agent.includes('Primary agent'))
    || !pendingState.agents.some((agent) => agent.includes('right-side agent activity panel'))
    || !pendingState.hasComposer
    || !pendingState.planText?.includes('Working on: Build the interactive agent component')
    || !pendingState.planText?.includes('Run focused regression tests')
    || pendingState.contextChip !== 'Auto · 16K'
    || pendingState.sessionDetails.Context !== 'Auto · 16K'
    || pendingState.sessionDetails['Active prompt'] !== '7.8K paged'
    || pendingState.sessionDetails['Model max'] !== '40K'
    || pendingState.sessionDetails['Limited by'] !== 'Qwen 4B paged target'
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

  // The delete control is visually quiet until the row has keyboard focus, but
  // it must become visible before Tab reaches it. A control that stays
  // `visibility: hidden` cannot itself receive focus, making the advertised
  // keyboard path impossible.
  await page.focus('.rail-code-thread__open')
  const deleteReveal = await page.$eval('.rail-code-thread__delete', (node) => ({
    visibility: getComputedStyle(node).visibility,
    opacity: getComputedStyle(node).opacity,
  }))
  await page.keyboard.press('Tab')
  const deleteKeyboardState = await page.evaluate(() => ({
    focused: document.activeElement?.matches('.rail-code-thread__delete') || false,
    visibility: getComputedStyle(document.querySelector('.rail-code-thread__delete')).visibility,
  }))
  if (deleteReveal.visibility !== 'visible'
    || deleteReveal.opacity !== '1'
    || !deleteKeyboardState.focused
    || deleteKeyboardState.visibility !== 'visible') {
    throw new Error(`Code-thread delete is not keyboard accessible: ${JSON.stringify({ deleteReveal, deleteKeyboardState })}`)
  }

  // Opening a rail entry remounts the Code surface, and the remount cancels the
  // live turn on the server. A run only ends when it finishes or the user stops
  // it, so the rail has to ask — and declining must leave the run untouched.
  await page.click('.rail-code-thread__open')
  await page.waitForSelector('.cx-modal', { timeout: 5000 })
  const railGuard = await page.evaluate(() => ({
    text: document.querySelector('.cx-modal')?.textContent,
    confirmLabel: document.querySelector('.cx-modal .cx-btn--danger')?.textContent.trim(),
  }))
  if (!railGuard.text?.includes('stops that turn on the server')
    || railGuard.confirmLabel !== 'Stop and switch') {
    throw new Error(`rail navigation did not warn about the live run: ${JSON.stringify(railGuard)}`)
  }
  await page.click('.cx-modal .cx-btn--ghost')
  await page.waitForSelector('.cx-modal', { hidden: true })
  const railDeclined = await page.evaluate(() => ({
    sourceClosed: Boolean(globalThis.__codeEventSource?.closed),
    approvalStillPending: Boolean(document.querySelector('.code-inline-approval.is-pending')),
  }))
  if (railDeclined.sourceClosed || !railDeclined.approvalStillPending || cancels.length !== 0) {
    throw new Error(`declining the rail switch still killed the run: ${JSON.stringify({ ...railDeclined, cancels: cancels.length })}`)
  }

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

  // An approval-gated write puts `approval.required` between the call and its
  // result. The card must still resolve to Done, and the result must not also
  // appear as a second, orphaned card. `update_plan` renders as a plan card, so
  // the two tool cards here are read_file and write_file.
  const pairedState = await page.evaluate(() => ({
    toolCards: [...document.querySelectorAll('.code-tool-card')].map((node) => ({
      title: node.querySelector('summary strong')?.textContent,
      state: node.querySelector('.code-tool-card__state')?.textContent,
    })),
    dividers: [...document.querySelectorAll('.code-worked-divider')].map((node) => node.textContent.trim()),
  }))
  if (pairedState.toolCards.length !== 2
    || pairedState.toolCards.some((card) => card.state !== 'Done')
    || !pairedState.toolCards.some((card) => card.title === 'Write File')) {
    throw new Error(`approval-gated tool call did not pair with its result: ${JSON.stringify(pairedState)}`)
  }
  // The server reports a normal completion as `answered`; the divider labels it.
  if (pairedState.dividers.length !== 1 || pairedState.dividers[0] !== 'Complete') {
    throw new Error(`terminal divider mismatch: ${JSON.stringify(pairedState)}`)
  }

  const completedState = await page.evaluate(() => ({
    status: document.querySelector('.code-stage__status')?.textContent.trim(),
    assistant: document.querySelector('.code-message--assistant')?.textContent.trim(),
    inspectorHeading: document.querySelector('.code-inspector .ci-group__label')?.textContent.trim(),
    changedFiles: [...document.querySelectorAll('.code-file-list li')].map((node) => node.textContent.trim()),
    activeWork: document.querySelector('.code-process-row')?.textContent.trim(),
    sendEnabled: !document.querySelector('.code-composer__send')?.disabled,
    bodyWidth: [document.documentElement.clientWidth, document.documentElement.scrollWidth],
  }))
  if (completedState.status !== 'Complete'
    || !completedState.assistant?.includes('interactive agent component')
    || !completedState.inspectorHeading?.startsWith('Primary agent')) {
    throw new Error(`completed workbench state mismatch: ${JSON.stringify(completedState)}`)
  }
  if (!completedState.changedFiles.some((file) => file.includes('InteractiveAgent.jsx'))) {
    throw new Error(`changes inspector did not refresh: ${JSON.stringify(completedState)}`)
  }
  if (!completedState.sendEnabled || completedState.bodyWidth[0] !== completedState.bodyWidth[1]) {
    throw new Error(`follow-up or layout state mismatch: ${JSON.stringify(completedState)}`)
  }
  await page.screenshot({ path: join(outputDir, 'code-workbench-complete.png'), fullPage: true })

  // A follow-up turn opens a second stream whose sequence numbers repeat turn
  // one's. Both turns' activity has to survive in one feed.
  await page.click('.code-composer__send')
  await page.waitForFunction(() => document.body.textContent.includes('Writing focused tests'), { timeout: 5000 })
  const followUpState = await page.evaluate(() => ({
    toolCards: [...document.querySelectorAll('.code-tool-card summary strong')].map((node) => node.textContent),
    userMessages: [...document.querySelectorAll('.code-message--user')].map((node) => node.textContent),
    status: document.querySelector('.code-stage__status')?.textContent.trim(),
    streamUrls: [...globalThis.__codeEventSourceUrls],
    staleTerminalReplayed: globalThis.__staleCodeTerminalReplayed,
  }))
  if (followUps.length !== 1
    || followUpState.toolCards.length !== 3
    || followUpState.userMessages.length !== 2
    || !followUpState.userMessages[1]?.includes('focused tests')
    || followUpState.status !== 'Working'
    || followUpState.staleTerminalReplayed
    || followUpState.streamUrls.length !== 2
    || followUpState.streamUrls.some((url) => new URL(url).searchParams.get('after') === '0')) {
    throw new Error(`follow-up turn did not render alongside the first: ${JSON.stringify({ ...followUpState, followUps })}`)
  }

  // Overflow the 240-entry activity ring. The two turns above hold 18 entries,
  // so 225 more evicts the first three — including turn one's `turn.user`, the
  // marker the view used to count live turns by. Losing it promoted a turn whose
  // answer is still buffered into the finished-history list, printing that
  // answer a second time. Code turns have no step cap, so this is reachable.
  await page.evaluate(() => globalThis.__floodCodeEvents(225))
  await page.waitForFunction(() => document.body.textContent.includes('Checkpoint 224'), { timeout: 5000 })
  const evictedState = await page.evaluate(() => ({
    answers: [...document.querySelectorAll('.code-message--assistant')]
      .filter((node) => node.textContent.includes('Implemented the interactive agent component')).length,
    historyPairs: document.querySelectorAll('.code-turn-pair').length,
    notices: document.querySelectorAll('.code-session-notice').length,
  }))
  if (evictedState.answers !== 1 || evictedState.historyPairs !== 0) {
    throw new Error(`evicted activity duplicated a live turn: ${JSON.stringify(evictedState)}`)
  }

  // Stop while the stream is still open: the server sends its own `aborted`
  // terminal event, so the client must not append a second ending of its own.
  await page.click('.code-composer__send.is-stop')
  await page.evaluate(() => globalThis.__abortCodeTurn())
  await page.waitForFunction(() => !document.querySelector('.code-composer__send.is-stop'), { timeout: 8000 })
  const stoppedState = await page.evaluate(() => ({
    dividers: [...document.querySelectorAll('.code-worked-divider')].map((node) => node.textContent.trim()),
    status: document.querySelector('.code-stage__status')?.textContent.trim(),
  }))
  if (stoppedState.dividers.length !== 2
    || stoppedState.dividers[1] !== 'Stopped'
    || stoppedState.status !== 'Stopped'
    || cancels.length !== 1) {
    throw new Error(`Stop reported the turn ending twice: ${JSON.stringify({ ...stoppedState, cancels: cancels.length })}`)
  }

  // The streamed tool-call syntax must not be rendered as model output. Left
  // raw, a normal Qwen step reads as the agent having lost its mind.
  const liveText = await page.evaluate(() => ({
    thinking: document.querySelector('.code-thinking-card')?.textContent || '',
    body: document.body.textContent || '',
  }))
  if (liveText.body.includes('<tool_call>') || liveText.body.includes('"arguments"')) {
    throw new Error(`raw tool-call syntax leaked into the transcript: ${liveText.thinking.slice(0, 200)}`)
  }

  // A finished session must NOT kill the access control. It was disabled
  // whenever a session existed, so after the very first turn the dropdown was
  // dead and full auto unreachable without starting over — and this smoke missed
  // it precisely because it only ever opened the menu AFTER clicking "New
  // coding session". Assert the live case first.
  const accessAfterTurn = await page.evaluate(() => ({
    disabled: Boolean(document.querySelector('.code-access-chip')?.disabled),
  }))
  if (accessAfterTurn.disabled) {
    throw new Error('the access control is dead once a session exists')
  }
  await page.click('.code-access-chip')
  await page.waitForSelector('.code-access-menu')
  const carriedNote = await page.$eval('.code-access-menu__heading', (node) => node.textContent)
  if (!carriedNote.includes('next task')) {
    throw new Error(`the menu must say a change applies to the next task: ${carriedNote}`)
  }
  await page.keyboard.press('Escape')
  await page.waitForSelector('.code-access-menu', { hidden: true })

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
  // The warning must state what full auto grants AND how far confinement
  // actually goes on each OS. It said "on Windows ..." only, which read as a
  // sandbox claim on hosts where none applied; the kernel-sandbox clause and the
  // Windows caveat are both load-bearing, so both are pinned here.
  if (!confirmation.includes('edit files and run shell commands without asking')
    || !confirmation.includes('not filesystem- or network-isolated')
    || !confirmation.includes('kernel sandbox')
    || !confirmation.includes('on Windows')) {
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

  // Stop pressed while the create request is still in flight has to be real: the
  // session the server hands back is cancelled instead of adopted, and no event
  // stream is ever opened for it.
  await page.click('button[aria-label="New coding session"]')
  await page.click('.code-composer textarea')
  await page.type('.code-composer textarea', 'Draft a migration plan for the workspace.')
  await page.click('.code-composer__send')
  await page.waitForSelector('.code-composer__send.is-stop', { timeout: 3000 })
  await page.click('.code-composer__send.is-stop')
  await page.waitForFunction(
    () => document.querySelector('.code-stage__status')?.textContent.trim() === 'Stopped',
    { timeout: 8000 },
  )
  const abandonedStart = await page.evaluate(() => ({
    status: document.querySelector('.code-stage__status')?.textContent.trim(),
    streamsOpened: globalThis.__codeStreamsOpened,
    streamUrls: [...globalThis.__codeEventSourceUrls],
    composerPlaceholder: document.querySelector('.code-composer textarea')?.placeholder,
  }))
  if (sessionBodies.length !== 3
    || cancels.length !== 4
    || abandonedStart.streamsOpened !== 3
    || abandonedStart.streamUrls.length !== 3
    || abandonedStart.streamUrls.some((url) => new URL(url).searchParams.get('after') === '0')
    || !abandonedStart.composerPlaceholder?.includes('Describe the change')) {
    throw new Error(`Stop during session creation did not take: ${JSON.stringify({ ...abandonedStart, sessions: sessionBodies.length, cancels: cancels.length })}`)
  }

  console.log(`code-workbench: PASS ${JSON.stringify({ pendingState, railGuard, backgroundSwitchState, pairedState, completedState, followUpState, evictedState, stoppedState, menuState, fullAutoBody: sessionBodies[1], abandonedStart })}`)
} finally {
  await browser.close()
}
