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
  // Deliberately use the catalog id rather than the filename. Context-profile
  // eligibility must follow the resolved exact row, not this display/runtime id.
  active_model_id: 'qwen3_4b_instruct_q8_0',
  backend: 'llama',
  model_family: 'qwen3',
  execution_plan: { selected_backend: 'cuda_resident_q8', cuda_resident_active: true },
}
const models = {
  object: 'list',
  data: [{ id: 'qwen3_4b_instruct_q8_0', object: 'model', created: 0, owned_by: 'camelid', meta: { size: 4_280_404_704 } }],
}
const currentModel = {
  id: 'qwen3_4b_instruct_q8_0',
  path: 'models/Qwen3-4B-Q8_0.gguf',
  gguf: { metadata: { 'general.file_type': 7 } },
  tokenizer: { status: 'available' },
}
const localModels = {
  models_dir: 'C:/models',
  models: [{
    filename: 'Qwen3-4B-Q8_0.gguf',
    size_bytes: 4_280_404_704,
    architecture: 'qwen3',
    quantization: 'Q8_0',
    tokenizer_kind: 'gpt2_bpe',
    admitted: true,
    chat_capable: true,
    context_length: 40960,
    lane_class: 'supported',
  }],
}
const capabilities = {
  model_compatibility: [{
    id: 'qwen3_4b_instruct_q8_0',
    family: 'qwen3',
    quantization: 'Q8_0',
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
    class MockEventSource {
      constructor(url) {
        this.listeners = new Map()
        this.closed = false
        this.stream = 0
        if (String(url || '').includes('/api/agent/workspace/sessions/')) {
          streamsOpened += 1
          this.stream = streamsOpened
          globalThis.__codeStreamsOpened = streamsOpened
          globalThis.__codeEventSource = this
        }
      }
      addEventListener(type, callback) {
        this.listeners.set(type, callback)
        if (type !== 'workspace' || this.stream === 0) return
        if (this.stream > 1) return this.followUpTurn()
        this.emitAfter(20, { sequence: 1, event: 'session.started', workspace: 'C:/projects/camelid-demo', model_id: 'Qwen3-4B-Q8_0.gguf' })
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
        this.emitAfter(20, { sequence: 1, event: 'session.started', workspace: 'C:/projects/camelid-demo', model_id: 'Qwen3-4B-Q8_0.gguf' })
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
      source?.emit({
        sequence: 16,
        event: 'model.timing',
        total_ms: 2480,
        ttft_ms: 190,
        output_tokens: 92,
        prefill_ms: 120,
        server_first_content_ms: 165,
        decode_ms: 2200,
        prompt_cache_hit: true,
        reused_tokens: 2884,
        prefilled_tokens: 302,
        prompt_cache_decision: 'block_prefix_hit',
        common_prefix_tokens: 2884,
        divergent_suffix_tokens: 302,
        candidate_tokens: 3000,
        cache_block_tokens: 64,
        matched_cache_blocks: 45,
      })
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
      // The third request is an in-thread replacement and the fifth a brand-new
      // session. Both answer slowly so Stop lands before a session id exists.
      if (sessionBodies.length === 3) await new Promise((resolve) => setTimeout(resolve, 1_250))
      if (sessionBodies.length === 5) await new Promise((resolve) => setTimeout(resolve, 500))
      const standardContext = body.context_profile === 'standard'
      return respondJson(request, {
        id: 'code-workbench-smoke',
        workspace: 'C:/projects/camelid-demo',
        model_id: 'Qwen3-4B-Q8_0.gguf',
        state: 'waiting_for_events',
        max_steps: 0,
        max_tokens: 768,
        context_profile: body.context_profile || 'auto',
        context_window: {
          mode: 'auto',
          effective_tokens: standardContext ? 8_192 : 16_384,
          recommended_max_tokens: 8_192,
          memory_safe_max_tokens: 2_048,
          model_max_tokens: 40_960,
          validated_max_tokens: 8_192,
          kv_owner_slots: 1,
          paged_target_tokens: standardContext ? null : 16_384,
          paged_working_set_tokens: standardContext ? null : 8_000,
          limiting_factor: standardContext ? 'validated_agent_maximum' : 'paged_model_target',
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
      // A replacement session on the same saved thread owns a fresh checkpoint
      // journal. Its Changes view must not retain files from the prior session.
      const completed = decisions.length > 0 && sessionBodies.length < 2
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
      .find((summary) => summary.textContent.includes('Session'))
    sessionSummary?.click()
  })
  await page.waitForSelector('.ci-kv--wide', { timeout: 5000 })

  const pendingState = await page.evaluate(() => ({
    href: location.hash,
    workspace: document.querySelector('.code-workspace-picker input')?.value,
    hasThread: Boolean(document.querySelector('.code-feed')),
    hasToolCard: Boolean(document.querySelector('.code-tool-card')),
    planText: document.querySelector('.code-plan-update')?.textContent.replace(/\s+/g, ' ').trim(),
    hasStepChip: [...document.querySelectorAll('.code-composer__chips > span')].some((node) => node.textContent.includes('steps')),
    hasApproval: Boolean(document.querySelector('.code-inline-approval.is-pending')),
    hasInspector: Boolean(document.querySelector('.code-inspector')),
    agents: [...document.querySelectorAll('.code-agent-list li')].map((node) => node.textContent.replace(/\s+/g, ' ').trim()),
    hasComposer: Boolean(document.querySelector('.code-composer')),
    contextChip: document.querySelector('.code-context-chip')?.textContent.replace(/\s+/g, ' ').trim(),
    sessionDetails: Object.fromEntries([...document.querySelectorAll('.ci-kv--wide > div')].map((row) => [
      row.querySelector('dt')?.textContent.trim(),
      row.querySelector('dd')?.textContent.replace(/\s+/g, ' ').trim(),
    ])),
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
    || pendingState.sessionDetails['Active working set'] !== '7.8K paged'
    || pendingState.sessionDetails['Memory estimate'] !== '2K / KV owner'
    || pendingState.sessionDetails['Model max'] !== '40K'
    || pendingState.sessionDetails['Agent ceiling'] !== '8K'
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
    || sessionBodies[0].context_profile !== 'auto'
    || sessionBodies[0].allow_network !== false) {
    throw new Error(`Code session contract mismatch: ${JSON.stringify(sessionBodies)}`)
  }
  await page.screenshot({ path: join(outputDir, 'code-workbench-approval.png'), fullPage: true })

  // Opening a rail entry remounts the Code surface, and the remount cancels the
  // live turn on the server. A run only ends when it finishes or the user stops
  // it, so the rail has to ask — and declining must leave the run untouched.
  await page.click('.rail-code-thread')
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
  await page.waitForFunction(() => document.body.textContent.includes('Reviewed'), { timeout: 5000 })
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

  // A follow-up turn opens a second stream whose sequence numbers repeat turn
  // one's. Both turns' activity has to survive in one feed.
  await page.click('.code-composer__send')
  await page.waitForFunction(() => document.body.textContent.includes('Writing focused tests'), { timeout: 5000 })
  const followUpState = await page.evaluate(() => ({
    toolCards: [...document.querySelectorAll('.code-tool-card summary strong')].map((node) => node.textContent),
    userMessages: [...document.querySelectorAll('.code-message--user')].map((node) => node.textContent),
  }))
  if (followUps.length !== 1
    || followUpState.toolCards.length !== 3
    || followUpState.userMessages.length !== 2
    || !followUpState.userMessages[1]?.includes('focused tests')) {
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

  // Access and context are fixed per session. Changing either after a turn must
  // resume the same saved thread through a replacement session, not silently
  // send a follow-up under the old configuration. The replacement owns fresh
  // diagnostics and a fresh checkpoint journal.
  await page.click('.code-access-chip')
  await page.waitForSelector('.code-access-menu')
  await page.click('.code-access-menu > button.is-writes-auto')
  await page.click('.code-context-chip')
  await page.waitForSelector('.code-context-menu', { timeout: 5000 })
  await page.click('.code-context-menu > button:last-child')
  await page.click('.code-composer textarea')
  await page.type('.code-composer textarea', 'Continue with automatic writes in the standard context.')
  await page.click('.code-composer__send')
  await page.waitForFunction(
    () => globalThis.__codeStreamsOpened === 3
      && document.querySelector('.code-context-chip')?.textContent.includes('Q4 8K'),
    { timeout: 5000 },
  )
  await new Promise((resolve) => setTimeout(resolve, 650))
  const replacementState = await page.evaluate(() => ({
    primary: document.querySelector('.code-agent-list li')?.textContent.replace(/\s+/g, ' ').trim(),
    contextChip: document.querySelector('.code-context-chip')?.textContent.replace(/\s+/g, ' ').trim(),
    changedFiles: document.querySelectorAll('.code-file-list li').length,
    hasContextComposition: [...document.querySelectorAll('.ci-fold > summary')]
      .some((summary) => summary.textContent.includes('Composition')),
  }))
  if (sessionBodies.length !== 2
    || sessionBodies[1].thread_id !== 'code-workbench-smoke'
    || sessionBodies[1].approval_mode !== 'writes_auto'
    || sessionBodies[1].context_profile !== 'standard'
    || sessionBodies[1].allow_network !== false
    || !replacementState.primary?.includes('0 model steps this session')
    || replacementState.contextChip !== 'Q4 8K · 8K'
    || replacementState.changedFiles !== 0
    || replacementState.hasContextComposition) {
    throw new Error(`replacement session kept stale configuration or diagnostics: ${JSON.stringify({ sessionBodies, replacementState })}`)
  }
  await page.click('.code-composer__send.is-stop')
  await page.evaluate(() => globalThis.__abortCodeTurn())
  await page.waitForFunction(() => !document.querySelector('.code-composer__send.is-stop'), { timeout: 8000 })

  // Cancelling a replacement while its POST is still pending must put the old
  // completed session back exactly as it was. `session.starting` clears
  // diagnostics optimistically, so treating the abandoned replacement as a
  // terminal event would leave the old session mislabeled and half-empty.
  const beforeAbandonedReplacement = await page.evaluate(() => ({
    status: document.querySelector('.code-stage__status')?.textContent.trim(),
    elapsed: document.querySelector('.code-elapsed')?.textContent.trim(),
    transcript: document.querySelector('.code-thread')?.textContent.replace(/\s+/g, ' ').trim(),
    environment: document.querySelector('.ci-environment__rows')?.textContent.replace(/\s+/g, ' ').trim(),
    runSummary: [...(document.querySelector('.code-agent-list li')?.querySelectorAll('.ci-task__meta') || [])]
      .at(-1)?.textContent.replace(/\s+/g, ' ').trim(),
    process: document.querySelector('.code-process-row')?.textContent.replace(/\s+/g, ' ').trim(),
    changedFiles: document.querySelectorAll('.code-file-list li').length,
  }))
  await page.click('.code-context-chip')
  await page.waitForSelector('.code-context-menu')
  await page.click('.code-context-menu > button:first-of-type')
  await page.click('.code-composer textarea')
  await page.type('.code-composer textarea', 'Try a replacement, then stop before it starts.')
  await page.click('.code-composer__send')
  await page.waitForSelector('.code-composer__send.is-stop', { timeout: 3000 })
  await page.click('.code-composer__send.is-stop')
  await page.waitForFunction(() => !document.querySelector('.code-composer__send.is-stop'), { timeout: 8000 })
  const abandonedReplacement = await page.evaluate(() => ({
    status: document.querySelector('.code-stage__status')?.textContent.trim(),
    elapsed: document.querySelector('.code-elapsed')?.textContent.trim(),
    transcript: document.querySelector('.code-thread')?.textContent.replace(/\s+/g, ' ').trim(),
    environment: document.querySelector('.ci-environment__rows')?.textContent.replace(/\s+/g, ' ').trim(),
    runSummary: [...(document.querySelector('.code-agent-list li')?.querySelectorAll('.ci-task__meta') || [])]
      .at(-1)?.textContent.replace(/\s+/g, ' ').trim(),
    process: document.querySelector('.code-process-row')?.textContent.replace(/\s+/g, ' ').trim(),
    changedFiles: document.querySelectorAll('.code-file-list li').length,
    streamsOpened: globalThis.__codeStreamsOpened,
    contextChip: document.querySelector('.code-context-chip')?.textContent.replace(/\s+/g, ' ').trim(),
  }))
  if (sessionBodies.length !== 3
    || sessionBodies[2].thread_id !== 'code-workbench-smoke'
    || sessionBodies[2].context_profile !== 'auto'
    || cancels.length !== 3
    || abandonedReplacement.streamsOpened !== 3
    || abandonedReplacement.status !== beforeAbandonedReplacement.status
    || abandonedReplacement.elapsed !== beforeAbandonedReplacement.elapsed
    || abandonedReplacement.transcript !== beforeAbandonedReplacement.transcript
    || abandonedReplacement.environment !== beforeAbandonedReplacement.environment
    || abandonedReplacement.runSummary !== beforeAbandonedReplacement.runSummary
    || abandonedReplacement.process !== beforeAbandonedReplacement.process
    || abandonedReplacement.changedFiles !== beforeAbandonedReplacement.changedFiles
    || abandonedReplacement.contextChip !== 'Auto') {
    throw new Error(`abandoned replacement did not restore the prior session: ${JSON.stringify({ beforeAbandonedReplacement, abandonedReplacement, sessionBodies, cancels: cancels.length })}`)
  }

  await page.click('button[aria-label="New coding session"]')
  await page.click('.code-access-chip')
  await page.waitForSelector('.code-access-menu')
  const menuState = await page.evaluate(() => ({
    text: document.querySelector('.code-access-menu')?.textContent,
    checked: [...document.querySelectorAll('.code-access-menu > button')].map((node) => node.getAttribute('aria-checked')),
    bodyWidth: [document.documentElement.clientWidth, document.documentElement.scrollWidth],
  }))
  if (!menuState.text?.includes('Writes auto')
    || !menuState.text?.includes('Full auto')
    || !menuState.text?.includes('Network and web search')
    || menuState.checked.at(-1) !== 'false'
    || menuState.bodyWidth[0] !== menuState.bodyWidth[1]) {
    throw new Error(`access menu mismatch: ${JSON.stringify(menuState)}`)
  }
  await page.screenshot({ path: join(outputDir, 'code-workbench-access-menu.png'), fullPage: true })
  await page.waitForFunction(() => document.activeElement?.getAttribute('role') === 'menuitemradio')
  await page.keyboard.press('ArrowDown')
  const accessArrowFocus = await page.evaluate(() => document.activeElement?.textContent.replace(/\s+/g, ' ').trim())
  if (!accessArrowFocus?.startsWith('Writes auto')) {
    throw new Error(`access menu did not move focus with ArrowDown: ${accessArrowFocus}`)
  }
  await page.keyboard.press('Escape')
  await page.waitForSelector('.code-access-menu', { hidden: true })
  const accessReturnedFocus = await page.evaluate(() => document.activeElement?.classList.contains('code-access-chip'))
  if (!accessReturnedFocus) throw new Error('access menu Escape did not restore trigger focus')

  // A wrapped context chip sits near the left edge on phones. The popup must
  // clamp to the viewport instead of anchoring its right edge there and opening
  // mostly offscreen. Exercise the same ARIA focus contract while narrow.
  await page.setViewport({ width: 320, height: 820, deviceScaleFactor: 1 })
  await page.click('.code-inspector__header button')
  await page.waitForSelector('.code-inspector', { hidden: true })
  await page.$eval('.code-context-chip', (node) => node.click())
  await page.waitForSelector('.code-context-menu', { timeout: 5000 })
  await page.waitForFunction(() => document.activeElement?.getAttribute('role') === 'menuitemradio')
  const narrowContextMenu = await page.evaluate(() => {
    const rect = document.querySelector('.code-context-menu')?.getBoundingClientRect()
    const trigger = document.querySelector('.code-context-chip')?.getBoundingClientRect()
    const firstChip = document.querySelector('.code-access-chip')?.getBoundingClientRect()
    const footer = document.querySelector('.code-composer__footer')?.getBoundingClientRect()
    return rect && trigger && firstChip && footer ? {
      left: rect.left,
      right: rect.right,
      width: rect.width,
      viewport: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      wrapped: trigger.top > firstChip.top + 2,
      footerLeft: footer.left,
      footerRight: footer.right,
    } : null
  })
  if (!narrowContextMenu
    || !narrowContextMenu.wrapped
    || narrowContextMenu.width < 240
    || narrowContextMenu.left < -0.5
    || narrowContextMenu.right > narrowContextMenu.viewport + 0.5
    || narrowContextMenu.left < narrowContextMenu.footerLeft - 0.5
    || narrowContextMenu.right > narrowContextMenu.footerRight + 0.5
    || narrowContextMenu.scrollWidth !== narrowContextMenu.viewport) {
    throw new Error(`context menu escaped the narrow viewport: ${JSON.stringify(narrowContextMenu)}`)
  }
  await page.keyboard.press('ArrowDown')
  const contextArrowFocus = await page.evaluate(() => document.activeElement?.textContent.replace(/\s+/g, ' ').trim())
  if (!contextArrowFocus?.startsWith('Q8 16K')) {
    throw new Error(`context menu did not move focus with ArrowDown: ${contextArrowFocus}`)
  }
  await page.keyboard.press('Escape')
  await page.waitForSelector('.code-context-menu', { hidden: true })
  const contextReturnedFocus = await page.evaluate(() => document.activeElement?.classList.contains('code-context-chip'))
  if (!contextReturnedFocus) throw new Error('context menu Escape did not restore trigger focus')
  await page.setViewport({ width: 1180, height: 820, deviceScaleFactor: 1 })

  await page.click('.code-access-chip')
  await page.waitForSelector('.code-access-menu')
  await page.waitForFunction(() => document.activeElement?.getAttribute('role') === 'menuitemradio')
  await page.keyboard.press('ArrowDown')
  await page.keyboard.press('ArrowDown')
  const fullAutoKeyboardFocus = await page.evaluate(() => document.activeElement?.textContent.replace(/\s+/g, ' ').trim())
  if (!fullAutoKeyboardFocus?.startsWith('Full auto')) {
    throw new Error(`access menu did not reach Full auto by keyboard: ${fullAutoKeyboardFocus}`)
  }
  await page.keyboard.press('Enter')
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
  if (sessionBodies.length !== 4
    || sessionBodies[3].approval_mode !== 'full_auto'
    || sessionBodies[3].context_profile !== 'auto'
    || sessionBodies[3].allow_network !== true) {
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
    composerPlaceholder: document.querySelector('.code-composer textarea')?.placeholder,
  }))
  if (sessionBodies.length !== 5
    || cancels.length !== 6
    || abandonedStart.streamsOpened !== 4
    || !abandonedStart.composerPlaceholder?.includes('Describe the change')) {
    throw new Error(`Stop during session creation did not take: ${JSON.stringify({ ...abandonedStart, sessions: sessionBodies.length, cancels: cancels.length })}`)
  }

  console.log(`code-workbench: PASS ${JSON.stringify({ pendingState, railGuard, backgroundSwitchState, pairedState, completedState, followUpState, evictedState, stoppedState, abandonedReplacement: { status: abandonedReplacement.status, streamsOpened: abandonedReplacement.streamsOpened, contextChip: abandonedReplacement.contextChip }, menuState, fullAutoBody: sessionBodies[3], abandonedStart })}`)
} finally {
  await browser.close()
}
