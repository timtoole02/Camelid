import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { findCompatibilityHint } from '../lib/capabilities'
import { appStorage } from '../lib/appStorage.js'
import {
  browseWorkspaceFolders,
  cancelWorkspaceSession,
  compactWorkspaceThread,
  createWorkspaceSession,
  deleteWorkspaceThread,
  getWorkspaceCompatibleModels,
  getWorkspaceSession,
  getWorkspaceThread,
  getWorkspaceThreads,
  reduceWorkspaceEvent,
  sendWorkspaceMessage,
  waitForWorkspaceSessionTerminal,
  WORKSPACE_IDLE_STATE,
  workspaceEndpoint,
  workspaceFollowUpDisposition,
  workspaceSessionMatchesRuntime,
} from '../lib/workspaceAgent'
import { Button } from '../components/ui/Button'
import { Modal } from '../components/ui/Modal'
import { AssistantMarkdown, copyText } from '../lib/markdown'
import {
  IconBolt, IconCheckCircle, IconClose, IconError, IconModels, IconPlay, IconReceipt, IconSearch, IconSend, IconStop,
} from '../components/ui/icons'

const PHASE_LABEL = {
  idle: 'Ready',
  starting: 'Starting',
  running: 'Running',
  finished: 'Complete',
  aborted: 'Stopped',
  cancelled: 'Stopped',
  step_capped: 'Step limit reached',
  repeated: 'No progress',
  driver_error: 'Model error',
  cancelling: 'Stopping',
  recovering: 'Recovering',
  cancel_error: 'Stop failed',
  error: 'Error',
}

// Human-readable outcome shown in the Result panel when a session ends WITHOUT a
// written answer, so the user gets a plain-language reason rather than a raw phase.
const TERMINAL_RESULT = {
  aborted: { title: 'Session stopped', detail: 'You stopped this session before it finished.' },
  cancelled: { title: 'Session stopped', detail: 'You stopped this session before it finished.' },
  step_capped: { title: 'Reached the step limit', detail: 'Camelid ran out of steps before finishing. Try a narrower goal or fewer files.' },
  repeated: { title: 'Stopped — no progress', detail: 'Camelid was repeating itself, so it stopped. Try rephrasing the goal.' },
  driver_error: { title: 'The model had a problem', detail: 'The model could not complete the task. Try again or pick a different goal.' },
  cancel_error: { title: 'Stop could not be confirmed', detail: 'The turn may still be running. Retry Stop before sending another request.' },
  error: { title: 'Something went wrong', detail: '' },
}
const DEFAULT_SETUP_PERCENT = 46
const MIN_SETUP_PX = 360
const MIN_ACTIVITY_PX = 400
const SPLITTER_PX = 10
const MAX_RENDERED_TURNS = 100

function initialSetupPercent() {
  const saved = Number.parseFloat(appStorage.getItem('camelid.workspaceSetupPercent') || '')
  return Number.isFinite(saved) ? saved : DEFAULT_SETUP_PERCENT
}

function clampSetupPercentForWidth(percent, width) {
  if (!width) return Math.min(75, Math.max(25, percent))
  const min = Math.min(50, (MIN_SETUP_PX / width) * 100)
  const max = Math.max(min, ((width - SPLITTER_PX - MIN_ACTIVITY_PX) / width) * 100)
  return Math.min(max, Math.max(min, percent))
}

function initialWorkspaceState() {
  return { ...WORKSPACE_IDLE_STATE, events: [], turns: [] }
}

function eventKey(event, index) {
  return `${event.sequence || 'local'}-${event.event}-${index}`
}

function workspaceFitLabel(fit) {
  if (fit === 'fits_resident') return 'Fits this machine'
  if (fit === 'fits_with_offload') return 'Fits with offload'
  if (fit === 'cpu_only_ok') return 'Fits on CPU'
  if (fit === 'wont_fit') return 'Too big for this machine'
  return 'Fit unknown'
}

function CompatibleModelRow({ model, onEvidence }) {
  return (
    <div className="workspace-compatible-model">
      <div className="workspace-compatible-model__identity">
        <strong>{model.name}</strong>
        <code title={model.filename}>{model.filename}</code>
      </div>
      <div className="workspace-compatible-model__meta">
        <span className={model.fit === 'wont_fit' ? 'is-bad' : ''}>
          {model.installed ? 'On disk' : model.catalog_id ? 'Available in Models' : 'Import exact file'} · {workspaceFitLabel(model.fit)}
        </span>
        <button type="button" className="workspace-evidence-link" onClick={() => onEvidence(model.row_id)} title={`View evidence for ${model.name}`}>
          <IconReceipt size={15} /> Evidence
        </button>
      </div>
    </div>
  )
}

function ActivityRow({ event }) {
  const kind = event.event
  if (kind === 'session.started') {
    return <li className="workspace-event workspace-event--system"><IconPlay size={16} /><div><strong>Session started</strong><span>{event.model_id}</span></div></li>
  }
  if (kind === 'tool.call') {
    return <li className="workspace-event workspace-event--tool"><IconBolt size={16} /><div><strong>Tool requested</strong><code>{event.detail}</code></div></li>
  }
  if (kind === 'memory.compacted') {
    return <li className="workspace-event workspace-event--system"><IconCheckCircle size={16} /><div><strong>Conversation compacted</strong><span>{event.archived_turns} turns moved out of recent context at {event.trigger_tokens} / {event.budget_total} tokens</span></div></li>
  }
  if (kind === 'tool.result') {
    const failed = event.outcome === 'error'
    return <li className={`workspace-event ${failed ? 'workspace-event--error' : 'workspace-event--result'}`}>{failed ? <IconError size={16} /> : <IconCheckCircle size={16} />}<div><strong>{failed ? 'Tool failed' : 'Tool complete'}</strong><span>{event.tool}</span><pre>{event.content}</pre></div></li>
  }
  if (kind === 'model.live' || kind === 'model.answer') {
    return <li className={`workspace-event workspace-event--model ${kind === 'model.live' ? 'is-live' : ''}`}><IconBolt size={16} /><div><strong>{kind === 'model.live' ? 'Model working' : 'Camelid'}</strong><pre>{event.content}</pre></div></li>
  }
  if (kind === 'session.finished') {
    return <li className="workspace-event workspace-event--system"><IconCheckCircle size={16} /><div><strong>Session finished</strong><span>{PHASE_LABEL[event.outcome] || event.outcome}</span></div></li>
  }
  if (kind === 'session.error') {
    return <li className="workspace-event workspace-event--error"><IconError size={16} /><div><strong>Session error</strong><span>{event.message}</span></div></li>
  }
  return <li className="workspace-event workspace-event--system"><IconBolt size={16} /><div><strong>Workspace</strong><span>{event.content || kind}</span></div></li>
}

/* Folder and up-arrow glyphs for the picker, drawn in the icons.jsx style
   (24-viewBox, currentColor fill). The shared set has no folder glyph, a
   magnifier would misread as search, and IconSend stays reserved for
   message-send actions — so these live with their only consumer. */
function IconFolder({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
      <path d="M10 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-8l-2-2z" />
    </svg>
  )
}
function IconArrowUp({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
      <path d="M12 4 6 10l1.4 1.4L11 7.8V20h2V7.8l3.6 3.6L18 10l-6-6z" />
    </svg>
  )
}

function FolderPicker({ apiBase, initialPath, onClose, onPick }) {
  const [view, setView] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const requestId = useRef(0)
  const abortRef = useRef(null)

  const load = useCallback((path, fallbackToRoots = false) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const id = ++requestId.current
    setLoading(true)
    setError('')
    browseWorkspaceFolders(apiBase, path, { signal: controller.signal })
      .then((data) => {
        if (id !== requestId.current) return
        setView(data)
        setLoading(false)
      })
      .catch((err) => {
        if (id !== requestId.current) return
        if (err.name === 'AbortError') return
        if (fallbackToRoots && path) { load(null); return }
        setError(err.message || 'Could not open that folder.')
        setLoading(false)
      })
  }, [apiBase])

  useEffect(() => {
    load(initialPath || null, true)
    return () => {
      requestId.current += 1
      abortRef.current?.abort()
    }
  }, [load, initialPath])

  const atRoots = Boolean(view && view.path === null)
  const canGoUp = Boolean(view && (view.parent !== null || (view.hasRoots && view.path !== null)))
  const goUp = () => {
    if (!view) return
    if (view.parent !== null) load(view.parent)
    else if (view.hasRoots) load(null)
  }

  return (
    <Modal
      open
      onClose={onClose}
      title="Choose workspace folder"
      labelledById="workspace-folder-title"
      size="md"
      footer={
        <div className="folder-picker__actions">
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" onClick={() => onPick(view?.path)} disabled={!view || view.path === null}>Use this folder</Button>
        </div>
      }
    >
      <div className="folder-picker">
        <div className="folder-picker__bar">
          <button type="button" className="folder-picker__up" onClick={goUp} disabled={!canGoUp}>
            <IconArrowUp size={16} /> Up
          </button>
          <code className="folder-picker__path">{atRoots ? 'This PC' : (view?.path || '…')}</code>
        </div>
        {error ? <p className="folder-picker__error">{error}</p> : null}
        <ul className="folder-picker__list">
          {loading ? (
            <li className="folder-picker__empty">Loading…</li>
          ) : view && view.entries.length ? (
            view.entries.map((entry) => (
              <li key={entry.path}>
                <button type="button" className="folder-picker__entry" onClick={() => load(entry.path)}>
                  <IconFolder size={16} /> <span>{entry.name}</span>
                </button>
              </li>
            ))
          ) : (
            <li className="folder-picker__empty">{atRoots ? 'No drives found.' : 'No subfolders here.'}</li>
          )}
        </ul>
        {view?.truncated ? <p className="folder-picker__note">Showing the first {view.entries.length} folders.</p> : null}
      </div>
    </Modal>
  )
}

function ContextInspector({ budget, timing, runtimeContext, compaction, busy, disabled, onCompact, onUndo }) {
  if (!budget) return null
  const promptUsed = Number(budget.prompt_tokens || 0)
  const generation = Number(budget.generation_tokens || 0)
  const total = Number(budget.budget_total || 0)
  const rows = [
    ['System instructions', budget.system_tokens_estimate],
    ['Tool definitions', budget.tool_definition_tokens_estimate],
    ['Messages', budget.message_tokens_estimate],
    ['Recent memory', budget.recent_memory_tokens_estimate],
    ['Retrieved memory', budget.retrieved_memory_tokens_estimate],
    ['Evidence memory', budget.evidence_memory_tokens_estimate],
    ['Tool results', budget.tool_result_tokens_estimate],
  ].filter(([, value]) => Number(value || 0) > 0)
  return (
    <details className="workspace-context-inspector">
      <summary title="Exact rendered prompt plus reserved generation">
        <span>Context</span>
        <progress value={promptUsed + generation} max={total} />
        <strong>{promptUsed + generation} / {total}</strong>
      </summary>
      <div className="workspace-context-inspector__panel">
        <div className="workspace-context-inspector__total">
          <span>Exact prompt</span><strong>{promptUsed}</strong>
          <span>Reserved response</span><strong>{generation}</strong>
          {timing ? <><span>Last model call</span><strong>{(timing.total_ms / 1000).toFixed(1)} s</strong></> : null}
          {timing?.ttft_ms != null ? <><span>First token</span><strong>{(timing.ttft_ms / 1000).toFixed(1)} s</strong></> : null}
          {runtimeContext?.resident_cuda ? <><span>Resident capacity</span><strong>{runtimeContext.resident_cuda.max_positions}</strong></> : null}
          {runtimeContext?.resident_cuda ? <><span>Layer placement</span><strong>{runtimeContext.resident_cuda.offloaded ? 'Offloaded' : 'Resident'}</strong></> : null}
        </div>
        <p>Estimated breakdown</p>
        <dl>
          {rows.map(([label, value]) => (
            <div key={label}><dt>{label}</dt><dd>{Number(value || 0)}</dd></div>
          ))}
        </dl>
        <div className="workspace-context-inspector__actions">
          <Button variant="outline" onClick={() => onCompact()} disabled={disabled || busy} loading={busy}>
            Compact conversation
          </Button>
          {Number(compaction?.compaction_count || 0) > 0 ? (
            <Button variant="ghost" onClick={() => onUndo()} disabled={disabled || busy}>Undo last</Button>
          ) : null}
        </div>
        <small>At 75% context use, Camelid compacts automatically after a completed turn. Raw history and lexical retrieval remain intact, and the last compaction can be undone.</small>
      </div>
    </details>
  )
}

export default function WorkspaceView({ apiBase, capabilities, selectedModel, runtime, setTab }) {
  const [workspacePath, setWorkspacePath] = useState(() => appStorage.getItem('camelid.workspacePath') || '')
  const [goal, setGoal] = useState('')
  const [followUp, setFollowUp] = useState('')
  const [savedThreads, setSavedThreads] = useState([])
  const [selectedThreadId, setSelectedThreadId] = useState('')
  const [previewedThreadId, setPreviewedThreadId] = useState('')
  const [threadPreviewLoading, setThreadPreviewLoading] = useState(false)
  const [threadPreviewError, setThreadPreviewError] = useState('')
  const [threadDeleteBusy, setThreadDeleteBusy] = useState(false)
  const [compactionBusy, setCompactionBusy] = useState(false)
  const [compaction, setCompaction] = useState(null)
  const [session, setSession] = useState(null)
  const [sessionRuntime, setSessionRuntime] = useState(null)
  const [state, dispatch] = useReducer(reduceWorkspaceEvent, undefined, initialWorkspaceState)
  const [browseOpen, setBrowseOpen] = useState(false)
  const [activityOpen, setActivityOpen] = useState(false)
  const [answerCopyState, setAnswerCopyState] = useState('idle')
  const [compatibleModels, setCompatibleModels] = useState([])
  const [compatibleModelsLoading, setCompatibleModelsLoading] = useState(true)
  const [compatibleModelsError, setCompatibleModelsError] = useState('')
  const [setupPercent, setSetupPercent] = useState(initialSetupPercent)
  const [resizing, setResizing] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const workspaceRef = useRef(null)
  const eventSourceRef = useRef(null)
  const sessionRef = useRef(null)
  const workspacePathRef = useRef(workspacePath)
  const sessionApiBaseRef = useRef(apiBase)
  const copyTimerRef = useRef(null)
  const intentionalClosuresRef = useRef(new WeakSet())
  const timelineRef = useRef(null)
  const mountedRef = useRef(false)
  const operationEpochRef = useRef(0)
  const pendingControllersRef = useRef(new Set())
  const startInFlightRef = useRef(false)
  const pendingStartRef = useRef(null)
  const stopInFlightRef = useRef(false)
  const followUpAttemptRef = useRef(null)
  const recoveringEpochRef = useRef(null)
  const hasLoadedModel = Boolean(runtime?.loaded_now)

  const compatibility = useMemo(
    () => hasLoadedModel ? findCompatibilityHint(capabilities, selectedModel, null) : null,
    [capabilities, hasLoadedModel, selectedModel],
  )
  const target = compatibility?.target || null
  const toolCapable = Boolean(hasLoadedModel && compatibility?.exact && target?.tool_capable && String(target.status || '').startsWith('supported'))
  const runtimeReady = runtime?.status === 'online' && runtime?.loaded_now && runtime?.generation_ready
  const running = stopPending || ['starting', 'running', 'cancelling', 'recovering', 'cancel_error'].includes(state.phase)
  const stopping = stopPending
  const sessionIdentityLocked = Boolean(session) || running
  const sessionBackendReady = !session || String(sessionApiBaseRef.current).replace(/\/$/, '') === String(apiBase).replace(/\/$/, '')
  const sessionModelReady = sessionBackendReady && workspaceSessionMatchesRuntime(session, runtime, toolCapable)
  const selectedThreadReady = !selectedThreadId || previewedThreadId === selectedThreadId
  const canStart = Boolean(workspacePath.trim() && goal.trim() && toolCapable && runtimeReady && !running && !session && selectedThreadReady && !threadPreviewLoading && !threadPreviewError && !threadDeleteBusy)
  const conversation = state.turns
  const visibleConversation = conversation.length > MAX_RENDERED_TURNS ? conversation.slice(-MAX_RENDERED_TURNS) : conversation
  const hiddenTurnCount = Math.max(0, Math.max(state.totalTurns || 0, conversation.length) - visibleConversation.length)
  const answers = conversation.map((turn) => turn.assistant).filter(Boolean)
  const finalAnswer = answers.at(-1) || ''
  const stepCount = useMemo(
    () => state.events.filter((event) => event.event === 'tool.call').length,
    [state.events],
  )
  const budget = useMemo(() => {
    for (let index = state.events.length - 1; index >= 0; index -= 1) {
      if (state.events[index].event === 'memory.updated') return state.events[index]
    }
    return null
  }, [state.events])
  const timing = useMemo(() => {
    for (let index = state.events.length - 1; index >= 0; index -= 1) {
      if (state.events[index].event === 'model.timing') return state.events[index]
    }
    return null
  }, [state.events])

  const isCurrentOperation = (epoch) => mountedRef.current && operationEpochRef.current === epoch
  const trackRequest = () => {
    const controller = new AbortController()
    pendingControllersRef.current.add(controller)
    return controller
  }
  const finishRequest = (controller) => pendingControllersRef.current.delete(controller)
  const abortTrackedRequests = (preserveControllers = []) => {
    const preserved = new Set(preserveControllers)
    for (const controller of pendingControllersRef.current) {
      if (!preserved.has(controller)) {
        controller.abort()
        pendingControllersRef.current.delete(controller)
      }
    }
  }
  const beginOperation = (preserveControllers = []) => {
    operationEpochRef.current += 1
    abortTrackedRequests(preserveControllers)
    recoveringEpochRef.current = null
    return operationEpochRef.current
  }
  const closeEventStream = (source = eventSourceRef.current) => {
    if (!source) return
    intentionalClosuresRef.current.add(source)
    source.close()
    if (eventSourceRef.current === source) eventSourceRef.current = null
  }
  const refreshSavedThreadsFor = async (path, epoch, requestApiBase = apiBase) => {
    if (!path || !isCurrentOperation(epoch)) return
    const controller = trackRequest()
    try {
      const threads = await getWorkspaceThreads(requestApiBase, path, { signal: controller.signal })
      if (isCurrentOperation(epoch) && workspacePathRef.current.trim() === path) setSavedThreads(threads)
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrentOperation(epoch) && workspacePathRef.current.trim() === path) setSavedThreads([])
    } finally {
      finishRequest(controller)
    }
  }
  const restoreDurableThread = async (created, epoch, requestApiBase = apiBase) => {
    const controller = trackRequest()
    try {
      const restored = await getWorkspaceThread(requestApiBase, created.workspace, created.id, { signal: controller.signal })
      if (!isCurrentOperation(epoch)) return false
      if (String(restored?.thread?.id || '') !== String(created.id)) {
        throw new Error('Saved Workspace thread returned a mismatched identity.')
      }
      dispatch({ event: 'thread.restored', turns: restored.turns, turnCount: restored.thread.turn_count })
      setCompaction({
        compacted_through_turn: restored.thread.compacted_through_turn,
        archived_turns: 0,
        compaction_count: restored.thread.compaction_count || 0,
      })
      return true
    } finally {
      finishRequest(controller)
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    setCompatibleModelsLoading(true)
    setCompatibleModelsError('')
    getWorkspaceCompatibleModels(apiBase, { signal: controller.signal })
      .then((models) => { if (!controller.signal.aborted) setCompatibleModels(models) })
      .catch((error) => {
        if (!controller.signal.aborted && error.name !== 'AbortError') setCompatibleModelsError('Compatible model details are unavailable from this running backend. Open Models to browse local and curated options.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setCompatibleModelsLoading(false)
      })
    return () => controller.abort()
  }, [apiBase])

  useEffect(() => {
    const path = workspacePath.trim()
    setSelectedThreadId('')
    setPreviewedThreadId('')
    setThreadPreviewError('')
    if (!path) { setSavedThreads([]); return undefined }
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      getWorkspaceThreads(apiBase, path, { signal: controller.signal })
        .then((threads) => {
          if (!controller.signal.aborted && workspacePathRef.current.trim() === path) setSavedThreads(threads)
        })
        .catch((error) => {
          if (!controller.signal.aborted && error.name !== 'AbortError' && workspacePathRef.current.trim() === path) setSavedThreads([])
        })
    }, 250)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [apiBase, workspacePath])

  useEffect(() => {
    if (session) return undefined
    const path = workspacePath.trim()
    if (!selectedThreadId || !path) {
      setPreviewedThreadId('')
      setThreadPreviewLoading(false)
      setThreadPreviewError('')
      dispatch({ event: 'session.reset' })
      setCompaction(null)
      return undefined
    }
    const controller = new AbortController()
    setPreviewedThreadId('')
    setThreadPreviewLoading(true)
    setThreadPreviewError('')
    getWorkspaceThread(apiBase, path, selectedThreadId, { signal: controller.signal })
      .then((restored) => {
        if (controller.signal.aborted) return
        dispatch({ event: 'thread.restored', turns: restored.turns, turnCount: restored.thread.turn_count })
        setCompaction({
          compacted_through_turn: restored.thread.compacted_through_turn,
          archived_turns: 0,
          compaction_count: restored.thread.compaction_count || 0,
        })
        setPreviewedThreadId(selectedThreadId)
      })
      .catch((error) => {
        if (!controller.signal.aborted && error.name !== 'AbortError') setThreadPreviewError(error.message || 'Saved conversation could not be loaded.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setThreadPreviewLoading(false)
      })
    return () => controller.abort()
  }, [apiBase, workspacePath, selectedThreadId, session])

  useEffect(() => {
    if (workspacePath.trim()) appStorage.setItem('camelid.workspacePath', workspacePath)
    else appStorage.removeItem('camelid.workspacePath')
  }, [workspacePath])

  useEffect(() => {
    sessionRef.current = session
    workspacePathRef.current = workspacePath
  }, [session, workspacePath])

  useEffect(() => {
    appStorage.setItem('camelid.workspaceSetupPercent', String(setupPercent))
  }, [setupPercent])

  useEffect(() => {
    const workspace = workspaceRef.current
    if (!workspace || typeof ResizeObserver === 'undefined') return undefined
    const observer = new ResizeObserver(([entry]) => {
      if (window.matchMedia('(max-width: 980px)').matches) return
      setSetupPercent((current) => clampSetupPercentForWidth(current, entry.contentRect.width))
    })
    observer.observe(workspace)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const live = state.events.at(-1)?.event === 'model.live'
    const frame = window.requestAnimationFrame(() => {
      timelineRef.current?.scrollTo({ top: timelineRef.current.scrollHeight, behavior: live ? 'auto' : 'smooth' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [state.events.length, state.events.at(-1)?.content?.length, state.phase])

  useEffect(() => {
    if (running) setActivityOpen(true)
    else if (state.phase === 'finished' && finalAnswer) setActivityOpen(false)
  }, [running, state.phase, finalAnswer])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      operationEpochRef.current += 1
      abortTrackedRequests()
      if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
      const activeSession = sessionRef.current
      sessionRef.current = null
      if (activeSession) cancelWorkspaceSession(sessionApiBaseRef.current, activeSession.id).catch(() => {})
      closeEventStream()
    }
  }, [])

  const settleBrokenStream = async (created, epoch, message, requestApiBase = apiBase) => {
    if (!isCurrentOperation(epoch) || recoveringEpochRef.current === epoch) return
    recoveringEpochRef.current = epoch
    closeEventStream()
    dispatch({ event: 'session.recovering', message })
    const controller = trackRequest()
    let terminal = null
    try {
      const status = await cancelWorkspaceSession(requestApiBase, created.id, { signal: controller.signal })
      terminal = status === 404
        ? { state: 'cancelled' }
        : await waitForWorkspaceSessionTerminal(requestApiBase, created.id, { signal: controller.signal })
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrentOperation(epoch)) {
        dispatch({ event: 'turn.stop_failed', message: `Workspace recovery could not confirm the turn stopped: ${error.message}` })
      }
      return
    } finally {
      finishRequest(controller)
    }
    if (!isCurrentOperation(epoch)) return
    setSessionRuntime(terminal)
    try { await restoreDurableThread(created, epoch, requestApiBase) } catch {}
    await refreshSavedThreadsFor(created.workspace, epoch, requestApiBase)
    if (isCurrentOperation(epoch)) {
      recoveringEpochRef.current = null
      dispatch({ event: 'session.error', message })
    }
  }

  const openEventStream = (created, epoch, requestApiBase = apiBase) => {
    if (!isCurrentOperation(epoch)) return
    closeEventStream()
    const url = workspaceEndpoint(requestApiBase, `/${encodeURIComponent(created.id)}/events`)
    let source
    try {
      source = new EventSource(url)
    } catch {
      void settleBrokenStream(created, epoch, 'The Workspace event stream could not be opened.', requestApiBase)
      return
    }
    eventSourceRef.current = source
    source.addEventListener('workspace', (message) => {
      if (eventSourceRef.current !== source || !isCurrentOperation(epoch)) return
      try {
        const envelope = JSON.parse(message.data)
        if (envelope.event === 'memory.compacted') setCompaction(envelope)
        dispatch(envelope)
        if (envelope.event === 'approval.required') {
          void settleBrokenStream(created, epoch, 'Read-only Workspace received an unexpected approval request.', requestApiBase)
          return
        }
        if (['session.finished', 'session.error'].includes(envelope.event)) {
          closeEventStream(source)
          const controller = trackRequest()
          getWorkspaceSession(requestApiBase, created.id, { signal: controller.signal })
            .then((status) => { if (isCurrentOperation(epoch)) setSessionRuntime(status) })
            .catch(() => {})
            .finally(() => finishRequest(controller))
          void refreshSavedThreadsFor(created.workspace, epoch, requestApiBase)
        }
      } catch {
        void settleBrokenStream(created, epoch, 'Camelid returned an unreadable Workspace event.', requestApiBase)
      }
    })
    source.onerror = () => {
      if (intentionalClosuresRef.current.has(source)) return
      if (eventSourceRef.current !== source || !isCurrentOperation(epoch)) return
      void settleBrokenStream(created, epoch, 'The Workspace event stream disconnected.', requestApiBase)
    }
  }

  const start = async () => {
    if (!canStart || startInFlightRef.current || sessionRef.current) return
    startInFlightRef.current = true
    const epoch = beginOperation()
    const requestApiBase = apiBase
    const requestedGoal = goal.trim()
    const controller = trackRequest()
    let settleAttempt
    const attempt = {
      controller,
      requestApiBase,
      stopRequested: false,
      result: new Promise((resolve) => { settleAttempt = resolve }),
    }
    pendingStartRef.current = attempt
    dispatch({ event: 'session.starting' })
    try {
      const created = await createWorkspaceSession(requestApiBase, {
        workspace: workspacePath.trim(),
        goal: requestedGoal,
        thread_id: selectedThreadId || undefined,
        max_steps: 12,
        max_tokens: 512,
        temperature: 0,
        allow_writes: false,
      }, { signal: controller.signal })
      settleAttempt({ created })
      if (!isCurrentOperation(epoch)) {
        if (!attempt.stopRequested && created?.id) cancelWorkspaceSession(requestApiBase, created.id).catch(() => {})
        return
      }
      if (!created?.id || !created?.workspace || !created?.model_id) {
        if (created?.id) cancelWorkspaceSession(requestApiBase, created.id).catch(() => {})
        throw new Error('Workspace start returned an incomplete session identity.')
      }
      sessionRef.current = created
      sessionApiBaseRef.current = requestApiBase
      workspacePathRef.current = created.workspace
      setWorkspacePath(created.workspace)
      setSession(created)
      setSessionRuntime(null)
      followUpAttemptRef.current = null
      dispatch({ event: 'turn.user', content: requestedGoal })
      openEventStream(created, epoch, requestApiBase)
    } catch (error) {
      settleAttempt({ error })
      if (error.name !== 'AbortError' && isCurrentOperation(epoch)) {
        dispatch({ event: 'session.error', message: error.message })
      }
    } finally {
      finishRequest(controller)
      startInFlightRef.current = false
      if (pendingStartRef.current === attempt) pendingStartRef.current = null
    }
  }

  const sendFollowUp = async () => {
    const boundSession = sessionRef.current
    const text = followUp.trim()
    if (!boundSession || !text || running || compactionBusy || !sessionBackendReady || !workspaceSessionMatchesRuntime(boundSession, runtime, toolCapable)) return
    let attempt = followUpAttemptRef.current
    if (!attempt || attempt.text !== text) {
      attempt = { text, id: window.crypto.randomUUID(), inFlight: false }
      followUpAttemptRef.current = attempt
    }
    if (attempt.inFlight) return
    attempt.inFlight = true
    const epoch = beginOperation()
    const requestApiBase = sessionApiBaseRef.current
    const controller = trackRequest()
    dispatch({ event: 'turn.starting' })
    const acceptStream = () => {
      if (!isCurrentOperation(epoch)) return
      dispatch({ event: 'turn.user', content: text })
      setFollowUp('')
      followUpAttemptRef.current = null
      openEventStream(boundSession, epoch, requestApiBase)
    }
    try {
      const response = await sendWorkspaceMessage(requestApiBase, boundSession.id, text, attempt.id, { signal: controller.signal })
      if (!isCurrentOperation(epoch)) return
      const disposition = workspaceFollowUpDisposition(response, boundSession.id)
      if (disposition === 'stream') {
        acceptStream()
      } else if (disposition === 'restore') {
        await restoreDurableThread(boundSession, epoch, requestApiBase)
        await refreshSavedThreadsFor(boundSession.workspace, epoch, requestApiBase)
        if (isCurrentOperation(epoch)) {
          setFollowUp('')
          followUpAttemptRef.current = null
        }
      } else {
        setFollowUp('')
        followUpAttemptRef.current = null
        await settleBrokenStream(boundSession, epoch, `Workspace returned an unsafe follow-up state (${response.state || 'unknown'}).`, requestApiBase)
      }
    } catch (error) {
      if (error.name === 'AbortError' || !isCurrentOperation(epoch)) return
      try {
        const status = await getWorkspaceSession(requestApiBase, boundSession.id, { signal: controller.signal })
        if (!isCurrentOperation(epoch)) return
        if (status.state === 'waiting_for_events') {
          acceptStream()
          return
        }
        if (['running', 'cancelling'].includes(status.state)) {
          setFollowUp('')
          followUpAttemptRef.current = null
          await settleBrokenStream(boundSession, epoch, 'Workspace could not safely resume the follow-up event stream.', requestApiBase)
          return
        }
      } catch (statusError) {
        if (statusError.name === 'AbortError' || !isCurrentOperation(epoch)) return
      }
      if (isCurrentOperation(epoch)) dispatch({ event: 'session.error', message: error.message })
    } finally {
      finishRequest(controller)
      if (followUpAttemptRef.current === attempt) attempt.inFlight = false
    }
  }

  const stop = async () => {
    if (stopInFlightRef.current) return
    const boundSession = sessionRef.current
    const pendingStart = !boundSession && startInFlightRef.current ? pendingStartRef.current : null
    if (!boundSession && !pendingStart && !running) return
    stopInFlightRef.current = true
    // The server can publish a session before its create response reaches the
    // browser. Preserve that request so Stop can learn the published ID and
    // cancel it instead of abandoning an unclaimed session on the backend.
    if (pendingStart) pendingStart.stopRequested = true
    const epoch = beginOperation(pendingStart ? [pendingStart.controller] : [])
    let sessionToCancel = boundSession
    let requestApiBase = boundSession ? sessionApiBaseRef.current : pendingStart?.requestApiBase || apiBase
    followUpAttemptRef.current = null
    closeEventStream()
    setStopPending(true)
    dispatch({ event: 'turn.stopping' })
    let controller = null
    try {
      if (!sessionToCancel && pendingStart) {
        const startResult = await pendingStart.result
        if (!isCurrentOperation(epoch)) return
        if (!startResult.created?.id) {
          dispatch({ event: 'session.finished', outcome: 'cancelled' })
          return
        }
        sessionToCancel = startResult.created
        requestApiBase = pendingStart.requestApiBase
      }
      if (!sessionToCancel) {
        dispatch({ event: 'session.finished', outcome: 'cancelled' })
        return
      }
      controller = trackRequest()
      const status = await cancelWorkspaceSession(requestApiBase, sessionToCancel.id, { signal: controller.signal })
      const terminal = status === 404
        ? { state: 'cancelled' }
        : await waitForWorkspaceSessionTerminal(requestApiBase, sessionToCancel.id, { signal: controller.signal })
      if (!isCurrentOperation(epoch)) return
      setSessionRuntime(terminal)
      dispatch({ event: 'session.finished', outcome: 'cancelled' })
      await refreshSavedThreadsFor(sessionToCancel.workspace || workspacePathRef.current.trim(), epoch, requestApiBase)
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrentOperation(epoch)) {
        dispatch({ event: 'turn.stop_failed', message: error.message })
      }
    } finally {
      if (controller) finishRequest(controller)
      if (isCurrentOperation(epoch)) setStopPending(false)
      stopInFlightRef.current = false
    }
  }

  const reset = async () => {
    const boundSession = sessionRef.current
    const refreshPath = boundSession?.workspace || workspacePathRef.current.trim()
    const cancellationApiBase = boundSession ? sessionApiBaseRef.current : apiBase
    const currentApiBase = apiBase
    const epoch = beginOperation()
    closeEventStream()
    sessionRef.current = null
    followUpAttemptRef.current = null
    setSession(null)
    setSessionRuntime(null)
    setStopPending(false)
    setSelectedThreadId('')
    setPreviewedThreadId('')
    setThreadPreviewError('')
    setFollowUp('')
    setCompaction(null)
    dispatch({ event: 'session.reset' })
    if (boundSession) {
      const controller = trackRequest()
      try { await cancelWorkspaceSession(cancellationApiBase, boundSession.id, { signal: controller.signal }) } catch {}
      finally { finishRequest(controller) }
    }
    await refreshSavedThreadsFor(refreshPath, epoch, currentApiBase)
  }

  const deleteSelectedThread = async () => {
    if (!selectedThreadId || threadDeleteBusy) return
    const selected = savedThreads.find((thread) => thread.id === selectedThreadId)
    const label = selected?.title || selected?.goal || selectedThreadId
    if (!window.confirm(`Delete the saved Workspace thread “${label}”? This cannot be undone.`)) return
    const epoch = operationEpochRef.current
    const controller = trackRequest()
    setThreadDeleteBusy(true)
    try {
      await deleteWorkspaceThread(apiBase, workspacePath.trim(), selectedThreadId, { signal: controller.signal })
      if (isCurrentOperation(epoch)) {
        setSavedThreads((threads) => threads.filter((thread) => thread.id !== selectedThreadId))
        setSelectedThreadId('')
      }
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrentOperation(epoch)) dispatch({ event: 'session.error', message: error.message })
    } finally {
      finishRequest(controller)
      if (isCurrentOperation(epoch)) setThreadDeleteBusy(false)
    }
  }

  const updateCompaction = async (undo = false) => {
    const boundSession = sessionRef.current
    if (!boundSession || running || compactionBusy) return
    const epoch = operationEpochRef.current
    const controller = trackRequest()
    setCompactionBusy(true)
    try {
      const result = await compactWorkspaceThread(sessionApiBaseRef.current, boundSession.workspace, boundSession.id, undo, { signal: controller.signal })
      if (!isCurrentOperation(epoch)) return
      setCompaction(result)
      dispatch({
        event: 'session.notice',
        content: undo
          ? `Restored ${result.archived_turns} compacted turns to recent context.`
          : `Compacted ${result.archived_turns} turns. Raw history remains searchable.`,
      })
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrentOperation(epoch)) dispatch({ event: 'session.error', message: error.message })
    } finally {
      finishRequest(controller)
      if (isCurrentOperation(epoch)) setCompactionBusy(false)
    }
  }

  const statusLabel = stopPending
    ? 'Stopping'
    : state.phase === 'idle'
    ? !toolCapable
      ? 'Model required'
      : runtimeReady
        ? 'Ready'
        : 'Model not ready'
    : PHASE_LABEL[state.phase] || state.phase
  const statusClass = stopPending ? 'cancelling' : state.phase === 'idle' && !toolCapable ? 'blocked' : state.phase

  const openEvidence = (rowId) => {
    window.dispatchEvent(new CustomEvent('camelid:open-ledger', { detail: { rowId } }))
  }
  const clampSetupPercent = (percent) => {
    const width = workspaceRef.current?.getBoundingClientRect().width || 0
    return clampSetupPercentForWidth(percent, width)
  }
  const resizeFromClientX = (clientX) => {
    const bounds = workspaceRef.current?.getBoundingClientRect()
    if (!bounds) return
    setSetupPercent(clampSetupPercent(((clientX - bounds.left) / bounds.width) * 100))
  }
  const startResize = (event) => {
    if (window.matchMedia('(max-width: 980px)').matches) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    setResizing(true)
    resizeFromClientX(event.clientX)
  }
  const moveResize = (event) => {
    if (resizing) resizeFromClientX(event.clientX)
  }
  const stopResize = (event) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    setResizing(false)
  }
  const resizeWithKeyboard = (event) => {
    let next = setupPercent
    if (event.key === 'ArrowLeft') next -= 4
    else if (event.key === 'ArrowRight') next += 4
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = 100
    else return
    event.preventDefault()
    setSetupPercent(clampSetupPercent(next))
  }
  const installedCompatibleModels = compatibleModels.filter((model) => model.installed)
  const featuredCompatibleModels = installedCompatibleModels.length
    ? installedCompatibleModels.slice(0, 2)
    : compatibleModels.slice(0, 2)
  const featuredFilenames = new Set(featuredCompatibleModels.map((model) => model.filename))
  const otherCompatibleModels = compatibleModels.filter((model) => !featuredFilenames.has(model.filename))

  const copyAnswer = async () => {
    const copied = await copyText(finalAnswer)
    if (!mountedRef.current) return
    setAnswerCopyState(copied ? 'copied' : 'failed')
    if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
    copyTimerRef.current = window.setTimeout(() => {
      copyTimerRef.current = null
      if (mountedRef.current) setAnswerCopyState('idle')
    }, 1500)
  }

  const renderResult = () => {
    if (threadPreviewLoading) {
      return (
        <div className="workspace-result__working" role="status">
          <span className="workspace-result__spinner" aria-hidden="true" />
          <strong>Loading saved conversation…</strong>
        </div>
      )
    }
    if (threadPreviewError) {
      return (
        <div className="workspace-result__status is-error" role="alert">
          <IconError size={20} />
          <strong>Conversation could not be loaded</strong>
          <span>{threadPreviewError}</span>
        </div>
      )
    }
    const terminalError = ['error', 'driver_error', 'cancel_error'].includes(state.phase)
    if (terminalError && conversation.length === 0) {
      const meta = TERMINAL_RESULT[state.phase]
      return (
        <div className="workspace-result__status is-error" role="alert">
          <IconError size={20} />
          <strong>{meta.title}</strong>
          <span>{state.error || meta.detail}</span>
        </div>
      )
    }
    if (running && conversation.length === 0) {
      return (
        <div className="workspace-result__working" role="status">
          <span className="workspace-result__spinner" aria-hidden="true" />
          <strong>Camelid is working…</strong>
          <span>Reading your files and preparing the answer. Watch each step under “What Camelid did”.</span>
        </div>
      )
    }
    if (conversation.length > 0) {
      const errorMeta = terminalError ? TERMINAL_RESULT[state.phase] : null
      return (
        <div className="workspace-conversation">
          {errorMeta ? (
            <div className="workspace-result__status is-error" role="alert">
              <IconError size={20} />
              <strong>{errorMeta.title}</strong>
              <span>{state.error || errorMeta.detail}</span>
            </div>
          ) : null}
          {hiddenTurnCount ? <p className="workspace-conversation__truncated">Showing the latest {visibleConversation.length} turns. {hiddenTurnCount} older turns remain saved in this conversation.</p> : null}
          {visibleConversation.map((turn, index) => (
            <article className="workspace-answer" key={`answer-${hiddenTurnCount + index}-${turn.assistant.length}`}>
              {turn.user ? <p className="workspace-answer__question">{turn.user}</p> : null}
              <div className="workspace-answer__bar">
                <span className="workspace-answer__label"><IconCheckCircle size={15} /> Answer {index + 1}</span>
                {turn.assistant && index === visibleConversation.length - 1 && !running ? (
                  <button type="button" className="workspace-answer__copy" onClick={copyAnswer} aria-live="polite">
                    {answerCopyState === 'copied' ? 'Copied' : answerCopyState === 'failed' ? 'Copy failed' : 'Copy'}
                  </button>
                ) : null}
              </div>
              <div className="workspace-answer__body">
                {turn.assistant
                  ? <AssistantMarkdown content={turn.assistant} />
                  : running && index === visibleConversation.length - 1
                    ? <span className="workspace-answer__pending">Camelid is working…</span>
                    : <span className="workspace-answer__pending">{PHASE_LABEL[turn.outcome] || 'No answer was saved.'}</span>}
              </div>
            </article>
          ))}
          {!running && session ? <form className="workspace-follow-up" onSubmit={(event) => { event.preventDefault(); sendFollowUp() }}>
            <label htmlFor="workspace-follow-up">Follow up</label>
            <div className="workspace-follow-up__control">
              <textarea
                id="workspace-follow-up"
                value={followUp}
                onChange={(event) => {
                  const value = event.target.value
                  if (followUpAttemptRef.current?.text !== value.trim()) followUpAttemptRef.current = null
                  setFollowUp(value)
                }}
                placeholder="Ask about this folder or an earlier finding…"
                rows={3}
                disabled={!sessionModelReady || compactionBusy}
                aria-describedby={!sessionModelReady ? 'workspace-follow-up-model-warning' : undefined}
              />
              <Button variant="primary" type="submit" disabled={!followUp.trim() || !sessionModelReady || compactionBusy}>
                <IconSend size={16} /> Send
              </Button>
            </div>
            {!sessionModelReady ? (
              <small id="workspace-follow-up-model-warning" role="status">
                {sessionBackendReady
                  ? `Reload the exact Workspace model used by this session (${session.model_id}) before sending a follow-up.`
                  : 'Return to the Camelid backend that created this session before sending a follow-up.'}
              </small>
            ) : null}
          </form> : null}
        </div>
      )
    }
    if (state.events.length === 0) {
      return (
        <div className="workspace-result__empty">
          <IconReceipt size={22} />
          <strong>Your answer will appear here</strong>
          <span>Pick a folder, describe what you want, and Start. Camelid reads what it needs and shows the result here.</span>
        </div>
      )
    }
    const meta = TERMINAL_RESULT[state.phase] || { title: 'Session finished', detail: 'Camelid finished without a written answer.' }
    return (
      <div className="workspace-result__status" role="status">
        <IconBolt size={20} />
        <strong>{meta.title}</strong>
        {meta.detail ? <span>{meta.detail}</span> : null}
      </div>
    )
  }

  return (
    <div
      ref={workspaceRef}
      className={`workspace-view${resizing ? ' is-resizing' : ''}`}
      style={{ '--workspace-setup-percent': `${setupPercent}%` }}
    >
      <section className="workspace-setup" aria-labelledby="workspace-heading">
        <div className="workspace-setup__heading">
          <div>
            <p className="workspace-kicker">Local file workspace</p>
            <h2 id="workspace-heading">Give Camelid a bounded task</h2>
          </div>
          <span className={`workspace-status is-${statusClass}`} role="status" aria-live="polite" aria-atomic="true">{statusLabel}</span>
        </div>

        <div className="workspace-model-line">
          <div className="workspace-model-line__identity">
            <span>Active model</span>
            <strong>{hasLoadedModel ? runtime?.active_model_id || selectedModel?.name || 'Loaded model' : 'No model loaded'}</strong>
          </div>
          <span className={`workspace-model-eligibility ${toolCapable ? 'is-ready' : 'is-blocked'}`}>
            {toolCapable ? <IconCheckCircle size={14} /> : <IconError size={14} />}
            {toolCapable ? 'Workspace ready' : hasLoadedModel ? 'Chat only' : 'Not loaded'}
          </span>
        </div>

        {!toolCapable && (
          <section className="workspace-prerequisite" aria-labelledby="workspace-model-requirement" role="status">
            <div className="workspace-prerequisite__head">
              <IconError size={18} />
              <div>
                <h3 id="workspace-model-requirement">{hasLoadedModel ? 'Choose a Workspace-ready model' : 'Load a Workspace-ready model'}</h3>
                <p>{hasLoadedModel
                  ? 'Your active model remains available for Chat, but it has no passing agent-evaluation receipt for Workspace.'
                  : 'Workspace needs an agent-evaluated model before a task can start.'}</p>
              </div>
            </div>

            <div className="workspace-compatible-models" aria-live="polite">
              {compatibleModelsLoading ? <p>Checking evaluated models…</p> : null}
              {!compatibleModelsLoading && compatibleModelsError ? <p>{compatibleModelsError}</p> : null}
              {!compatibleModelsLoading && !compatibleModelsError && compatibleModels.length === 0 ? (
                <p>This build does not advertise a Workspace-ready exact model.</p>
              ) : null}
              {featuredCompatibleModels.map((model) => (
                <CompatibleModelRow key={model.filename} model={model} onEvidence={openEvidence} />
              ))}
              {otherCompatibleModels.length > 0 ? (
                <details className="workspace-compatible-more">
                  <summary>{otherCompatibleModels.length} other evaluated {otherCompatibleModels.length === 1 ? 'model' : 'models'}</summary>
                  {otherCompatibleModels.map((model) => (
                    <CompatibleModelRow key={model.filename} model={model} onEvidence={openEvidence} />
                  ))}
                </details>
              ) : null}
            </div>

            <div className="workspace-prerequisite__actions">
              <Button variant="outline" onClick={() => setTab('library')}><IconModels size={16} /> {hasLoadedModel ? 'Switch in Models' : 'Load in Models'}</Button>
              <span>{compatibleModels.length ? 'Load an exact listed file, then return here.' : 'Browse local and curated models.'}</span>
            </div>
          </section>
        )}

        <div className="workspace-field">
          <span>Workspace folder</span>
          <div className="workspace-field__control">
            <input
              value={session?.workspace || workspacePath}
              onChange={(event) => {
                workspacePathRef.current = event.target.value
                setWorkspacePath(event.target.value)
              }}
              placeholder={navigator.platform?.startsWith('Win') ? 'C:\\projects\\example' : '/workspace/example'}
              disabled={sessionIdentityLocked}
              spellCheck="false"
              aria-label="Workspace folder"
            />
            <Button
              variant="outline"
              className="workspace-field__browse"
              icon={<IconSearch size={16} />}
              onClick={() => setBrowseOpen(true)}
              disabled={sessionIdentityLocked}
            >
              Browse…
            </Button>
          </div>
          <small>{session
            ? `This session is locked to ${session.workspace}. Clear activity before choosing another folder.`
            : 'Camelid canonicalizes this directory and rejects paths that leave it.'}</small>
        </div>
        {savedThreads.length > 0 && !session ? (
          <label className="workspace-field workspace-thread-picker">
            <span>Conversation</span>
            <div className="workspace-thread-picker__control">
              <select value={selectedThreadId} onChange={(event) => setSelectedThreadId(event.target.value)} disabled={running || threadDeleteBusy}>
                <option value="">Start a new conversation</option>
                {savedThreads.map((thread) => (
                  <option key={thread.id} value={thread.id} title={thread.title || 'Workspace conversation'}>
                    {thread.title || 'Workspace conversation'} · {thread.turn_count} {thread.turn_count === 1 ? 'turn' : 'turns'} · {new Date(thread.updated_at * 1000).toLocaleString()}
                  </option>
                ))}
              </select>
              <Button variant="ghost" onClick={deleteSelectedThread} disabled={!selectedThreadId || threadDeleteBusy} loading={threadDeleteBusy}>
                Delete
              </Button>
            </div>
            <small>Choose a saved conversation to preview it on the right, then enter the next goal and Resume &amp; send.</small>
          </label>
        ) : null}
        {browseOpen ? (
          <FolderPicker
            apiBase={apiBase}
            initialPath={workspacePath.trim() || null}
            onClose={() => setBrowseOpen(false)}
            onPick={(path) => {
              if (path) {
                workspacePathRef.current = path
                setWorkspacePath(path)
              }
              setBrowseOpen(false)
            }}
          />
        ) : null}

        <label className="workspace-field workspace-field--goal">
          <span>{selectedThreadId ? 'Next goal' : 'Goal'}</span>
          <textarea
            value={goal}
            onChange={(event) => setGoal(event.target.value)}
            placeholder="Review this folder, find why the tests fail, and propose the smallest repair."
            rows={4}
            disabled={sessionIdentityLocked}
          />
        </label>

        <div className="workspace-setup__actions">
          {running ? (
            <Button variant="outline" onClick={stop} disabled={stopping} loading={stopping}><IconStop size={17} /> {stopping ? 'Stopping' : 'Stop'}</Button>
          ) : !session ? (
            <Button variant="primary" onClick={start} disabled={!canStart}><IconPlay size={17} /> {selectedThreadId ? 'Resume & send' : 'Start Workspace'}</Button>
          ) : null}
          {!running && (session || state.events.length > 0) && <Button variant="ghost" onClick={reset}><IconClose size={17} /> Clear activity</Button>}
          <span>12 steps · read-only tools run automatically · files are never changed</span>
        </div>
      </section>

      <div
        className="workspace-splitter"
        role="separator"
        aria-label="Resize Workspace setup and activity panes"
        aria-orientation="vertical"
        aria-valuemin={Math.round(clampSetupPercent(0))}
        aria-valuemax={Math.round(clampSetupPercent(100))}
        aria-valuenow={Math.round(setupPercent)}
        tabIndex={0}
        onPointerDown={startResize}
        onPointerMove={moveResize}
        onPointerUp={stopResize}
        onPointerCancel={stopResize}
        onDoubleClick={() => setSetupPercent(clampSetupPercent(DEFAULT_SETUP_PERCENT))}
        onKeyDown={resizeWithKeyboard}
      />

      <section className="workspace-activity" aria-labelledby="workspace-result-heading">
        <div className="workspace-activity__header">
          <div>
            <p className="workspace-kicker">Result</p>
            <h2 id="workspace-result-heading">Answer</h2>
          </div>
          {session?.workspace && <code title={session.workspace}>{session.workspace}</code>}
          <ContextInspector
            budget={budget}
            timing={timing}
            runtimeContext={sessionRuntime}
            compaction={compaction}
            busy={compactionBusy}
            disabled={running || !session}
            onCompact={() => updateCompaction(false)}
            onUndo={() => updateCompaction(true)}
          />
        </div>

        <div className="workspace-result">
          {renderResult()}
        </div>

        {state.events.length > 0 ? (
          <details
            className="workspace-activity-details"
            open={activityOpen}
            onToggle={(event) => setActivityOpen(event.currentTarget.open)}
          >
            <summary className="workspace-activity-summary">
              <IconBolt size={15} />
              <span>What Camelid did</span>
              <span className="workspace-activity-count">{stepCount} {stepCount === 1 ? 'step' : 'steps'}</span>
            </summary>
            <div className="workspace-activity__scroll" ref={timelineRef} role="log" aria-live="polite" aria-relevant="additions">
              <ol className="workspace-timeline">
                {state.events.map((event, index) => <ActivityRow key={eventKey(event, index)} event={event} />)}
              </ol>
            </div>
          </details>
        ) : null}
      </section>
    </div>
  )
}
