import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { findCompatibilityHint } from '../lib/capabilities'
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
  const saved = Number.parseFloat(window.localStorage.getItem('camelid.workspaceSetupPercent') || '')
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

export function FolderPicker({ apiBase, initialPath, onClose, onPick }) {
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
    return () => abortRef.current?.abort()
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
            <IconSend size={16} /> Up
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
                  <IconSearch size={16} /> <span>{entry.name}</span>
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
  const [workspacePath, setWorkspacePath] = useState(() => window.localStorage.getItem('camelid.workspacePath') || '')
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
  const [answerCopied, setAnswerCopied] = useState(false)
  const [compatibleModels, setCompatibleModels] = useState([])
  const [compatibleModelsLoading, setCompatibleModelsLoading] = useState(true)
  const [compatibleModelsError, setCompatibleModelsError] = useState('')
  const [setupPercent, setSetupPercent] = useState(initialSetupPercent)
  const [resizing, setResizing] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const workspaceRef = useRef(null)
  const eventSourceRef = useRef(null)
  const sessionRef = useRef(null)
  const apiBaseRef = useRef(apiBase)
  const copyTimerRef = useRef(null)
  const intentionalClosuresRef = useRef(new WeakSet())
  const timelineRef = useRef(null)
  const hasLoadedModel = Boolean(runtime?.loaded_now)

  const compatibility = useMemo(
    () => hasLoadedModel ? findCompatibilityHint(capabilities, selectedModel, null) : null,
    [capabilities, hasLoadedModel, selectedModel],
  )
  const target = compatibility?.target || null
  const toolCapable = Boolean(hasLoadedModel && compatibility?.exact && target?.tool_capable && String(target.status || '').startsWith('supported'))
  const runtimeReady = runtime?.status === 'online' && runtime?.loaded_now && runtime?.generation_ready
  const running = stopPending || ['starting', 'running', 'cancelling', 'cancel_error'].includes(state.phase)
  const stopping = stopPending
  const selectedThreadReady = !selectedThreadId || previewedThreadId === selectedThreadId
  const canStart = Boolean(workspacePath.trim() && goal.trim() && toolCapable && runtimeReady && !running && !session && selectedThreadReady && !threadPreviewLoading && !threadPreviewError)
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

  useEffect(() => {
    const controller = new AbortController()
    setCompatibleModelsLoading(true)
    setCompatibleModelsError('')
    getWorkspaceCompatibleModels(apiBase, { signal: controller.signal })
      .then(setCompatibleModels)
      .catch((error) => {
        if (error.name !== 'AbortError') setCompatibleModelsError('Compatible model details are unavailable from this running backend. Open Models to browse local and curated options.')
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
        .then(setSavedThreads)
        .catch((error) => { if (error.name !== 'AbortError') setSavedThreads([]) })
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
        if (error.name !== 'AbortError') setThreadPreviewError(error.message || 'Saved conversation could not be loaded.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setThreadPreviewLoading(false)
      })
    return () => controller.abort()
  }, [apiBase, workspacePath, selectedThreadId, session])

  useEffect(() => {
    if (workspacePath) window.localStorage.setItem('camelid.workspacePath', workspacePath)
  }, [workspacePath])

  useEffect(() => {
    sessionRef.current = session
    apiBaseRef.current = apiBase
  }, [apiBase, session])

  useEffect(() => {
    window.localStorage.setItem('camelid.workspaceSetupPercent', String(setupPercent))
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
    timelineRef.current?.scrollTo({ top: timelineRef.current.scrollHeight, behavior: 'smooth' })
  }, [state.events.length, state.phase])

  useEffect(() => {
    if (running) setActivityOpen(true)
    else if (state.phase === 'finished' && finalAnswer) setActivityOpen(false)
  }, [running, state.phase, finalAnswer])

  useEffect(() => () => {
    if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
    if (sessionRef.current) {
      cancelWorkspaceSession(apiBaseRef.current, sessionRef.current.id).catch(() => {})
    }
    if (eventSourceRef.current) {
      intentionalClosuresRef.current.add(eventSourceRef.current)
      eventSourceRef.current.close()
    }
  }, [])

  const openEventStream = (created) => {
    const url = workspaceEndpoint(apiBase, `/${encodeURIComponent(created.id)}/events`)
    const source = new EventSource(url)
    eventSourceRef.current = source
    source.addEventListener('workspace', (message) => {
      if (eventSourceRef.current !== source) return
      try {
        const envelope = JSON.parse(message.data)
        if (envelope.event === 'memory.compacted') setCompaction(envelope)
        dispatch(envelope)
        if (['session.finished', 'session.error'].includes(envelope.event)) {
          getWorkspaceSession(apiBase, created.id).then(setSessionRuntime).catch(() => {})
          intentionalClosuresRef.current.add(source)
          source.close()
          eventSourceRef.current = null
        }
      } catch {
        dispatch({ event: 'session.error', message: 'Camelid returned an unreadable Workspace event.' })
        intentionalClosuresRef.current.add(source)
        source.close()
      }
    })
    source.onerror = () => {
      if (intentionalClosuresRef.current.has(source)) return
      if (eventSourceRef.current !== source) return
      dispatch({ event: 'session.error', message: 'The Workspace event stream disconnected.' })
      source.close()
      eventSourceRef.current = null
    }
  }

  const start = async () => {
    if (!canStart) return
    dispatch({ event: 'session.starting' })
    try {
      const created = await createWorkspaceSession(apiBase, {
        workspace: workspacePath.trim(),
        goal: goal.trim(),
        thread_id: selectedThreadId || undefined,
        max_steps: 12,
        max_tokens: 512,
        temperature: 0,
        allow_writes: false,
      })
      setSession(created)
      dispatch({ event: 'turn.user', content: goal.trim() })
      openEventStream(created)
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    }
  }

  const sendFollowUp = async () => {
    const text = followUp.trim()
    if (!session || !text || running) return
    dispatch({ event: 'turn.starting' })
    try {
      await sendWorkspaceMessage(apiBase, session.id, text, window.crypto.randomUUID())
      dispatch({ event: 'turn.user', content: text })
      setFollowUp('')
      openEventStream(session)
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    }
  }

  const stop = async () => {
    if (!session || stopping) return
    setStopPending(true)
    dispatch({ event: 'turn.stopping' })
    try {
      const status = await cancelWorkspaceSession(apiBase, session.id)
      if (status !== 404) await waitForWorkspaceSessionTerminal(apiBase, session.id)
      dispatch({ event: 'session.finished', outcome: 'cancelled' })
      if (eventSourceRef.current) {
        intentionalClosuresRef.current.add(eventSourceRef.current)
        eventSourceRef.current.close()
      }
      eventSourceRef.current = null
    } catch (error) {
      dispatch({ event: 'turn.stop_failed', message: error.message })
    } finally {
      setStopPending(false)
    }
  }

  const reset = async () => {
    if (session) {
      try { await cancelWorkspaceSession(apiBase, session.id) } catch {}
    }
    if (eventSourceRef.current) {
      intentionalClosuresRef.current.add(eventSourceRef.current)
      eventSourceRef.current.close()
    }
    eventSourceRef.current = null
    setSession(null)
    setSessionRuntime(null)
    setSelectedThreadId('')
    setPreviewedThreadId('')
    setThreadPreviewError('')
    setFollowUp('')
    setCompaction(null)
    dispatch({ event: 'session.reset' })
  }

  const deleteSelectedThread = async () => {
    if (!selectedThreadId || threadDeleteBusy) return
    setThreadDeleteBusy(true)
    try {
      await deleteWorkspaceThread(apiBase, workspacePath.trim(), selectedThreadId)
      setSavedThreads((threads) => threads.filter((thread) => thread.id !== selectedThreadId))
      setSelectedThreadId('')
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    } finally {
      setThreadDeleteBusy(false)
    }
  }

  const updateCompaction = async (undo = false) => {
    if (!session || running || compactionBusy) return
    setCompactionBusy(true)
    try {
      const result = await compactWorkspaceThread(apiBase, workspacePath.trim(), session.id, undo)
      setCompaction(result)
      dispatch({
        event: 'session.notice',
        content: undo
          ? `Restored ${result.archived_turns} compacted turns to recent context.`
          : `Compacted ${result.archived_turns} turns. Raw history remains searchable.`,
      })
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    } finally {
      setCompactionBusy(false)
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
    await copyText(finalAnswer)
    setAnswerCopied(true)
    if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
    copyTimerRef.current = window.setTimeout(() => {
      copyTimerRef.current = null
      setAnswerCopied(false)
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
        <div className="workspace-result__status is-error">
          <IconError size={20} />
          <strong>{meta.title}</strong>
          <span>{state.error || meta.detail}</span>
        </div>
      )
    }
    if (running && conversation.length === 0) {
      return (
        <div className="workspace-result__working">
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
            <div className="workspace-result__status is-error">
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
                  <button type="button" className="workspace-answer__copy" onClick={copyAnswer}>
                    {answerCopied ? 'Copied' : 'Copy'}
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
                onChange={(event) => setFollowUp(event.target.value)}
                placeholder="Ask about this folder or an earlier finding…"
                rows={3}
              />
              <Button variant="primary" type="submit" disabled={!followUp.trim()}>
                <IconSend size={16} /> Send
              </Button>
            </div>
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
      <div className="workspace-result__status">
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
          <span className={`workspace-status is-${statusClass}`}>{statusLabel}</span>
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
              value={workspacePath}
              onChange={(event) => setWorkspacePath(event.target.value)}
              placeholder={navigator.platform?.startsWith('Win') ? 'C:\\projects\\example' : '/workspace/example'}
              disabled={running}
              spellCheck="false"
              aria-label="Workspace folder"
            />
            <Button
              variant="outline"
              className="workspace-field__browse"
              icon={<IconSearch size={16} />}
              onClick={() => setBrowseOpen(true)}
              disabled={running}
            >
              Browse…
            </Button>
          </div>
          <small>Camelid canonicalizes this directory and rejects paths that leave it.</small>
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
            onPick={(path) => { if (path) setWorkspacePath(path); setBrowseOpen(false) }}
          />
        ) : null}

        <label className="workspace-field workspace-field--goal">
          <span>{selectedThreadId ? 'Next goal' : 'Goal'}</span>
          <textarea
            value={goal}
            onChange={(event) => setGoal(event.target.value)}
            placeholder="Review this folder, find why the tests fail, and propose the smallest repair."
            rows={4}
            disabled={running}
          />
        </label>

        <div className="workspace-setup__actions">
          {running ? (
            <Button variant="outline" onClick={stop} disabled={stopping} loading={stopping}><IconStop size={17} /> {stopping ? 'Stopping' : 'Stop'}</Button>
          ) : !session ? (
            <Button variant="primary" onClick={start} disabled={!canStart}><IconPlay size={17} /> {selectedThreadId ? 'Resume & send' : 'Start Workspace'}</Button>
          ) : null}
          {!running && state.events.length > 0 && <Button variant="ghost" onClick={reset}><IconClose size={17} /> Clear activity</Button>}
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
            <div className="workspace-activity__scroll" ref={timelineRef}>
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
