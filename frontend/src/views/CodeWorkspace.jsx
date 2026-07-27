import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { findCompatibilityHint } from '../lib/capabilities'
import {
  cancelWorkspaceSession,
  createWorkspaceSession,
  decideWorkspaceApproval,
  getWorkspaceChanges,
  getWorkspaceThread,
  getWorkspaceThreads,
  reduceCodeEvent,
  sendWorkspaceMessage,
  undoWorkspaceChange,
  waitForWorkspaceSessionTerminal,
  WORKSPACE_IDLE_STATE,
  workspaceEndpoint,
} from '../lib/workspaceAgent'
import { Button } from '../components/ui/Button'
import { Modal } from '../components/ui/Modal'
import { AssistantMarkdown } from '../lib/markdown'
import { FolderPicker } from './WorkspaceView'
import {
  IconBolt, IconCheckCircle, IconClose, IconEdit, IconError, IconPlay, IconSearch,
  IconSend, IconStop, IconWarning,
} from '../components/ui/icons'

const RUNNING_PHASES = new Set(['starting', 'running', 'awaiting_approval', 'cancelling', 'cancel_error'])
const PHASE_LABEL = {
  idle: 'Ready',
  starting: 'Starting',
  running: 'Working',
  awaiting_approval: 'Approval needed',
  cancelling: 'Stopping',
  finished: 'Complete',
  aborted: 'Stopped',
  cancelled: 'Stopped',
  step_capped: 'Step limit reached',
  repeated: 'No progress',
  driver_error: 'Model error',
  error: 'Error',
}

function initialCodeState() {
  return { ...WORKSPACE_IDLE_STATE, events: [], turns: [], approval: null }
}

function ActivityEvent({ event }) {
  if (event.event === 'tool.call') {
    return <li className="workspace-event workspace-event--tool"><IconBolt size={16} /><div><strong>Tool requested</strong><code>{event.detail}</code></div></li>
  }
  if (event.event === 'approval.required') {
    return <li className="workspace-event workspace-event--approval"><IconWarning size={16} /><div><strong>Waiting for approval</strong><span>{event.tool} · {event.risk}</span><pre>{event.detail}</pre></div></li>
  }
  if (event.event === 'tool.result') {
    const failed = event.outcome === 'error'
    return <li className={`workspace-event ${failed ? 'workspace-event--error' : 'workspace-event--result'}`}>{failed ? <IconError size={16} /> : <IconCheckCircle size={16} />}<div><strong>{failed ? 'Tool failed' : 'Tool complete'}</strong><span>{event.tool}</span><pre>{event.content}</pre></div></li>
  }
  if (event.event === 'model.live' || event.event === 'model.answer') {
    return <li className={`workspace-event workspace-event--model ${event.event === 'model.live' ? 'is-live' : ''}`}><IconBolt size={16} /><div><strong>{event.event === 'model.live' ? 'Camelid is working' : 'Camelid answered'}</strong><pre>{event.content}</pre></div></li>
  }
  if (event.event === 'session.error') {
    return <li className="workspace-event workspace-event--error"><IconError size={16} /><div><strong>Session error</strong><span>{event.message}</span></div></li>
  }
  if (event.event === 'session.finished') {
    return <li className="workspace-event workspace-event--system"><IconCheckCircle size={16} /><div><strong>Session finished</strong><span>{PHASE_LABEL[event.outcome] || event.outcome}</span></div></li>
  }
  if (event.event === 'session.notice') {
    return <li className="workspace-event workspace-event--system"><IconBolt size={16} /><div><strong>Agent notice</strong><span>{event.content}</span></div></li>
  }
  return null
}

export default function CodeWorkspace({
  apiBase,
  capabilities,
  selectedModel,
  runtime,
  setTab,
  requestedThread,
  onHistoryChanged,
}) {
  const [workspacePath, setWorkspacePath] = useState(() => window.localStorage.getItem('camelid.codeWorkspacePath') || '')
  const [goal, setGoal] = useState('')
  const [followUp, setFollowUp] = useState('')
  const [savedThreads, setSavedThreads] = useState([])
  const [selectedThreadId, setSelectedThreadId] = useState('')
  const [session, setSession] = useState(null)
  const [state, dispatch] = useReducer(reduceCodeEvent, undefined, initialCodeState)
  const [browseOpen, setBrowseOpen] = useState(false)
  const [changes, setChanges] = useState({ summary: 'no checkpoints this session', diff: 'no changes this session', files: [] })
  const [changesOpen, setChangesOpen] = useState(false)
  const [undoBusy, setUndoBusy] = useState(false)
  const [decisionBusy, setDecisionBusy] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const eventSourceRef = useRef(null)
  const sessionRef = useRef(null)
  const intentionalClosuresRef = useRef(new WeakSet())

  const hasLoadedModel = Boolean(runtime?.loaded_now)
  const compatibility = useMemo(
    () => hasLoadedModel ? findCompatibilityHint(capabilities, selectedModel, null) : null,
    [capabilities, hasLoadedModel, selectedModel],
  )
  const target = compatibility?.target || null
  const toolCapable = Boolean(hasLoadedModel && compatibility?.exact && target?.tool_capable && String(target.status || '').startsWith('supported'))
  const runtimeReady = runtime?.status === 'online' && runtime?.loaded_now && runtime?.generation_ready
  const running = stopPending || RUNNING_PHASES.has(state.phase)
  const canStart = Boolean(workspacePath.trim() && goal.trim() && toolCapable && runtimeReady && !running && !session)
  const finalAnswer = [...state.turns].reverse().find((turn) => turn.assistant)?.assistant || ''

  useEffect(() => {
    sessionRef.current = session
  }, [session])

  useEffect(() => {
    if (workspacePath) window.localStorage.setItem('camelid.codeWorkspacePath', workspacePath)
  }, [workspacePath])

  const refreshThreads = () => {
    const path = workspacePath.trim()
    if (!path) {
      setSavedThreads([])
      return
    }
    getWorkspaceThreads(apiBase, path, { mode: 'code' })
      .then(setSavedThreads)
      .catch(() => setSavedThreads([]))
  }

  useEffect(refreshThreads, [apiBase, workspacePath])

  useEffect(() => {
    if (!requestedThread?.id || session) return
    setWorkspacePath(requestedThread.canonical_root || '')
    setSelectedThreadId(requestedThread.id)
  }, [requestedThread, session])

  useEffect(() => {
    if (!selectedThreadId || !workspacePath.trim() || session) return
    getWorkspaceThread(apiBase, workspacePath.trim(), selectedThreadId)
      .then((restored) => {
        dispatch({ event: 'thread.restored', turns: restored.turns, turnCount: restored.thread.turn_count })
      })
      .catch((error) => dispatch({ event: 'session.error', message: error.message }))
  }, [apiBase, selectedThreadId, workspacePath, session])

  useEffect(() => () => {
    if (sessionRef.current) cancelWorkspaceSession(apiBase, sessionRef.current.id).catch(() => {})
    if (eventSourceRef.current) {
      intentionalClosuresRef.current.add(eventSourceRef.current)
      eventSourceRef.current.close()
    }
  }, [apiBase])

  const refreshChanges = (sessionId = session?.id) => {
    if (!sessionId) return
    getWorkspaceChanges(apiBase, sessionId).then(setChanges).catch(() => {})
  }

  const signalHistoryChanged = () => {
    refreshThreads()
    onHistoryChanged?.()
    window.dispatchEvent(new CustomEvent('camelid:code-history-changed'))
  }

  const openEventStream = (created) => {
    const source = new EventSource(workspaceEndpoint(apiBase, `/${encodeURIComponent(created.id)}/events`))
    eventSourceRef.current = source
    source.addEventListener('workspace', (message) => {
      if (eventSourceRef.current !== source) return
      try {
        const envelope = JSON.parse(message.data)
        dispatch(envelope)
        if (envelope.event === 'tool.result') refreshChanges(created.id)
        if (['session.finished', 'session.error'].includes(envelope.event)) {
          refreshChanges(created.id)
          signalHistoryChanged()
          intentionalClosuresRef.current.add(source)
          source.close()
          eventSourceRef.current = null
        }
      } catch {
        dispatch({ event: 'session.error', message: 'Camelid returned an unreadable Code-mode event.' })
        source.close()
        eventSourceRef.current = null
      }
    })
    source.onerror = () => {
      if (intentionalClosuresRef.current.has(source) || eventSourceRef.current !== source) return
      dispatch({ event: 'session.error', message: 'The Code-mode event stream disconnected.' })
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
        max_steps: 20,
        max_tokens: 768,
        temperature: 0,
        mode: 'code',
        allow_writes: true,
      })
      setSession(created)
      dispatch({ event: 'turn.user', content: goal.trim() })
      setGoal('')
      openEventStream(created)
      signalHistoryChanged()
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
    if (!session || stopPending) return
    setStopPending(true)
    dispatch({ event: 'turn.stopping' })
    try {
      await cancelWorkspaceSession(apiBase, session.id)
      await waitForWorkspaceSessionTerminal(apiBase, session.id)
      dispatch({ event: 'session.finished', outcome: 'cancelled' })
      signalHistoryChanged()
    } catch (error) {
      dispatch({ event: 'turn.stop_failed', message: error.message })
    } finally {
      setStopPending(false)
    }
  }

  const decide = async (decision) => {
    if (!session || !state.approval || decisionBusy) return
    setDecisionBusy(true)
    try {
      await decideWorkspaceApproval(apiBase, session.id, state.approval.approval_id, decision)
      dispatch({ event: 'approval.resolved' })
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    } finally {
      setDecisionBusy(false)
    }
  }

  const undo = async () => {
    if (!session || running || undoBusy) return
    setUndoBusy(true)
    try {
      const result = await undoWorkspaceChange(apiBase, session.id)
      setChanges(result)
      dispatch({ event: 'session.notice', content: result.result })
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    } finally {
      setUndoBusy(false)
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
    setSelectedThreadId('')
    setGoal('')
    setFollowUp('')
    setChanges({ summary: 'no checkpoints this session', diff: 'no changes this session', files: [] })
    dispatch({ event: 'session.reset' })
    signalHistoryChanged()
  }

  const status = stopPending ? 'Stopping' : PHASE_LABEL[state.phase] || state.phase

  return (
    <div className="workspace-view code-workspace">
      <section className="workspace-setup" aria-labelledby="code-heading">
        <div className="workspace-setup__heading">
          <div>
            <p className="workspace-kicker">Agent coding</p>
            <h2 id="code-heading">Work in a local codebase</h2>
          </div>
          <span className={`workspace-status is-${state.phase}`}>{status}</span>
        </div>

        <div className="workspace-model-line">
          <div className="workspace-model-line__identity">
            <span>Active model</span>
            <strong>{hasLoadedModel ? runtime?.active_model_id || selectedModel?.name || 'Loaded model' : 'No model loaded'}</strong>
          </div>
          <span className={`workspace-model-eligibility ${toolCapable ? 'is-ready' : 'is-blocked'}`}>
            {toolCapable ? <IconCheckCircle size={14} /> : <IconError size={14} />}
            {toolCapable ? 'Agent evaluated' : 'Code mode unavailable'}
          </span>
        </div>

        {!toolCapable ? (
          <section className="workspace-prerequisite" role="status">
            <div className="workspace-prerequisite__head"><IconError size={18} /><div><h3>Load an agent-evaluated model</h3><p>Code mode fails closed unless the exact active model row has a passing tool-capability receipt.</p></div></div>
            <div className="workspace-prerequisite__actions"><Button variant="outline" onClick={() => setTab('library')}>Open Models</Button></div>
          </section>
        ) : null}

        <div className="workspace-field">
          <span>Workspace folder</span>
          <div className="workspace-field__control">
            <input value={workspacePath} onChange={(event) => { setWorkspacePath(event.target.value); setSelectedThreadId('') }} disabled={running || Boolean(session)} spellCheck="false" placeholder={navigator.platform?.startsWith('Win') ? 'C:\\projects\\example' : '/workspace/example'} />
            <Button variant="outline" icon={<IconSearch size={16} />} onClick={() => setBrowseOpen(true)} disabled={running || Boolean(session)}>Browse…</Button>
          </div>
          <small>Reads and edits stay inside this canonical root. Network, GUI, MCP, and subagents are off.</small>
        </div>

        {savedThreads.length > 0 && !session ? (
          <label className="workspace-field workspace-thread-picker">
            <span>Coding session</span>
            <select value={selectedThreadId} onChange={(event) => setSelectedThreadId(event.target.value)}>
              <option value="">Start a new coding session</option>
              {savedThreads.map((thread) => <option key={thread.id} value={thread.id}>{thread.title} · {thread.turn_count} turns</option>)}
            </select>
          </label>
        ) : null}

        {browseOpen ? <FolderPicker apiBase={apiBase} initialPath={workspacePath.trim() || null} onClose={() => setBrowseOpen(false)} onPick={(path) => { if (path) setWorkspacePath(path); setBrowseOpen(false) }} /> : null}

        {!session ? (
          <label className="workspace-field workspace-field--goal">
            <span>{selectedThreadId ? 'Next task' : 'Task'}</span>
            <textarea value={goal} onChange={(event) => setGoal(event.target.value)} rows={6} placeholder="Inspect the project, implement the requested change, and run the relevant tests." disabled={running} />
          </label>
        ) : null}

        <div className="workspace-setup__actions">
          {running ? <Button variant="outline" onClick={stop} disabled={stopPending} loading={stopPending}><IconStop size={17} /> Stop</Button>
            : !session ? <Button variant="primary" onClick={start} disabled={!canStart}><IconPlay size={17} /> {selectedThreadId ? 'Resume session' : 'Start coding'}</Button>
              : <Button variant="ghost" onClick={reset}><IconClose size={17} /> New session</Button>}
          <span>20 steps · every write and shell command requires approval · checkpoints are automatic</span>
        </div>
      </section>

      <section className="workspace-activity code-activity" aria-labelledby="code-activity-heading">
        <div className="workspace-activity__header">
          <div><p className="workspace-kicker">Coding session</p><h2 id="code-activity-heading">Agent activity</h2></div>
          <div className="code-change-actions">
            <Button variant="ghost" size="sm" icon={<IconEdit size={15} />} onClick={() => { refreshChanges(); setChangesOpen(true) }} disabled={!session}>Changes ({changes.files?.length || 0})</Button>
            <Button variant="ghost" size="sm" onClick={undo} disabled={!session || running || !changes.files?.length || undoBusy} loading={undoBusy}>Undo last</Button>
          </div>
        </div>

        <div className="workspace-result code-session-body">
          {!state.turns.length && !state.events.length ? (
            <div className="workspace-result__empty"><IconBolt size={28} /><strong>Ready for a coding task</strong><span>Select a folder and describe the outcome you want. Camelid will inspect before editing and ask before every mutation.</span></div>
          ) : (
            <>
              <div className="code-conversation">
                {state.turns.map((turn, index) => (
                  <article className="workspace-answer" key={`${index}-${turn.user}`}>
                    {turn.user ? <p className="workspace-answer__question">{turn.user}</p> : null}
                    {turn.assistant ? <div className="workspace-answer__body"><AssistantMarkdown content={turn.assistant} /></div> : null}
                  </article>
                ))}
                {session && !running ? (
                  <form className="workspace-follow-up" onSubmit={(event) => { event.preventDefault(); sendFollowUp() }}>
                    <label htmlFor="code-follow-up">Follow up</label>
                    <div className="workspace-follow-up__control"><textarea id="code-follow-up" value={followUp} onChange={(event) => setFollowUp(event.target.value)} /><Button variant="primary" type="submit" disabled={!followUp.trim()}><IconSend size={16} /> Send</Button></div>
                  </form>
                ) : null}
              </div>
              <details className="workspace-activity-details" open={running || !finalAnswer}>
                <summary className="workspace-activity-summary"><IconBolt size={16} /> Activity <span className="workspace-activity-count">{state.events.length} events</span></summary>
                <div className="workspace-activity__scroll"><ol className="workspace-timeline">{state.events.map((event, index) => <ActivityEvent event={event} key={`${event.event}-${index}`} />)}</ol></div>
              </details>
            </>
          )}
        </div>
      </section>

      <Modal
        open={Boolean(state.approval)}
        onClose={() => decide('deny')}
        title="Review agent action"
        labelledById="code-approval-title"
        className="code-approval-modal"
        footer={<><Button variant="ghost" onClick={() => decide('abort')} disabled={decisionBusy}>Stop session</Button><Button variant="ghost" onClick={() => decide('deny')} disabled={decisionBusy}>Deny</Button><Button variant="outline" onClick={() => decide('always_tool')} disabled={decisionBusy}>Always allow {state.approval?.tool}</Button><Button variant="primary" onClick={() => decide('allow_once')} disabled={decisionBusy} loading={decisionBusy}>Allow once</Button></>}
      >
        <div className="code-approval">
          <div className="code-approval__risk"><IconWarning size={18} /><strong>{state.approval?.risk} action</strong></div>
          <pre>{state.approval?.detail}</pre>
          <p>This is the validated action resolved against the selected workspace. The browser cannot change its target.</p>
        </div>
      </Modal>

      <Modal open={changesOpen} onClose={() => setChangesOpen(false)} title="Session changes" labelledById="code-changes-title" size="lg">
        <div className="code-changes">
          <strong>{changes.summary}</strong>
          <pre>{changes.diff}</pre>
        </div>
      </Modal>
    </div>
  )
}
