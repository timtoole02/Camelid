import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { findCompatibilityHint } from '../lib/capabilities'
import {
  cancelWorkspaceSession,
  createWorkspaceSession,
  decideWorkspaceApproval,
  getWorkspaceChanges,
  getWorkspaceActivity,
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
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { AssistantMarkdown } from '../lib/markdown'
import { FolderPicker } from './WorkspaceView'
import {
  IconBolt, IconCheckCircle, IconChevronDown, IconChevronRight, IconClose, IconEdit,
  IconError, IconHistory, IconNetwork, IconPlay, IconRefresh, IconSearch, IconSend, IconSidebar,
  IconStop, IconWarning,
} from '../components/ui/icons'

// `cancel_error` is deliberately NOT here. It means a Stop could not be
// CONFIRMED, which is a warning, not a running turn — treating it as running
// left the composer dead and the only control a Stop button that had already
// failed, so the conversation could never be resumed.
const RUNNING_PHASES = new Set(['starting', 'running', 'awaiting_approval', 'cancelling'])
// Keyed by BOTH the reducer's phase and the server's terminal outcome string:
// the end-of-turn divider labels an outcome, and a normal completion reports
// `answered` while the phase it produces is `finished`. A missing key renders
// the raw token, which is how the commonest divider used to read "answered".
const PHASE_LABEL = {
  idle: 'Ready',
  starting: 'Starting',
  running: 'Working',
  awaiting_approval: 'Approval needed',
  cancelling: 'Stopping',
  cancel_error: 'Stop unconfirmed',
  answered: 'Complete',
  finished: 'Complete',
  aborted: 'Stopped',
  cancelled: 'Stopped',
  step_capped: 'Step limit reached',
  repeated: 'No progress',
  driver_error: 'Model error',
  failed: 'Error',
  error: 'Error',
}

/// How far from the bottom still counts as "following the stream".
const SCROLL_STICK_SLACK_PX = 96
/// Tool results can land in bursts; coalesce the /changes refetch they trigger.
const CHANGES_REFRESH_DEBOUNCE_MS = 400
/// How long to keep following a run whose live event stream dropped. Generous:
/// a local coding turn on a small GPU can legitimately run for many minutes.
const DETACHED_FOLLOW_TIMEOUT_MS = 30 * 60 * 1000
const DETACHED_FOLLOW_POLL_MS = 1500
/// Tail of the in-flight model output kept on screen while a step streams.
const LIVE_TAIL_CHARS = 2000
const SERVER_ACTIVE_STATES = new Set(['waiting_for_events', 'running', 'cancelling'])

const EMPTY_CHANGES = Object.freeze({
  summary: 'No checkpoints this session',
  diff: 'No changes this session',
  files: [],
})

function initialCodeState() {
  return { ...WORKSPACE_IDLE_STATE, events: [], turns: [], approval: null }
}

function formatToolName(tool) {
  return String(tool || 'tool')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function toolCallLabel(detail) {
  const value = String(detail || '').trim()
  const match = /^([a-zA-Z0-9_]+)\s*\(/.exec(value)
  return match ? formatToolName(match[1]) : 'Agent action'
}

function formatElapsed(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000))
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes < 60) return `${minutes}m ${seconds}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}

function compactPath(path) {
  const value = String(path || '')
  if (value.length <= 54) return value
  return `…${value.slice(-51)}`
}

// Events that mean this tool call will never get a result: another call started,
// the model answered, or the turn ended. Anything else — above all the
// `approval.required` card the default policy inserts — is scanned past.
const PAIRING_BARRIERS = new Set(['tool.call', 'model.answer', 'session.finished', 'session.error'])

function toolNameOf(detail) {
  const match = /^([a-zA-Z0-9_]+)\s*\(/.exec(String(detail || '').trim())
  return match ? match[1] : ''
}

// The agent runs one tool at a time, so the next `tool.result` before a barrier
// is this call's own. It is NOT necessarily the adjacent entry: in the default
// approval-gated mode the server emits `approval.required` between the call and
// its result, and pairing only on adjacency left every gated write showing a
// permanently 'Running' card beside a second, orphaned result card.
function findPairedResult(events, callIndex) {
  const tool = toolNameOf(events[callIndex].detail)
  for (let index = callIndex + 1; index < events.length; index += 1) {
    const candidate = events[index]
    if (candidate.event === 'tool.result') {
      return !tool || candidate.tool === tool ? index : -1
    }
    if (PAIRING_BARRIERS.has(candidate.event)) return -1
  }
  return -1
}

// Pair each tool call with its result, and stamp a key that survives the
// pairing. Grouping consumes two entries as one, so a raw array index shifts
// for every later item the moment a result lands — which remounts the rendered
// <details> cards and silently collapses whatever the user had expanded. The
// key is the reducer's per-arrival `uid`, not the envelope's `sequence`: the
// server counts sequences per event stream and every follow-up turn opens a new
// one, so sequences collide between the turns held in the same feed.
function groupActivityEvents(events) {
  const grouped = []
  const paired = new Set()
  for (let index = 0; index < events.length; index += 1) {
    if (paired.has(index)) continue
    const event = events[index]
    const key = event.uid != null ? `uid-${event.uid}` : `pos-${index}-${event.event}`
    const resultIndex = event.event === 'tool.call' ? findPairedResult(events, index) : -1
    if (resultIndex !== -1) paired.add(resultIndex)
    grouped.push({ key, event, pairedResult: resultIndex === -1 ? null : events[resultIndex] })
  }
  return grouped
}

// Tool-call syntax the model streams as ordinary tokens. It is not prose and
// must not be shown as the model's visible output: a Qwen/Hermes-style step
// renders as a raw `<tool_call>{"name":...}` blob, which reads as the agent
// losing its mind when in fact the call was parsed and executed normally. The
// call itself is surfaced properly by its own tool card, so strip it here.
const TOOL_CALL_BLOCK = /<tool_call>[\s\S]*?<\/tool_call>/g
const TOOL_CALL_OPEN = /<tool_call>[\s\S]*$/
// A bare JSON call, the Llama-style form, when it is the ONLY thing left.
const BARE_TOOL_CALL = /^\s*\{\s*"(?:name|type)"\s*:[\s\S]*$/
// The other shape small models emit: `list_dir({"path": ...})`, with no wrapper
// tag at all. Anchored, so prose that merely mentions a call is left alone.
const CALL_WITH_JSON = /^\s*[a-z_][a-z0-9_]*\s*\(\s*\{[\s\S]*$/i

function visibleLiveText(raw) {
  let text = String(raw || '').replace(TOOL_CALL_BLOCK, '')
  // A block still being generated: everything from the opening tag on is syntax.
  text = text.replace(TOOL_CALL_OPEN, '')
  if (BARE_TOOL_CALL.test(text) || CALL_WITH_JSON.test(text)) text = ''
  return text.trim()
}

function HistoricalTurn({ turn }) {
  return (
    <div className="code-turn-pair">
      {turn.user ? <article className="code-message code-message--user">{turn.user}</article> : null}
      {turn.assistant ? (
        <article className="code-message code-message--assistant">
          <div className="code-message__mark"><IconBolt size={16} /></div>
          <div className="code-message__content"><AssistantMarkdown content={turn.assistant} /></div>
        </article>
      ) : null}
    </div>
  )
}

function PlanUpdate({ content }) {
  const steps = String(content || '')
    .split(/\r?\n/)
    .map((line) => {
      const match = /^\[([x~ ])\]\s+(.+)$/.exec(line.trim())
      if (!match) return null
      return {
        status: match[1] === 'x' ? 'done' : match[1] === '~' ? 'active' : 'pending',
        text: match[2],
      }
    })
    .filter(Boolean)
  const active = steps.find((step) => step.status === 'active')
  const done = steps.filter((step) => step.status === 'done').length

  return (
    <article className="code-plan-update" aria-label="Camelid plan">
      <header>
        <span className="code-plan-update__icon"><IconHistory size={16} /></span>
        <span>
          <strong>Camelid&apos;s plan</strong>
          <small>{active ? `Working on: ${active.text}` : `${done} of ${steps.length} complete`}</small>
        </span>
      </header>
      {steps.length ? (
        <ol>
          {steps.map((step, index) => (
            <li className={`is-${step.status}`} key={`${index}-${step.text}`}>
              <span>{step.status === 'done' ? <IconCheckCircle size={14} /> : step.status === 'active' ? <IconPlay size={13} /> : index + 1}</span>
              <span>{step.text}</span>
            </li>
          ))}
        </ol>
      ) : <pre>{content}</pre>}
    </article>
  )
}

function LiveActivitySummary({ activity, running }) {
  if (!activity) return null
  const tokenText = Number.isFinite(activity.output_tokens)
    ? `${activity.output_tokens} tokens in the latest model step`
    : null
  return (
    <article className={`code-live-summary ${running ? 'is-running' : 'is-terminal'}`} role="status">
      <span className="code-live-summary__pulse">{running ? <span className="code-live-dot" /> : <IconCheckCircle size={15} />}</span>
      <div>
        <strong>{running ? 'Current activity' : (PHASE_LABEL[activity.phase] || 'Last activity')}</strong>
        <p>{activity.detail || 'Waiting for the next agent update'}</p>
        <small>{[activity.stage ? formatToolName(activity.stage) : null, tokenText].filter(Boolean).join(' · ')}</small>
      </div>
    </article>
  )
}

function ActivityEvent({ event, pairedResult, activeApproval, decisionBusy, onDecision }) {
  if (event.event === 'turn.user') {
    return <article className="code-message code-message--user">{event.content}</article>
  }

  if (event.event === 'model.live') {
    // Only the tail is rendered. This node re-renders on every generated token,
    // and a long step (a whole file inside a write_file argument) otherwise
    // grows an ever-taller <pre> that costs more to lay out with each token.
    const live = visibleLiveText(event.content)
    const tail = live.length > LIVE_TAIL_CHARS ? `…${live.slice(-LIVE_TAIL_CHARS)}` : live
    return (
      <article className="code-thinking-card" aria-live="polite">
        <header><span className="code-live-dot" /><strong>Camelid is working</strong></header>
        {tail ? <pre>{tail}</pre> : <p className="code-thinking-card__quiet">Preparing the next tool call…</p>}
      </article>
    )
  }

  if (event.event === 'model.answer') {
    return (
      <article className="code-message code-message--assistant">
        <div className="code-message__mark"><IconBolt size={16} /></div>
        <div className="code-message__content"><AssistantMarkdown content={event.content} /></div>
      </article>
    )
  }

  if (event.event === 'tool.call') {
    const result = pairedResult || null
    if (result && String(event.detail || '').startsWith('update_plan(') && result.outcome !== 'error') {
      return <PlanUpdate content={result.content} />
    }
    const failed = result?.outcome === 'error'
    return (
      <details className={`code-tool-card ${result ? (failed ? 'is-error' : 'is-complete') : ''}`} open={failed}>
        <summary>
          <span className="code-tool-card__icon">{result ? (failed ? <IconError size={15} /> : <IconCheckCircle size={15} />) : <IconChevronRight size={15} />}</span>
          <span><strong>{toolCallLabel(event.detail)}</strong><small>{result ? (failed ? 'Tool failed' : 'Tool completed') : 'Tool requested'}</small></span>
          <span className={`code-tool-card__state ${result ? (failed ? 'is-error' : 'is-complete') : 'is-running'}`}>{result ? (failed ? 'Failed' : 'Done') : 'Running'}</span>
        </summary>
        <pre>{result ? `${event.detail}\n\n${result.content}` : event.detail}</pre>
      </details>
    )
  }

  if (event.event === 'tool.result') {
    const failed = event.outcome === 'error'
    return (
      <details className={`code-tool-card ${failed ? 'is-error' : 'is-complete'}`} open={failed}>
        <summary>
          <span className="code-tool-card__icon">{failed ? <IconError size={15} /> : <IconCheckCircle size={15} />}</span>
          <span><strong>{formatToolName(event.tool)}</strong><small>{failed ? 'Tool failed' : 'Tool completed'}</small></span>
          <span className={`code-tool-card__state ${failed ? 'is-error' : 'is-complete'}`}>{failed ? 'Failed' : 'Done'}</span>
        </summary>
        <pre>{event.content}</pre>
      </details>
    )
  }

  if (event.event === 'approval.required') {
    const pending = activeApproval?.approval_id === event.approval_id
    return (
      <article className={`code-inline-approval ${pending ? 'is-pending' : 'is-resolved'}`}>
        <header>
          <span><IconWarning size={17} /></span>
          <div><strong>Review {formatToolName(event.tool)}</strong><small>{event.risk} action</small></div>
        </header>
        <pre>{event.detail}</pre>
        {pending ? (
          <div className="code-inline-approval__actions">
            <Button variant="ghost" size="sm" onClick={() => onDecision('abort')} disabled={decisionBusy}>Stop</Button>
            <Button variant="ghost" size="sm" onClick={() => onDecision('deny')} disabled={decisionBusy}>Deny</Button>
            <Button variant="outline" size="sm" onClick={() => onDecision('always_tool')} disabled={decisionBusy}>Always allow</Button>
            <Button variant="primary" size="sm" onClick={() => onDecision('allow_once')} loading={decisionBusy}>Allow once</Button>
          </div>
        ) : <p>Decision sent</p>}
      </article>
    )
  }

  if (event.event === 'model.timing') {
    const bits = [
      Number.isFinite(event.total_ms) ? `${(event.total_ms / 1000).toFixed(1)}s` : null,
      Number.isFinite(event.output_tokens) ? `${event.output_tokens} tokens` : null,
      Number.isFinite(event.ttft_ms) ? `${event.ttft_ms}ms to first token` : null,
    ].filter(Boolean)
    return <div className="code-meta-event"><IconBolt size={13} /><span>{bits.join(' · ')}</span></div>
  }

  if (event.event === 'memory.compacted') {
    return <div className="code-meta-event"><IconHistory size={13} /><span>Context compacted · {event.archived_turns || 0} turns archived</span></div>
  }

  if (event.event === 'session.notice') {
    return <div className="code-session-notice"><IconCheckCircle size={15} /><span>{event.content}</span></div>
  }

  if (event.event === 'session.error') {
    return <div className="code-session-notice is-error"><IconError size={15} /><span>{event.message}</span></div>
  }

  if (event.event === 'session.finished') {
    return (
      <div className="code-worked-divider">
        <span>{PHASE_LABEL[event.outcome] || event.outcome}</span>
      </div>
    )
  }

  return null
}

function CodeInspector({
  activity,
  agents,
  approvalMode,
  allowNetwork,
  changes,
  latestTool,
  latestResult,
  modelName,
  running,
  session,
  workspacePath,
  undoBusy,
  onClose,
  onRefreshChanges,
  onUndo,
}) {
  return (
    <aside className="code-inspector" aria-label="Coding session details">
      <header className="code-inspector__header">
        <div><span>Session</span><strong>Work details</strong></div>
        <button type="button" aria-label="Close work details" onClick={onClose}><IconClose size={18} /></button>
      </header>

      <section className="code-inspector__section">
        <div className="code-inspector__section-head">
          <h3>Changes <span>{changes.files?.length || 0}</span></h3>
          <button type="button" aria-label="Refresh changes" onClick={onRefreshChanges} disabled={!session}><IconRefresh size={15} /></button>
        </div>
        {changes.files?.length ? (
          <>
            <ul className="code-file-list">
              {changes.files.map((file) => <li key={file}><IconEdit size={14} /><span>{file}</span></li>)}
            </ul>
            <details className="code-patch">
              <summary><IconChevronRight size={14} /> View patch</summary>
              <pre>{changes.diff}</pre>
            </details>
            <Button variant="ghost" size="sm" onClick={onUndo} loading={undoBusy} disabled={running}>Undo last change</Button>
          </>
        ) : <p className="code-inspector__empty">No file changes yet.</p>}
      </section>

      <section className="code-inspector__section">
        <div className="code-inspector__section-head"><h3>Active work</h3></div>
        {activity ? (
          <div className={`code-process-row ${running ? 'is-running' : ''}`}>
            {running ? <span className="code-live-dot" /> : <IconCheckCircle size={15} />}
            <div><strong>{activity.stage ? formatToolName(activity.stage) : (running ? 'Working' : 'Finished')}</strong><small>{activity.detail || 'Waiting for an update'}</small></div>
          </div>
        ) : running && latestTool ? (
          <div className="code-process-row is-running">
            <span className="code-live-dot" />
            <div><strong>{toolCallLabel(latestTool.detail)}</strong><small>{String(latestTool.detail || '').slice(0, 110)}</small></div>
          </div>
        ) : latestResult ? (
          <div className="code-process-row">
            <IconCheckCircle size={15} />
            <div><strong>{formatToolName(latestResult.tool)}</strong><small>Last tool completed</small></div>
          </div>
        ) : <p className="code-inspector__empty">No active tool or command.</p>}
      </section>

      <section className="code-inspector__section">
        <div className="code-inspector__section-head"><h3>Agents <span>{agents.length}</span></h3></div>
        {agents.length ? (
          <ul className="code-agent-list">
            {agents.map((agent) => (
              <li key={agent.id} className={`is-${agent.status || 'running'}`}>
                <span className="code-agent-list__status"><span /></span>
                <div>
                  <strong>{agent.id === 'main' ? 'Primary agent' : agent.label}</strong>
                  <small className="code-agent-list__state">{formatToolName(agent.status || 'running')}</small>
                  <p>{agent.task || 'No scoped task reported'}</p>
                  {agent.detail ? <small>{agent.detail}</small> : null}
                </div>
              </li>
            ))}
          </ul>
        ) : <p className="code-inspector__empty">Agents will appear when a coding task starts.</p>}
      </section>

      <section className="code-inspector__section">
        <div className="code-inspector__section-head"><h3>Context</h3></div>
        <dl className="code-context-list">
          <div><dt>Workspace</dt><dd title={workspacePath}>{workspacePath ? compactPath(workspacePath) : 'Not selected'}</dd></div>
          <div><dt>Model</dt><dd>{modelName || 'No model loaded'}</dd></div>
          <div><dt>Access</dt><dd>{approvalMode === 'full_auto' ? 'Full auto' : 'Approval gated'}</dd></div>
          <div><dt>Web tools</dt><dd>{allowNetwork ? 'On · search and fetch enabled' : 'Off'}</dd></div>
        </dl>
      </section>
    </aside>
  )
}

export default function CodeWorkspace({
  apiBase,
  capabilities,
  selectedModel,
  runtime,
  setTab,
  requestedThread,
  onHistoryChanged,
  onRunningChange,
}) {
  const [workspacePath, setWorkspacePath] = useState(() => window.localStorage.getItem('camelid.codeWorkspacePath') || '')
  const [goal, setGoal] = useState('')
  const [followUp, setFollowUp] = useState('')
  const [savedThreads, setSavedThreads] = useState([])
  const [selectedThreadId, setSelectedThreadId] = useState('')
  const [session, setSession] = useState(null)
  const [state, dispatch] = useReducer(reduceCodeEvent, undefined, initialCodeState)
  const [browseOpen, setBrowseOpen] = useState(false)
  const [changes, setChanges] = useState({ ...EMPTY_CHANGES })
  const [undoBusy, setUndoBusy] = useState(false)
  const [decisionBusy, setDecisionBusy] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const [approvalMode, setApprovalMode] = useState('approval_gated')
  const [allowNetwork, setAllowNetwork] = useState(false)
  const [accessMenuOpen, setAccessMenuOpen] = useState(false)
  const [fullAutoConfirmOpen, setFullAutoConfirmOpen] = useState(false)
  const [inspectorOpen, setInspectorOpen] = useState(() => window.localStorage.getItem('camelid.codeInspectorOpen') !== 'false')
  const [startedAt, setStartedAt] = useState(null)
  const [clock, setClock] = useState(Date.now())
  const eventSourceRef = useRef(null)
  const sessionRef = useRef(null)
  const intentionalClosuresRef = useRef(new WeakSet())
  // Set once a terminal event for the current turn has been delivered, so a
  // Stop that lands alongside the server's own `aborted` does not append a
  // second divider for the same ending.
  const terminalHandledRef = useRef(false)
  // The rail request is applied ONCE. The effect that applies it re-runs
  // whenever `session` drops back to null, so re-applying there is what made
  // the header's New session button restore the thread it had just cleared.
  const consumedThreadRef = useRef('')
  // Identifies the create request a Stop can abandon while it is still in
  // flight — before there is any session id to cancel.
  const pendingStartRef = useRef(null)
  const activityRecoverySuppressedRef = useRef(false)
  const threadRef = useRef(null)
  const accessMenuRef = useRef(null)
  const stickToBottomRef = useRef(true)
  const changesTimerRef = useRef(null)

  const hasLoadedModel = Boolean(runtime?.loaded_now)
  const compatibility = useMemo(
    () => hasLoadedModel ? findCompatibilityHint(capabilities, selectedModel, null) : null,
    [capabilities, hasLoadedModel, selectedModel],
  )
  const target = compatibility?.target || null
  const toolCapable = Boolean(hasLoadedModel && compatibility?.exact && target?.tool_capable && String(target.status || '').startsWith('supported'))
  const runtimeReady = runtime?.status === 'online' && runtime?.loaded_now && runtime?.generation_ready
  const running = stopPending || RUNNING_PHASES.has(state.phase)
  // A Stop that could not be CONFIRMED: the composer is usable again, but the
  // run may still be executing server-side, so Stop stays offered as a retry.
  const stopUnconfirmed = state.phase === 'cancel_error' && !running
  const canStart = Boolean(workspacePath.trim() && goal.trim() && toolCapable && runtimeReady && !running && !session)
  const modelName = hasLoadedModel ? runtime?.active_model_id || selectedModel?.name || 'Loaded model' : ''
  const composerValue = session ? followUp : goal
  const canSubmit = session
    ? Boolean(followUp.trim() && !running)
    : canStart
  // Everything derived from the event list is computed ONCE per event, not per
  // render. Streamed deltas re-render this component on every token, and the
  // agent's event channel blocks on the consumer — an unmemoized scan here is
  // backpressure on the model's decode loop, not just UI jank.
  // `state.liveTurns` is counted by the reducer rather than recovered from the
  // event list: the activity buffer is a ring, `turn.user` is the first entry it
  // evicts, and a Code turn has no step cap to bound how long it runs. Inferring
  // it leaked a still-live turn into the history list, where its answer rendered
  // a second time beside the one still in the buffer.
  const historicalTurns = useMemo(
    () => state.turns.slice(0, Math.max(0, state.turns.length - (state.liveTurns || 0))),
    [state.liveTurns, state.turns],
  )
  const latestTool = state.latestTool
  const latestResult = state.latestResult
  const activity = state.liveActivity
  const agents = state.agents || []
  const feedEvents = useMemo(() => groupActivityEvents(state.events), [state.events])
  // Gated on `selectedThreadId` so a cleared session stops borrowing the title
  // of the rail entry App is still holding.
  const selectedThread = selectedThreadId
    ? savedThreads.find((thread) => thread.id === selectedThreadId) || requestedThread || null
    : null
  const title = selectedThread?.title
    || activity?.task?.slice(0, 72)
    || state.turns.find((turn) => turn.user)?.user?.slice(0, 72)
    || 'New coding session'
  const elapsed = startedAt ? formatElapsed(clock - startedAt) : ''

  useEffect(() => {
    sessionRef.current = session
  }, [session])

  useEffect(() => {
    if (workspacePath) window.localStorage.setItem('camelid.codeWorkspacePath', workspacePath)
  }, [workspacePath])

  // SSE is the rich transcript, but this backend-owned snapshot is the durable
  // truth for "what is it doing right now?". It lets a remounted Code page
  // recover the active task instead of displaying an unrelated blank composer.
  useEffect(() => {
    const controller = new AbortController()
    let polling = false
    const refreshActivity = async () => {
      if (polling) return
      polling = true
      try {
        const next = await getWorkspaceActivity(apiBase, { signal: controller.signal })
        if (!next || next.mode !== 'code') return
        const current = sessionRef.current
        const sameSession = current?.id === next.id
        const liveSession = SERVER_ACTIVE_STATES.has(next.state)
        if (!sameSession && (!liveSession || activityRecoverySuppressedRef.current)) return
        dispatch({ event: 'activity.snapshot', activity: next })
        if (!current && liveSession) {
          sessionRef.current = next
          setSession(next)
          setWorkspacePath(next.workspace || '')
          setSelectedThreadId(next.id)
          setApprovalMode(next.approval_mode || 'approval_gated')
          setAllowNetwork(Boolean(next.allow_network))
          setStartedAt(Number(next.started_at_ms) || Date.now())
          setClock(Date.now())
          getWorkspaceChanges(apiBase, next.id).then(setChanges).catch(() => {})
        }
      } catch (error) {
        if (error?.name !== 'AbortError') {
          // The live event stream remains authoritative while this optional
          // recovery poll is unavailable; do not turn a transient poll failure
          // into a fake session failure.
        }
      } finally {
        polling = false
      }
    }
    refreshActivity()
    const timer = window.setInterval(refreshActivity, 1000)
    return () => {
      controller.abort()
      window.clearInterval(timer)
    }
  }, [apiBase])

  useEffect(() => {
    window.localStorage.setItem('camelid.codeInspectorOpen', String(inspectorOpen))
  }, [inspectorOpen])

  useEffect(() => {
    if (!accessMenuOpen) return undefined
    const close = (event) => {
      if (!accessMenuRef.current?.contains(event.target)) setAccessMenuOpen(false)
    }
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setAccessMenuOpen(false)
    }
    window.addEventListener('pointerdown', close)
    window.addEventListener('keydown', closeOnEscape)
    return () => {
      window.removeEventListener('pointerdown', close)
      window.removeEventListener('keydown', closeOnEscape)
    }
  }, [accessMenuOpen])

  useEffect(() => {
    if (!running || !startedAt) return undefined
    const timer = window.setInterval(() => setClock(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [running, startedAt])

  // Track whether the user is parked at the bottom. Auto-scrolling someone who
  // has deliberately scrolled up to read an earlier tool result is worse than
  // not scrolling at all, so a scroll away from the bottom detaches the follow
  // and returning to the bottom re-attaches it.
  useEffect(() => {
    const thread = threadRef.current
    if (!thread) return undefined
    const onScroll = () => {
      const distance = thread.scrollHeight - thread.scrollTop - thread.clientHeight
      stickToBottomRef.current = distance <= SCROLL_STICK_SLACK_PX
    }
    thread.addEventListener('scroll', onScroll, { passive: true })
    return () => thread.removeEventListener('scroll', onScroll)
  }, [])

  // Keyed on `state.revision`, NOT `state.events.length`: a streamed delta is
  // merged into the existing tail entry, so the length never changes while text
  // is arriving and this effect used to sit out the entire stream. `auto` — not
  // `smooth` — because a smooth scroll retargeted every few milliseconds never
  // reaches the bottom.
  useEffect(() => {
    if (!threadRef.current || !stickToBottomRef.current) return undefined
    const frame = window.requestAnimationFrame(() => {
      const thread = threadRef.current
      if (thread) thread.scrollTo({ top: thread.scrollHeight, behavior: 'auto' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [state.revision, historicalTurns.length])

  const refreshThreads = (signal) => {
    const path = workspacePath.trim()
    if (!path) {
      setSavedThreads([])
      return
    }
    getWorkspaceThreads(apiBase, path, { mode: 'code', signal })
      .then((threads) => {
        if (!signal?.aborted) setSavedThreads(threads)
      })
      .catch(() => {
        if (!signal?.aborted) setSavedThreads([])
      })
  }

  // Aborted on change: the path is a text input, so typing fires one request
  // per keystroke and a slow early response could otherwise land last and
  // overwrite the list for the path the user actually settled on.
  useEffect(() => {
    const controller = new AbortController()
    refreshThreads(controller.signal)
    return () => controller.abort()
  }, [apiBase, workspacePath])

  useEffect(() => {
    if (!requestedThread?.id || session) return
    if (consumedThreadRef.current === requestedThread.id) return
    consumedThreadRef.current = requestedThread.id
    setWorkspacePath(requestedThread.canonical_root || '')
    setSelectedThreadId(requestedThread.id)
  }, [requestedThread, session])

  // App owns the rail actions that remount this component, and a remount runs
  // the unmount cleanup below — a real server-side cancel. It has to know a turn
  // is live so it can ask before ending one.
  useEffect(() => {
    onRunningChange?.(running)
    return () => onRunningChange?.(false)
  }, [onRunningChange, running])

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
    if (changesTimerRef.current) window.clearTimeout(changesTimerRef.current)
  }, [apiBase])

  const refreshChanges = (sessionId = session?.id) => {
    if (!sessionId) return Promise.resolve()
    return getWorkspaceChanges(apiBase, sessionId).then(setChanges).catch(() => {})
  }

  // A run that touches many files emits a tool.result per edit, and each one
  // used to trigger a full /changes fetch that re-serializes the entire diff.
  // Coalesce the bursts; the trailing call still reflects the final state.
  const scheduleChangesRefresh = (sessionId) => {
    if (changesTimerRef.current) window.clearTimeout(changesTimerRef.current)
    changesTimerRef.current = window.setTimeout(() => {
      changesTimerRef.current = null
      refreshChanges(sessionId)
    }, CHANGES_REFRESH_DEBOUNCE_MS)
  }

  const signalHistoryChanged = () => {
    refreshThreads()
    onHistoryChanged?.()
    window.dispatchEvent(new CustomEvent('camelid:code-history-changed'))
  }

  const openEventStream = (created) => {
    terminalHandledRef.current = false
    const source = new EventSource(workspaceEndpoint(apiBase, `/${encodeURIComponent(created.id)}/events`))
    eventSourceRef.current = source
    source.addEventListener('workspace', (message) => {
      if (eventSourceRef.current !== source) return
      try {
        const envelope = JSON.parse(message.data)
        dispatch(envelope)
        if (envelope.event === 'tool.result') scheduleChangesRefresh(created.id)
        if (['session.finished', 'session.error'].includes(envelope.event)) {
          terminalHandledRef.current = true
          if (changesTimerRef.current) {
            window.clearTimeout(changesTimerRef.current)
            changesTimerRef.current = null
          }
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
      // The server's /events claim is one-shot: an EventSource that reconnects
      // after any blip gets a 409, which is fatal to the EventSource. Closing it
      // here keeps that 409 out of the picture, but it does not save the run —
      // the response carries a cancel-on-drop guard, so losing the stream
      // cancels the turn server-side. All that is left to do is find out what it
      // managed to finish first.
      intentionalClosuresRef.current.add(source)
      source.close()
      eventSourceRef.current = null
      followDetachedSession(created)
    }
  }

  // The outcome the SERVER recorded for the turn, which is the only account of a
  // run whose event stream we lost. It is written before the session leaves its
  // running state, and a Code session id is also its thread id.
  const readRecordedOutcome = async (created) => {
    try {
      const restored = await getWorkspaceThread(apiBase, workspacePath.trim(), created.id)
      const last = Array.isArray(restored?.turns) ? restored.turns.at(-1) : null
      return String(last?.terminal_outcome || 'aborted')
    } catch {
      return 'aborted'
    }
  }

  const followDetachedSession = async (created) => {
    dispatch({
      event: 'session.notice',
      content: 'Lost the live activity stream. Camelid stops a run whose stream drops, so this turn is ending — reading the outcome it recorded.',
    })
    try {
      await waitForWorkspaceSessionTerminal(apiBase, created.id, {
        timeoutMs: DETACHED_FOLLOW_TIMEOUT_MS,
        pollMs: DETACHED_FOLLOW_POLL_MS,
      })
      terminalHandledRef.current = true
      dispatch({ event: 'session.finished', outcome: await readRecordedOutcome(created) })
    } catch (error) {
      terminalHandledRef.current = true
      dispatch({ event: 'session.error', message: error.message })
    }
    refreshChanges(created.id)
    signalHistoryChanged()
  }

  const start = async () => {
    if (!canStart) return
    activityRecoverySuppressedRef.current = false
    const pending = { abandoned: false }
    pendingStartRef.current = pending
    dispatch({ event: 'session.starting', task: goal.trim() })
    setStartedAt(Date.now())
    setClock(Date.now())
    try {
      const created = await createWorkspaceSession(apiBase, {
        workspace: workspacePath.trim(),
        goal: goal.trim(),
        thread_id: selectedThreadId || undefined,
        // A single write_file argument routinely carries a whole source file.
        // At the old 768 the call was cut off mid-JSON, parsed as no call, and
        // surfaced as a mangled "answer" while the write was silently dropped.
        // The backend clamps this down to whatever context headroom is left.
        max_tokens: 2048,
        temperature: 0,
        mode: 'code',
        allow_writes: true,
        approval_mode: approvalMode,
        allow_network: allowNetwork,
      })
      if (pending.abandoned) {
        // Stop was pressed while this request was in flight. The session exists
        // now, so cancel it rather than adopting a run the user has already
        // walked away from — and never open a stream for it.
        try { await cancelWorkspaceSession(apiBase, created.id) } catch {}
        dispatch({ event: 'session.finished', outcome: 'cancelled' })
        signalHistoryChanged()
        return
      }
      setAccessMenuOpen(false)
      setSession(created)
      dispatch({ event: 'turn.user', content: goal.trim() })
      setGoal('')
      openEventStream(created)
      signalHistoryChanged()
    } catch (error) {
      dispatch({ event: 'session.error', message: error.message })
    } finally {
      if (pendingStartRef.current === pending) pendingStartRef.current = null
      if (pending.abandoned) setStopPending(false)
    }
  }

  const sendFollowUp = async () => {
    const text = followUp.trim()
    if (!session || !text || running) return
    dispatch({ event: 'turn.starting', task: text })
    setStartedAt(Date.now())
    setClock(Date.now())
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
    if (stopPending) return
    // Stop offered during the starting phase used to be dead: the composer shows
    // it as soon as `session.starting` is dispatched, but the session id it
    // needs only exists once the create request comes back. Mark the pending
    // start abandoned and let `start` cancel the session the server hands over.
    if (!session) {
      if (!pendingStartRef.current || pendingStartRef.current.abandoned) return
      pendingStartRef.current.abandoned = true
      setStopPending(true)
      dispatch({ event: 'turn.stopping' })
      return
    }
    setStopPending(true)
    dispatch({ event: 'turn.stopping' })
    try {
      await cancelWorkspaceSession(apiBase, session.id)
      await waitForWorkspaceSessionTerminal(apiBase, session.id)
      // An open stream still owes this turn the server's own terminal event
      // (`aborted`), and it refreshes the history when it lands. Dispatching a
      // second ending here appended a duplicate "Stopped" divider, flipped the
      // turn outcome and double-fired the history refresh.
      if (!terminalHandledRef.current && !eventSourceRef.current) {
        terminalHandledRef.current = true
        dispatch({ event: 'session.finished', outcome: 'cancelled' })
        signalHistoryChanged()
      }
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
    activityRecoverySuppressedRef.current = true
    if (session) {
      try { await cancelWorkspaceSession(apiBase, session.id) } catch {}
    }
    if (eventSourceRef.current) {
      intentionalClosuresRef.current.add(eventSourceRef.current)
      eventSourceRef.current.close()
    }
    eventSourceRef.current = null
    sessionRef.current = null
    setSession(null)
    setSelectedThreadId('')
    setGoal('')
    setFollowUp('')
    setStartedAt(null)
    setChanges({ ...EMPTY_CHANGES })
    setApprovalMode('approval_gated')
    setAllowNetwork(false)
    setAccessMenuOpen(false)
    setFullAutoConfirmOpen(false)
    dispatch({ event: 'session.reset' })
    signalHistoryChanged()
  }

  // Whether the live session already runs under the access the user has
  // selected. Approval mode and the network grant are fixed when the server
  // creates a session, so a follow-up cannot carry a change — it would silently
  // run under the old posture, which is worse than refusing.
  const accessMatchesSession = !session
    || ((session.approval_mode || 'approval_gated') === approvalMode
      && Boolean(session.allow_network) === allowNetwork)

  const submitComposer = (event) => {
    event?.preventDefault()
    if (running) return
    // A changed posture starts a new session on the same thread, so the
    // transcript continues while the new access actually takes effect.
    if (session && accessMatchesSession) sendFollowUp()
    else start()
  }

  const composerKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent?.isComposing) {
      event.preventDefault()
      submitComposer()
    }
  }

  const status = stopPending ? 'Stopping' : PHASE_LABEL[state.phase] || state.phase
  const empty = !historicalTurns.length && !state.events.length && !activity?.task

  return (
    <div className={`code-workbench ${inspectorOpen ? '' : 'is-inspector-closed'}`}>
      <section className="code-stage" aria-labelledby="code-session-title">
        <header className="code-stage__header">
          <div className="code-stage__title">
            <span className={`code-stage__status is-${state.phase}`}><span />{status}</span>
            <h2 id="code-session-title">{title}</h2>
            {workspacePath ? <small title={workspacePath}>{compactPath(workspacePath)}</small> : null}
          </div>
          <div className="code-stage__actions">
            {elapsed ? <span className="code-elapsed">{running ? 'Working' : 'Worked'} for {elapsed}</span> : null}
            {!inspectorOpen ? <button type="button" aria-label="Open work details" onClick={() => setInspectorOpen(true)}><IconSidebar size={18} /></button> : null}
            <button type="button" aria-label="New coding session" onClick={reset}><IconEdit size={17} /><span>New session</span></button>
          </div>
        </header>

        <div className="code-thread" ref={threadRef}>
          {empty ? (
            <div className="code-landing">
              <span className="code-landing__mark"><IconBolt size={25} /></span>
              <h1>What should Camelid build?</h1>
              <p>Describe an outcome. Camelid will inspect the workspace, show each action as it happens, and follow the access policy you choose.</p>
              {!toolCapable ? (
                <section className="workspace-prerequisite" role="status">
                  <div className="workspace-prerequisite__head"><IconError size={18} /><div><h3>Load an agent-evaluated model</h3><p>Code mode fails closed unless the exact active model row has a passing tool-capability receipt.</p></div></div>
                  <div className="workspace-prerequisite__actions"><Button variant="outline" onClick={() => setTab('library')}>Open Models</Button></div>
                </section>
              ) : null}
              {savedThreads.length ? (
                <div className="code-recent">
                  <span>Continue in this workspace</span>
                  {savedThreads.slice(0, 3).map((thread) => (
                    <button type="button" key={thread.id} onClick={() => setSelectedThreadId(thread.id)}>
                      <IconHistory size={15} /><span><strong>{thread.title}</strong><small>{thread.turn_count} turns</small></span><IconChevronRight size={15} />
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="code-feed">
              <LiveActivitySummary activity={activity} running={running} />
              {historicalTurns.map((turn, index) => <HistoricalTurn turn={turn} key={`history-${index}-${turn.user}`} />)}
              {feedEvents.map((entry) => (
                <ActivityEvent
                  event={entry.event}
                  pairedResult={entry.pairedResult}
                  activeApproval={state.approval}
                  decisionBusy={decisionBusy}
                  onDecision={decide}
                  key={entry.key}
                />
              ))}
              {running && !state.events.some((event) => event.event === 'model.live') ? (
                <div className="code-agent-working"><span className="code-live-dot" /><span>Camelid is working…</span></div>
              ) : null}
            </div>
          )}
        </div>

        <footer className="code-composer-shell">
          {!session ? (
            <div className="code-workspace-picker">
              <IconSearch size={15} />
              <input
                value={workspacePath}
                onChange={(event) => { setWorkspacePath(event.target.value); setSelectedThreadId('') }}
                spellCheck="false"
                placeholder={navigator.platform?.startsWith('Win') ? 'Choose a workspace, for example C:\\projects\\app' : 'Choose a workspace, for example /workspace/app'}
                aria-label="Workspace folder"
              />
              <Button variant="ghost" size="sm" onClick={() => setBrowseOpen(true)}>Browse…</Button>
            </div>
          ) : null}
          <form className="code-composer" onSubmit={submitComposer}>
            <textarea
              value={composerValue}
              onChange={(event) => session ? setFollowUp(event.target.value) : setGoal(event.target.value)}
              onKeyDown={composerKeyDown}
              placeholder={session ? 'Ask for a follow-up or adjustment' : 'Describe the change you want Camelid to make'}
              rows={3}
              // Deliberately NOT gated on toolCapable/runtimeReady. Those depend
              // on which model happens to be loaded, and disabling the textarea
              // left a dead box you could not even draft a task in while going
              // to load a different model. The send button carries the gate.
              aria-label={session ? 'Follow-up instruction' : 'Coding task'}
            />
            <div className="code-composer__footer">
              <div className="code-composer__chips">
                <div className="code-access-control" ref={accessMenuRef}>
                  <button
                    type="button"
                    className={`code-access-chip ${approvalMode === 'full_auto' ? 'is-full-auto' : ''}`}
                    aria-haspopup="menu"
                    aria-expanded={accessMenuOpen}
                    onClick={() => setAccessMenuOpen((open) => !open)}
                    // Gated on a RUNNING turn, not on having a session at all.
                    // The server binds approval mode when a session is created,
                    // so a change cannot alter the turn in flight — but once one
                    // finishes, refusing to let the user change access left a
                    // permanently dead control whose only explanation was a
                    // tooltip. A change now applies to the next task, which
                    // `submitComposer` delivers by starting a fresh session on
                    // the same thread instead of sending a follow-up.
                    disabled={running}
                    title={running
                      ? 'Access cannot change while a turn is running'
                      : 'Choose approval and network access'}
                  >
                    <IconWarning size={14} />
                    {approvalMode === 'full_auto' ? 'Full auto' : 'Approval gated'}
                    {allowNetwork ? <IconNetwork size={13} /> : null}
                    <IconChevronDown size={12} />
                  </button>
                  {accessMenuOpen && !running ? (
                    <div className="code-access-menu" role="menu" aria-label="Agent access">
                      <div className="code-access-menu__heading">
                        <strong>Agent access</strong>
                        <small>
                          {session
                            ? 'Applies to your next task, which starts a fresh session on this thread'
                            : 'Applies to this coding session only'}
                        </small>
                      </div>
                      <button
                        type="button"
                        role="menuitemradio"
                        aria-checked={approvalMode === 'approval_gated'}
                        className={approvalMode === 'approval_gated' ? 'is-selected' : ''}
                        onClick={() => { setApprovalMode('approval_gated'); setAccessMenuOpen(false) }}
                      >
                        <span className="code-access-menu__icon"><IconWarning size={16} /></span>
                        <span><strong>Approval gated</strong><small>Ask before writes, commands, and network actions.</small></span>
                        <span className="code-access-menu__check">{approvalMode === 'approval_gated' ? '✓' : ''}</span>
                      </button>
                      <button
                        type="button"
                        role="menuitemradio"
                        aria-checked={approvalMode === 'full_auto'}
                        className={`is-danger ${approvalMode === 'full_auto' ? 'is-selected' : ''}`}
                        onClick={() => { setAccessMenuOpen(false); setFullAutoConfirmOpen(true) }}
                      >
                        <span className="code-access-menu__icon"><IconBolt size={16} /></span>
                        <span><strong>Today is a good day to die</strong><small>Full auto: run writes and shell commands without asking.</small></span>
                        <span className="code-access-menu__check">{approvalMode === 'full_auto' ? '✓' : ''}</span>
                      </button>
                      <div className="code-access-menu__divider" />
                      <button
                        type="button"
                        role="menuitemcheckbox"
                        aria-checked={allowNetwork}
                        className={allowNetwork ? 'is-selected' : ''}
                        onClick={() => setAllowNetwork((enabled) => !enabled)}
                      >
                        <span className="code-access-menu__icon"><IconNetwork size={16} /></span>
                        <span><strong>Network and web search</strong><small>Give the agent built-in web_search and http_fetch tools.</small></span>
                        <span className={`code-access-switch ${allowNetwork ? 'is-on' : ''}`}><span /></span>
                      </button>
                    </div>
                  ) : null}
                </div>
                <span title={modelName}><IconBolt size={14} /> {modelName || 'No model'}</span>
              </div>
              {running ? (
                <button type="button" className="code-composer__send is-stop" aria-label="Stop coding task" onClick={stop} disabled={stopPending}><IconStop size={17} /></button>
              ) : (
                <>
                  {/* An unconfirmed Stop means the run MAY still be alive, so the
                      Stop control has to survive alongside Send — dropping it
                      would leave the one state that needs a retry without one. */}
                  {stopUnconfirmed ? (
                    <button type="button" className="code-composer__send is-stop" aria-label="Retry stopping the coding task" onClick={stop} disabled={stopPending}><IconStop size={17} /></button>
                  ) : null}
                  <button type="submit" className="code-composer__send" aria-label={session ? 'Send follow-up' : 'Start coding'} disabled={!canSubmit}><IconSend size={18} /></button>
                </>
              )}
            </div>
          </form>
          <small className="code-composer-hint">
            Enter to send · Shift+Enter for a new line · {approvalMode === 'full_auto'
              ? `full auto can write and run commands without stopping${allowNetwork ? ' · web search on' : ''}`
              : `writes and commands require approval${allowNetwork ? ' · web search on' : ''}`}
          </small>
        </footer>
      </section>

      {inspectorOpen ? (
        <CodeInspector
          activity={activity}
          agents={agents}
          approvalMode={session?.approval_mode || approvalMode}
          allowNetwork={session?.allow_network ?? allowNetwork}
          changes={changes}
          latestTool={latestTool}
          latestResult={latestResult}
          modelName={modelName}
          running={running}
          session={session}
          workspacePath={workspacePath}
          undoBusy={undoBusy}
          onClose={() => setInspectorOpen(false)}
          onRefreshChanges={() => refreshChanges()}
          onUndo={undo}
        />
      ) : null}

      {browseOpen ? (
        <FolderPicker
          apiBase={apiBase}
          initialPath={workspacePath.trim() || null}
          onClose={() => setBrowseOpen(false)}
          onPick={(path) => {
            if (path) {
              setWorkspacePath(path)
              setSelectedThreadId('')
            }
            setBrowseOpen(false)
          }}
        />
      ) : null}

      <ConfirmDialog
        open={fullAutoConfirmOpen}
        title="Enable full auto?"
        detail="Today is a good day to die mode lets Camelid edit files and run shell commands without asking. File tools stay confined to the selected workspace. How far shell commands are confined depends on your OS: on Linux and macOS they run under a kernel sandbox with no network and writes limited to the workspace and the temp directory, while on Windows they are working-directory pinned and hard-timed but are not filesystem- or network-isolated. Stop remains available; the network switch separately controls Camelid's built-in web tools."
        confirmLabel="Enable full auto"
        cancelLabel="Keep approval gated"
        onCancel={() => setFullAutoConfirmOpen(false)}
        onConfirm={() => {
          setApprovalMode('full_auto')
          setFullAutoConfirmOpen(false)
        }}
      />
    </div>
  )
}
