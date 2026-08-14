import { memo, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { findCompatibilityHint } from '../lib/capabilities'
import {
  cancelWorkspaceSession,
  createWorkspaceSession,
  decideWorkspaceApproval,
  getWorkspaceChanges,
  getWorkspaceActivity,
  getWorkspaceThread,
  getWorkspaceThreads,
  contextLimitingFactorLabel,
  contextWindowModeLabel,
  formatContextTokens,
  normalizeContextWindow,
  parsePlanSteps,
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
  IconBolt, IconCheck, IconCheckCircle, IconChevronDown, IconChevronRight, IconClose, IconEdit,
  IconError, IconHistory, IconNetwork, IconRefresh, IconSearch, IconSend, IconSidebar,
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

/// How much of a tool argument — or a failure message — rides on the collapsed
/// line beside the tool name.
const TOOL_LINE_HINT_CHARS = 120

function formatToolName(tool) {
  return String(tool || 'tool')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

// One anchored parse per rendered tool line, producing BOTH the name and the
// hint. `detail` can be an entire source file (a write_file argument) and this
// feed re-renders while a step streams, so the head is SLICED before any regex
// or rewrite touches it — nothing here ever walks the whole string.
function describeToolCall(detail) {
  const head = String(detail || '').slice(0, TOOL_LINE_HINT_CHARS + 96).trim()
  const match = /^([a-zA-Z0-9_]+)\s*\(/.exec(head)
  if (!match) return { label: 'Agent action', argument: '' }
  const flat = head.slice(match[0].length).replace(/\s+/g, ' ').replace(/^["']/, '').trim()
  return {
    label: formatToolName(match[1]),
    argument: flat.length > TOOL_LINE_HINT_CHARS
      ? `${flat.slice(0, TOOL_LINE_HINT_CHARS)}…`
      : flat.replace(/[)"']+$/, ''),
  }
}

// A result body reduced to one line for the collapsed row. Bounded like the
// argument preview, and for the same reason.
function firstLine(text) {
  const value = String(text || '')
  const head = value.slice(0, TOOL_LINE_HINT_CHARS).replace(/\s+/g, ' ').trim()
  return value.length > TOOL_LINE_HINT_CHARS ? `${head}…` : head
}

function timingMetric(timing, snakeCase, camelCase) {
  const value = timing?.[snakeCase] ?? timing?.[camelCase]
  return Number.isFinite(value) ? value : null
}

function timingBoolean(timing, snakeCase, camelCase) {
  const value = timing?.[snakeCase] ?? timing?.[camelCase]
  return typeof value === 'boolean' ? value : null
}

function formatTimingBits(timing) {
  const totalMs = timingMetric(timing, 'total_ms', 'totalMs')
  const outputTokens = timingMetric(timing, 'output_tokens', 'outputTokens')
  const ttftMs = timingMetric(timing, 'ttft_ms', 'ttftMs')
  const firstContentMs = timingMetric(timing, 'server_first_content_ms', 'serverFirstContentMs')
    ?? timingMetric(timing, 'first_token_ms', 'firstTokenMs')
  const prefillMs = timingMetric(timing, 'prefill_ms', 'prefillMs')
  const decodeMs = timingMetric(timing, 'decode_ms', 'decodeMs')
  const cacheHit = timingBoolean(timing, 'prompt_cache_hit', 'promptCacheHit')
  const reusedTokens = timingMetric(timing, 'reused_tokens', 'reusedTokens')
  const prefilledTokens = timingMetric(timing, 'prefilled_tokens', 'prefilledTokens')
  return [
    Number.isFinite(totalMs) ? `${formatMs(totalMs)} total` : null,
    Number.isFinite(outputTokens) ? `${formatTokens(outputTokens)} output tokens` : null,
    Number.isFinite(ttftMs) ? `${formatMs(ttftMs)} TTFT` : null,
    Number.isFinite(firstContentMs) ? `${formatMs(firstContentMs)} server first content` : null,
    Number.isFinite(prefillMs) ? `${formatMs(prefillMs)} prefill` : null,
    Number.isFinite(decodeMs) ? `${formatMs(decodeMs)} decode` : null,
    cacheHit === true ? 'cache hit' : cacheHit === false ? 'prompt-cache miss' : null,
    Number.isFinite(reusedTokens) && reusedTokens > 0 ? `${formatTokens(reusedTokens)} reused` : null,
    Number.isFinite(prefilledTokens) && prefilledTokens > 0 ? `${formatTokens(prefilledTokens)} prefilled` : null,
  ].filter(Boolean)
}

function formatPromptStep(step) {
  const prefillMs = timingMetric(step, 'prefill_ms', 'prefillMs')
  const cacheHit = timingBoolean(step, 'prompt_cache_hit', 'promptCacheHit')
  const reusedTokens = timingMetric(step, 'reused_tokens', 'reusedTokens')
  const prefilledTokens = timingMetric(step, 'prefilled_tokens', 'prefilledTokens')
  const tokenCount = cacheHit === true ? reusedTokens : prefilledTokens
  const cache = cacheHit === true ? 'hit' : cacheHit === false ? 'miss' : null
  return [
    Number.isFinite(prefillMs) ? formatMs(prefillMs) : null,
    cache ? `${cache}${Number.isFinite(tokenCount) && tokenCount > 0 ? ` ${formatTokens(tokenCount)}` : ''}` : null,
  ].filter(Boolean).join(' · ') || '—'
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

// A `model.live` entry is only meaningful while its step is unresolved. The
// reducer pops the live tail only when it IS the tail, and `model.timing` lands
// between the last delta and the tool call it paid for — so without this the
// thinking card outlived its step for the rest of the turn. These are the
// events that PROVE a step resolved; anything else (an `agent.updated` from a
// sub-agent, a notice, an approval) can arrive mid-stream and must not hide
// text that is still the current output.
const LIVE_RESOLVERS = new Set(['model.live', 'model.timing', 'tool.call', 'model.answer', 'session.finished', 'session.error'])

// Pair each tool call with its result, hand each model step's cost to the work
// that step produced, and stamp a key that survives both. Grouping consumes two
// entries as one, so a raw array index shifts for every later item the moment a
// result lands — which remounts the rendered <details> and silently collapses
// whatever the user had expanded. The key is the reducer's per-arrival `uid`,
// not the envelope's `sequence`: the server counts sequences per event stream
// and every follow-up turn opens a new one, so sequences collide between the
// turns held in the same feed.
//
// `model.timing` is reported the instant a model step returns — BEFORE the tool
// call or the answer that step produced. It is the cost of the work that
// follows it, so it is held here and handed to that entry rather than taking a
// full-width row of its own between every pair of tool lines. A timing with
// nothing to ride on is flushed as its own quiet line; the event is never
// dropped, and it is never carried across a turn boundary.
function groupActivityEvents(events) {
  const entries = []
  const paired = new Set()
  let liveCutoff = -1
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (LIVE_RESOLVERS.has(events[index].event)) { liveCutoff = index; break }
  }
  let pendingTiming = null
  const flushTiming = () => {
    if (!pendingTiming) return
    entries.push({ key: pendingTiming.key, event: pendingTiming.event, pairedResult: null, timing: null })
    pendingTiming = null
  }
  let liveVisible = false
  for (let index = 0; index < events.length; index += 1) {
    if (paired.has(index)) continue
    const event = events[index]
    const key = event.uid != null ? `uid-${event.uid}` : `pos-${index}-${event.event}`
    if (event.event === 'model.timing') {
      flushTiming()
      pendingTiming = { key, event }
      continue
    }
    if (event.event === 'model.live' && index !== liveCutoff) continue
    const hostsTiming = event.event === 'tool.call' || event.event === 'model.answer'
    if (!hostsTiming) flushTiming()
    const resultIndex = event.event === 'tool.call' ? findPairedResult(events, index) : -1
    if (resultIndex !== -1) paired.add(resultIndex)
    const timing = hostsTiming ? pendingTiming?.event || null : null
    if (hostsTiming) pendingTiming = null
    if (event.event === 'model.live') liveVisible = true
    entries.push({ key, event, pairedResult: resultIndex === -1 ? null : events[resultIndex], timing })
  }
  flushTiming()
  return { entries, liveVisible }
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
          <div className="code-message__content"><AssistantMarkdown content={turn.assistant} /></div>
        </article>
      ) : null}
    </div>
  )
}

/// The plan the agent published, as ONE compact card: a title, the step it is
/// on, a count and a progress strip. The step list itself is a disclosure, so
/// the collapsed card is a row rather than the tallest block in the transcript.
/// Memoized on `content` because the parse used to re-run in the render body on
/// every streamed token.
const PlanUpdate = memo(function PlanUpdate({ content }) {
  const steps = useMemo(() => parsePlanSteps(content), [content])
  const active = steps.find((step) => step.status === 'active')
  const done = steps.filter((step) => step.status === 'done').length

  // An `update_plan` result the model wrote as free prose has no steps to show.
  if (!steps.length) {
    return (
      <article className="code-plan-update" aria-label="Agent plan">
        <span className="code-plan-update__head"><strong>Plan</strong></span>
        <pre>{content}</pre>
      </article>
    )
  }

  return (
    <details className="code-plan-update" aria-label="Agent plan">
      <summary>
        <span className="code-plan-update__head">
          <strong>Plan</strong>
          <span className="code-plan-update__now">{active ? `Working on: ${active.text}` : `${done} of ${steps.length} complete`}</span>
          <span className="code-plan-update__count">{done}/{steps.length}</span>
          <IconChevronRight size={12} className="code-plan-update__chevron" />
        </span>
        <span className="code-plan-update__pips" aria-hidden="true">
          {steps.map((step, index) => <i className={`is-${step.status}`} key={`pip-${index}-${step.text}`} />)}
        </span>
      </summary>
      <ol>
        {steps.map((step, index) => (
          <li className={`is-${step.status}`} key={`${index}-${step.text}`}>
            <span>{step.status === 'done' ? <IconCheck size={11} /> : step.status === 'active' ? <span className="code-live-dot" /> : null}</span>
            <span>{step.text}</span>
          </li>
        ))}
      </ol>
    </details>
  )
})

/// One quiet line per tool call: a glyph, the tool name, what it acted on, and a
/// chevron. No border, no fill, no status pill, no subtitle — five calls in a
/// row must read as five words, not five boxes.
///
/// Kept as <details> so the open state belongs to the DOM node React reconciles
/// by `uid`. The body is built only once the row has been opened: it is the
/// whole tool argument plus up to 16 KB of result, and concatenating it on
/// every render meant rebuilding — and having React compare — megabytes per
/// generated token for a payload a closed row never displays.
///
/// The model-step timing lives INSIDE the expansion. It is the cost of the step
/// that produced this call, not of the tool, so painting it beside the tool
/// name would attribute the seconds to the wrong subject.
function ToolLine({ label, hint, outcome, timing, detail, result }) {
  const [opened, setOpened] = useState(false)
  const settled = Boolean(outcome)
  const failed = outcome === 'error'
  const state = failed ? 'is-error' : settled ? 'is-complete' : 'is-running'
  const timingText = timing ? formatTimingBits(timing).join(' · ') : ''
  return (
    <details
      className={`code-tool-card ${state}`}
      onToggle={(event) => { if (event.currentTarget.open) setOpened(true) }}
    >
      <summary>
        <span className="code-tool-card__glyph">
          {failed ? <IconError size={13} /> : settled ? <IconCheck size={13} /> : <span className="code-live-dot" />}
        </span>
        <strong>{label}</strong>
        {/* Never painted unless it failed — see workspace.css. It stays in the
            DOM for assistive technology and for the workbench smoke. */}
        <span className={`code-tool-card__state ${state}`}>{failed ? 'failed' : settled ? 'Done' : 'Running'}</span>
        {hint ? <span className="code-tool-card__hint">{hint}</span> : null}
        <IconChevronRight size={12} className="code-tool-card__chevron" />
      </summary>
      {opened ? (
        <>
          {timingText ? <p className="code-tool-card__meta">Model step · {timingText}</p> : null}
          <pre>{result ? `${detail}\n\n${result}` : detail}</pre>
        </>
      ) : null}
    </details>
  )
}

// The feed re-renders on every accepted event. Every prop below is a stable
// reference across a stream — the reducer shallow-copies the event array and
// replaces only the `model.live` tail — so the default shallow compare bails
// out of every settled row. This is not a micro-optimisation: the agent's event
// channel blocks on this consumer, so work done here is backpressure on decode.
const ActivityEvent = memo(function ActivityEvent({ event, pairedResult, timing, activeApproval, decisionBusy, onDecision }) {
  if (event.event === 'turn.user') {
    return <article className="code-message code-message--user">{event.content}</article>
  }

  if (event.event === 'model.live') {
    // Only the tail is rendered. This node re-renders as text arrives, and a
    // long step (a whole file inside a write_file argument) otherwise grows an
    // ever-taller <pre> that costs more to lay out with each flush.
    const live = visibleLiveText(event.content)
    const tail = live.length > LIVE_TAIL_CHARS ? `…${live.slice(-LIVE_TAIL_CHARS)}` : live
    return (
      <article className="code-thinking-card" aria-live="polite">
        <header><span className="code-live-dot" /><span>Working</span></header>
        {tail ? <pre>{tail}</pre> : <p className="code-thinking-card__quiet">Preparing the next tool call…</p>}
      </article>
    )
  }

  if (event.event === 'model.answer') {
    return (
      <article className="code-message code-message--assistant">
        <div className="code-message__content">
          <AssistantMarkdown content={event.content} />
          {timing ? <p className="code-message__timing">{formatTimingBits(timing).join(' · ')}</p> : null}
        </div>
      </article>
    )
  }

  if (event.event === 'tool.call') {
    const result = pairedResult || null
    if (result && String(event.detail || '').startsWith('update_plan(') && result.outcome !== 'error') {
      return <PlanUpdate content={result.content} />
    }
    const { label, argument } = describeToolCall(event.detail)
    const failed = result?.outcome === 'error'
    return (
      <ToolLine
        label={label}
        // A failure states what went wrong on the line itself; a success states
        // what it acted on. Neither needs a subtitle and neither needs a card.
        hint={failed ? firstLine(result.content) : argument}
        outcome={result ? result.outcome : ''}
        timing={timing}
        detail={event.detail}
        result={result ? result.content : ''}
      />
    )
  }

  if (event.event === 'tool.result') {
    // Reached only when pairing failed: the ring buffer evicted the call, or a
    // barrier landed between the two. Same line, sourced from the result alone.
    return (
      <ToolLine
        label={formatToolName(event.tool)}
        hint={firstLine(event.content)}
        outcome={event.outcome || 'ok'}
        timing={null}
        detail={event.content}
        result=""
      />
    )
  }

  if (event.event === 'approval.required') {
    const pending = activeApproval?.approval_id === event.approval_id
    if (!pending) {
      // A decided approval is history, but it is also the audit record of what
      // was allowed — so it collapses to a line that still carries the payload
      // rather than a dimmed card that holds 260px of it on screen forever.
      return (
        <details className="code-inline-approval is-resolved">
          <summary>
            <IconWarning size={13} />
            <strong>Reviewed {formatToolName(event.tool)}</strong>
            <span>Decision sent</span>
            <IconChevronRight size={12} className="code-inline-approval__chevron" />
          </summary>
          <pre>{event.detail}</pre>
        </details>
      )
    }
    // The one element that keeps its box. It blocks the run and carries four
    // consequential actions; containment is the point.
    return (
      <article className="code-inline-approval is-pending" role="group" aria-label={`Review ${formatToolName(event.tool)}`}>
        <header>
          <IconWarning size={15} />
          <strong>Review {formatToolName(event.tool)}</strong>
          <small>{event.risk} action</small>
        </header>
        <pre>{event.detail}</pre>
        <div className="code-inline-approval__actions">
          <Button variant="ghost" size="sm" onClick={() => onDecision('abort')} disabled={decisionBusy}>Stop</Button>
          <Button variant="ghost" size="sm" onClick={() => onDecision('deny')} disabled={decisionBusy}>Deny</Button>
          <Button variant="outline" size="sm" onClick={() => onDecision('always_tool')} disabled={decisionBusy}>Always allow</Button>
          <Button variant="primary" size="sm" onClick={() => onDecision('allow_once')} loading={decisionBusy}>Allow once</Button>
        </div>
      </article>
    )
  }

  if (event.event === 'model.timing') {
    // Reached only when a step's metrics had nothing after them to ride on.
    const bits = formatTimingBits(event)
    if (!bits.length) return null
    return <div className="code-meta-event"><span>{bits.join(' · ')}</span></div>
  }

  if (event.event === 'memory.compacted') {
    return <div className="code-meta-event"><IconHistory size={12} /><span>Context compacted · {event.archived_turns || 0} turns archived</span></div>
  }

  if (event.event === 'session.notice') {
    return <div className="code-session-notice"><IconCheckCircle size={14} /><span>{event.content}</span></div>
  }

  if (event.event === 'session.error') {
    return (
      <div className="code-session-notice is-error">
        <IconError size={14} /><strong>Session error</strong><span>{event.message}</span>
      </div>
    )
  }

  if (event.event === 'session.finished') {
    return (
      <div className="code-worked-divider">
        <span>{PHASE_LABEL[event.outcome] || event.outcome}</span>
      </div>
    )
  }

  return null
})

// The three statuses that can still be producing work. Everything else the
// backend reports — completed, failed, inconclusive, cancelled — plus the
// client-side `stopped`, is terminal.
const RUNNING_AGENT_STATUS = new Set(['starting', 'running', 'waiting'])

const AGENT_STATUS_LABEL = {
  starting: 'Starting',
  running: 'Running',
  waiting: 'Waiting',
  completed: 'Completed',
  failed: 'Failed',
  inconclusive: 'Inconclusive',
  cancelled: 'Cancelled',
  stopped: 'Stopped',
}

/// Compact counts for meta lines. Table cells use toLocaleString: a column of
/// numbers is worth reading exactly.
function formatTokens(value) {
  if (!Number.isFinite(value)) return '—'
  if (value < 1000) return String(value)
  if (value < 1000000) return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)}k`
  return `${(value / 1000000).toFixed(1)}M`
}

function formatMs(value) {
  if (!Number.isFinite(value) || value <= 0) return '—'
  if (value < 1000) return `${Math.round(value)}ms`
  if (value < 60000) return `${(value / 1000).toFixed(1)}s`
  return formatElapsed(value)
}

/// Ticks on its own so the panel around it is not re-rendered once a second.
/// Passing a clock down as a prop defeats every memo below it.
const Elapsed = memo(function Elapsed({ anchor, title }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!Number.isFinite(anchor)) return undefined
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [anchor])
  if (!Number.isFinite(anchor)) return null
  return <span className="ci-num" title={title}>{formatElapsed(Math.max(0, now - anchor))}</span>
})

/// A disclosure whose body is not built until it has been opened. <details>
/// hides its children with CSS, so React still creates and diffs every node in
/// a closed one — and this panel is live while a turn streams.
function Fold({ className = '', label, count, children }) {
  const [open, setOpen] = useState(false)
  return (
    <details className={`ci-fold ${className}`} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>
        <span>{label}</span>
        <span className="ci-num">{count}</span>
        <span className="ci-chev" aria-hidden="true"><IconChevronRight size={12} /></span>
      </summary>
      {open ? children : null}
    </details>
  )
}

function AgentCard({ agent, isMain, anchor, anchorTitle, action, children }) {
  const status = agent.status || 'running'
  return (
    <li className={`ci-task is-${status}`}>
      <div className="ci-task__head">
        <span className="ci-task__glyph" aria-hidden="true"><span /></span>
        <strong className="ci-task__title">{isMain ? 'Primary agent' : agent.label}</strong>
        {action ? (
          <button type="button" className="ci-icon-btn" aria-label={action.label} title={action.label} onClick={action.onClick}>
            {action.icon}
          </button>
        ) : <span aria-hidden="true" />}
      </div>
      <p className="ci-task__meta">
        <span>{isMain ? 'Agent' : 'Subagent'} · {AGENT_STATUS_LABEL[status] || formatToolName(status)}</span>
        <Elapsed anchor={anchor} title={anchorTitle} />
      </p>
      {children}
      {agent.task ? <p className="ci-task__desc" title={agent.task}>{agent.task}</p> : null}
    </li>
  )
}

/// One row per completed model step. The columns are per STEP, not per agent: a
/// sub-agent runs in its own process behind a reporter that discards timing, so
/// it emits nothing the parent stream can attribute. See BACKEND_ASKS.
function StepTable({ steps, running }) {
  return (
    <div className="ci-table-wrap">
      <table className="ci-table ci-table--model">
        <thead>
          <tr>
            <th scope="col"><span className="sr-only">State</span></th>
            <th scope="col">#</th>
            <th scope="col">Out</th>
            <th scope="col">TTFT</th>
            <th scope="col">Prompt</th>
            <th scope="col">Total</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((step) => {
            const details = formatTimingBits(step).join(' · ')
            return (
              <tr key={step.index} title={details || undefined}>
                <td className="ci-table__mark"><IconCheck size={11} /></td>
                <td>{step.index}</td>
                <td>{Number.isFinite(step.outputTokens) ? step.outputTokens.toLocaleString() : '—'}</td>
                <td>{formatMs(step.ttftMs)}</td>
                <td className="ci-table__prompt">{formatPromptStep(step)}</td>
                <td>{formatMs(step.totalMs)}</td>
              </tr>
            )
          })}
          {running ? (
            <tr className="is-live">
              <td className="ci-table__mark"><span className="ci-spin" aria-hidden="true" /></td>
              <td>{steps.length + 1}</td>
              <td>—</td><td>—</td><td>—</td><td>—</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  )
}

const CodeInspector = memo(function CodeInspector({
  activity,
  agents,
  agentSeen,
  approvalMode,
  allowNetwork,
  changes,
  context,
  contextWindow,
  modelName,
  modelSteps,
  planSteps,
  running,
  session,
  tool,
  totals,
  undoBusy,
  workspacePath,
  onClose,
  onRefreshChanges,
  onStop,
  onUndo,
}) {
  // Local-only. No endpoint forgets an agent, so this hides rows in this panel
  // and nothing else — the button's title says so, and the primary agent is
  // never eligible, so one click can never blank the panel.
  const [hiddenIds, setHiddenIds] = useState(() => new Set())
  useEffect(() => { if (running) setHiddenIds(new Set()) }, [running])

  const main = agents.find((agent) => agent.id === 'main') || null
  const children = agents.filter((agent) => agent.id !== 'main')
  const runningChildren = children.filter((agent) => RUNNING_AGENT_STATUS.has(agent.status || 'running'))
  const finishedChildren = children.filter((agent) => !RUNNING_AGENT_STATUS.has(agent.status || 'running'))
  const visibleFinished = finishedChildren.filter((agent) => !hiddenIds.has(agent.id))
  const steps = modelSteps || []
  const runTotals = totals || { steps: 0, outputTokens: 0, elapsedMs: 0, tools: 0, toolFailures: 0 }

  const liveStage = activity.stage ? formatToolName(activity.stage) : (running ? 'Working' : 'Idle')
  const liveDetail = activity.detail
    || tool.hint
    || (tool.lastResult ? `${tool.lastResult} ${tool.lastResultFailed ? 'failed' : 'completed'}` : '')

  // main uses the SERVER's session clock. A child has no server-reported clock
  // at all, so its number is this panel's own observation and says so.
  const anchorFor = (agent) => agent.id === 'main'
    ? (activity.startedAt || agentSeen?.[agent.id]?.firstSeenAt || null)
    : (agentSeen?.[agent.id]?.firstSeenAt || null)
  const anchorTitleFor = (agent) => agent.id === 'main'
    ? 'Elapsed since the server started this session'
    : 'Observed by this panel since the agent first appeared. The server does not report per-agent runtime.'

  const usedPercent = context && context.budgetTotal > 0
    ? Math.min(100, Math.max(0, (context.promptTokens / context.budgetTotal) * 100))
    : 0
  const meterState = usedPercent >= 95 ? 'is-critical' : usedPercent >= 80 ? 'is-warn' : ''

  return (
    <aside className="code-inspector" aria-label="Coding session details">
      <header className="code-inspector__header">
        <div>
          <strong>Work details</strong>
          {workspacePath ? <small title={workspacePath}>{compactPath(workspacePath)}</small> : null}
        </div>
        <button type="button" aria-label="Close work details" onClick={onClose}><IconClose size={18} /></button>
      </header>

      <section className={`ci-group ${planSteps.length ? 'ci-group--joined' : ''}`}>
        <div className="ci-group__head">
          <h3 className="ci-group__label">Running <span className="ci-num">{(main ? 1 : 0) + runningChildren.length}</span></h3>
        </div>
        {main || runningChildren.length ? (
          <ul className="code-agent-list">
            {main ? (
              <AgentCard
                key={main.id}
                agent={main}
                isMain
                anchor={anchorFor(main)}
                anchorTitle={anchorTitleFor(main)}
                action={running ? { label: 'Stop this coding turn', icon: <IconStop size={12} />, onClick: onStop } : null}
              >
                <p className="ci-task__meta">
                  <span>{runTotals.steps} model {runTotals.steps === 1 ? 'step' : 'steps'} this session</span>
                  <span className="ci-num">{formatTokens(runTotals.outputTokens)} tokens</span>
                </p>
                {liveDetail ? (
                  <div className="code-process-row">
                    <span className="ci-task__stage">{liveStage}</span>
                    <small>{liveDetail}</small>
                  </div>
                ) : null}
                {steps.length || running ? (
                  <Fold className="ci-fold--nested" label="Model steps" count={runTotals.steps}>
                    <StepTable steps={steps} running={running} />
                  </Fold>
                ) : null}
              </AgentCard>
            ) : null}
            {runningChildren.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                isMain={false}
                anchor={anchorFor(agent)}
                anchorTitle={anchorTitleFor(agent)}
                action={null}
              >
                {agent.detail ? <p className="ci-task__note" title={agent.detail}>{agent.detail}</p> : null}
              </AgentCard>
            ))}
          </ul>
        ) : <p className="ci-empty">Agents appear here when a coding task starts.</p>}
      </section>

      {planSteps.length ? (
        <section className="ci-group">
          <div className="ci-group__head">
            <h3 className="ci-group__label">
              Plan <span className="ci-num">{planSteps.filter((step) => step.status === 'done').length}/{planSteps.length}</span>
            </h3>
          </div>
          <div className="ci-pips" role="img" aria-label={`${planSteps.filter((step) => step.status === 'done').length} of ${planSteps.length} plan steps complete`}>
            {planSteps.map((step, index) => <i className={`is-${step.status}`} key={`${index}-${step.text}`} />)}
          </div>
          <ol className="ci-plan">
            {planSteps.map((step, index) => (
              <li className={`is-${step.status}`} key={`${index}-${step.text}`}>
                <span className="ci-plan__mark" aria-hidden="true">
                  {step.status === 'done' ? <IconCheck size={11} />
                    : step.status === 'active' ? <span className="ci-spin" /> : null}
                </span>
                <span>{step.text}</span>
              </li>
            ))}
          </ol>
        </section>
      ) : null}

      <section className="ci-group">
        <div className="ci-group__head">
          <h3 className="ci-group__label">Changes <span className="ci-num">{changes.files?.length || 0}</span></h3>
          <button type="button" className="ci-icon-btn" aria-label="Refresh changes" onClick={onRefreshChanges} disabled={!session}>
            <IconRefresh size={14} />
          </button>
        </div>
        {changes.files?.length ? (
          <>
            <ul className="code-file-list">
              {changes.files.map((file) => <li key={file}><IconEdit size={13} /><span>{file}</span></li>)}
            </ul>
            {changes.summary ? <p className="ci-task__note">{changes.summary}</p> : null}
            <Fold className="code-patch" label="View patch" count="">
              <pre>{changes.diff}</pre>
            </Fold>
            <button type="button" className="ci-textaction" onClick={onUndo} disabled={running || undoBusy}>
              {undoBusy ? 'Undoing…' : 'Undo last change'}
            </button>
          </>
        ) : <p className="ci-empty">No file changes yet.</p>}
      </section>

      {context && context.budgetTotal > 0 ? (
        <section className="ci-group">
          <div className="ci-group__head">
            <h3 className="ci-group__label">Context window <span className="ci-num">{Math.round(usedPercent)}%</span></h3>
          </div>
          <div className={`ci-meter ${meterState}`}><i style={{ inlineSize: `${usedPercent}%` }} /></div>
          <p className="ci-task__meta">
            <span>prompt / budget</span>
            <span className="ci-num">{context.promptTokens.toLocaleString()} / {context.budgetTotal.toLocaleString()}</span>
          </p>
          <Fold label="Composition" count={context.parts.filter((part) => part.tokens > 0).length}>
            <dl className="ci-kv">
              {context.parts.filter((part) => part.tokens > 0).map((part) => (
                <div key={part.key}><dt>{part.label}</dt><dd className="ci-num">{part.tokens.toLocaleString()}</dd></div>
              ))}
            </dl>
          </Fold>
        </section>
      ) : null}

      <section className="ci-group">
        <Fold label="Session" count={approvalMode === 'full_auto' ? 'Full auto' : 'Gated'}>
          <dl className="ci-kv ci-kv--wide">
            <div><dt>Model</dt><dd>{modelName || 'No model loaded'}</dd></div>
            {contextWindow ? (
              <div><dt>Context</dt><dd className="ci-num">{contextWindowModeLabel(contextWindow)} · {formatContextTokens(contextWindow.effectiveTokens)}</dd></div>
            ) : null}
            {contextWindow?.pagedWorkingSetTokens ? (
              <div><dt>Active working set</dt><dd className="ci-num">{formatContextTokens(contextWindow.pagedWorkingSetTokens)} paged</dd></div>
            ) : null}
            {Number.isFinite(contextWindow?.memorySafeMaxTokens)
              && contextWindow.memorySafeMaxTokens < (contextWindow.pagedWorkingSetTokens || contextWindow.effectiveTokens) ? (
                <div><dt>Memory estimate</dt><dd className="ci-num">{formatContextTokens(contextWindow.memorySafeMaxTokens)} / KV owner</dd></div>
              ) : null}
            {contextWindow?.modelMaxTokens ? (
              <div><dt>Model max</dt><dd className="ci-num">{formatContextTokens(contextWindow.modelMaxTokens)}</dd></div>
            ) : null}
            {contextWindow?.validatedMaxTokens ? (
              <div><dt>Agent ceiling</dt><dd className="ci-num">{formatContextTokens(contextWindow.validatedMaxTokens)}</dd></div>
            ) : null}
            {contextWindow?.limitingFactor ? (
              <div><dt>Limited by</dt><dd>{contextLimitingFactorLabel(contextWindow.limitingFactor)}</dd></div>
            ) : null}
            <div><dt>Access</dt><dd>{approvalMode === 'full_auto' ? 'Full auto' : 'Approval gated'}</dd></div>
            <div><dt>Web tools</dt><dd>{allowNetwork ? 'On · search and fetch' : 'Off'}</dd></div>
            <div><dt>Tools run</dt><dd>{runTotals.tools} · {runTotals.toolFailures} failed</dd></div>
          </dl>
        </Fold>
      </section>

      {finishedChildren.length ? (
        <details className="ci-group ci-group--fold ci-fold">
          <summary>
            <span className="ci-group__label">Finished</span>
            <span className="ci-num">{visibleFinished.length}</span>
            <span className="ci-chev" aria-hidden="true"><IconChevronRight size={12} /></span>
          </summary>
          <div className="ci-group__foldbody">
            {visibleFinished.length ? (
              <ul className="code-agent-list">
                {visibleFinished.map((agent) => (
                  <AgentCard
                    key={agent.id}
                    agent={agent}
                    isMain={false}
                    anchor={anchorFor(agent)}
                    anchorTitle={anchorTitleFor(agent)}
                    action={null}
                  >
                    {agent.detail ? <p className="ci-task__note" title={agent.detail}>{agent.detail}</p> : null}
                  </AgentCard>
                ))}
              </ul>
            ) : <p className="ci-empty">All finished agents are hidden.</p>}
            {visibleFinished.length ? (
              <button
                type="button"
                className="ci-textaction"
                title="Hides these rows in this panel. The session keeps them; nothing is deleted."
                onClick={() => setHiddenIds(new Set(finishedChildren.map((agent) => agent.id)))}
              >
                Clear
              </button>
            ) : (
              <button type="button" className="ci-textaction" onClick={() => setHiddenIds(new Set())}>
                Show hidden ({finishedChildren.length})
              </button>
            )}
          </div>
        </details>
      ) : null}
    </aside>
  )
})

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
  // Highest session-scoped sequence this page has applied. Sequences are
  // monotonic, so a replay after a reconnect deduplicates on them alone.
  const lastSequenceRef = useRef(0)
  // A delta is one generated token. Dispatching each one is one full feed
  // render per token, and the agent's event channel blocks on this consumer, so
  // the render loop is backpressure on the model's decode loop. Deltas coalesce
  // onto a frame; the reducer already merges them into a single tail entry, so
  // a frame's worth is one dispatch, not N.
  const deltaBufferRef = useRef('')
  const deltaFrameRef = useRef(null)
  const decideRef = useRef(null)
  const inspectorActionsRef = useRef(null)

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
  const contextWindow = useMemo(
    () => normalizeContextWindow(session?.context_window, state.context?.budgetTotal),
    [session?.context_window, state.context?.budgetTotal],
  )
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
  const feed = useMemo(() => groupActivityEvents(state.events), [state.events])
  // Handlers handed to memoized children must have a permanently stable
  // identity or the children never bail out. `decide` closes over `session` and
  // `state.approval`, both of which move on approval events — a dependency list
  // would silently re-break the bail-out the first time either did. A ref
  // cannot, and it is read only from a user event, never during render.
  const onDecision = useMemo(() => (decision) => decideRef.current?.(decision), [])
  const inspectorActions = useMemo(() => ({
    onClose: () => setInspectorOpen(false),
    onRefreshChanges: () => inspectorActionsRef.current?.refresh(),
    onUndo: () => inspectorActionsRef.current?.undo(),
    onStop: () => inspectorActionsRef.current?.stop(),
  }), [])
  // `state.liveActivity` is rebuilt on every accepted event (it stamps its own
  // `updated_at_ms`), so handing the object to a memoized panel is the same as
  // not memoizing it. Project only the fields the panel displays.
  const inspectorActivity = useMemo(() => ({
    stage: activity?.stage || '',
    detail: activity?.detail || '',
    startedAt: Number(activity?.started_at_ms) || startedAt || null,
  }), [activity?.detail, activity?.stage, activity?.started_at_ms, startedAt])
  const inspectorTool = useMemo(() => {
    // The parsed hint, never the raw `detail`: a bare-JSON tool call would put
    // its argument syntax into the panel, which is not prose and is surfaced by
    // its own tool line.
    const described = latestTool ? describeToolCall(latestTool.detail) : null
    return {
      label: described?.label || '',
      hint: described?.argument || '',
      lastResult: latestResult ? formatToolName(latestResult.tool) : '',
      lastResultFailed: latestResult?.outcome === 'error',
    }
  }, [latestResult, latestTool])
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
          // Re-attach, do not just watch. The server keeps an unwatched run
          // alive for a bounded window and ends it if no stream comes back, so
          // adopting the session without reopening /events would still lose the
          // turn — just ninety seconds later.
          //
          // Order matters three ways. `thread.restored` REPLACES the event list,
          // so it has to land before any live event. Setting `session` above
          // permanently disables the restore effect at :1139-1146, so this is
          // now the only place those turns come back. And the live turn's own
          // prompt is not in the store yet, so it is re-seeded from the
          // snapshot's task or the transcript shows an answer with no question.
          const restored = await getWorkspaceThread(apiBase, next.workspace || '', next.id).catch(() => null)
          if (restored) {
            dispatch({ event: 'thread.restored', turns: restored.turns, turnCount: restored.thread.turn_count })
          }
          if (next.task) dispatch({ event: 'turn.user', content: String(next.task) })
          if (!eventSourceRef.current) {
            // A re-attach after a reload has no cursor of its own, so it asks
            // for the whole current turn: `resume` with the cursor still at 0.
            lastSequenceRef.current = 0
            openEventStream(next, true)
          }
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
  // is live so it can ask before ending one. Note the asymmetry with a browser
  // refresh, which no longer ends anything: this cancel is a DELIBERATE stop
  // behind App's confirm dialog, not a consequence of losing a socket.
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
    if (deltaFrameRef.current !== null) window.cancelAnimationFrame(deltaFrameRef.current)
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

  const flushDeltas = () => {
    deltaFrameRef.current = null
    const content = deltaBufferRef.current
    if (!content) return
    deltaBufferRef.current = ''
    dispatch({ event: 'model.delta', content })
  }

  // `resume` is true ONLY when re-attaching to a turn this page was already
  // following. Everything else — a new session, a follow-up turn — starts the
  // dedupe cursor over.
  //
  // The distinction is load-bearing and not merely tidy. Dedupe drops anything
  // at or below the cursor, so carrying a cursor into a stream that numbers
  // independently of the last one swallows the whole turn. The server now
  // allocates session-scoped monotonic sequences and would not do that, but a
  // client that only works while the server keeps that discipline is a client
  // that breaks silently the day it changes, and "silently" here means an empty
  // transcript for a run that is really executing.
  const openEventStream = (created, resume = false) => {
    // A stream this page has stopped reading is exactly what the server's grace
    // window counts as a live viewer, so there is never more than one.
    const previous = eventSourceRef.current
    if (previous) {
      intentionalClosuresRef.current.add(previous)
      previous.close()
    }
    terminalHandledRef.current = false
    if (!resume) lastSequenceRef.current = 0
    // 0 asks for the whole current turn, which is what a page that just
    // reloaded wants. The browser's own reconnect adds Last-Event-ID, which the
    // server prefers, so an automatic retry resumes exactly where it stopped.
    const after = lastSequenceRef.current
    const source = new EventSource(
      workspaceEndpoint(apiBase, `/${encodeURIComponent(created.id)}/events?after=${after}`),
    )
    eventSourceRef.current = source
    const closeStream = () => {
      intentionalClosuresRef.current.add(source)
      source.close()
      if (eventSourceRef.current === source) eventSourceRef.current = null
    }
    // The server's definitive end-of-response marker. EventSource reconnects
    // after ANY close and cannot see the status line, so without acting on this
    // a reader that attached to an already-settled turn reconnects forever.
    source.addEventListener('workspace.closed', () => {
      if (eventSourceRef.current !== source) return
      closeStream()
      if (!terminalHandledRef.current) followDetachedSession(created)
    })
    source.addEventListener('workspace', (message) => {
      if (eventSourceRef.current !== source) return
      try {
        const envelope = JSON.parse(message.data)
        const sequence = Number(envelope.sequence) || 0
        if (sequence && sequence <= lastSequenceRef.current) return
        if (sequence) lastSequenceRef.current = sequence
        if (envelope.replay_gap) {
          dispatch({
            event: 'session.notice',
            content: 'Reconnected to a turn that had already run past what Camelid keeps in memory, so its earliest steps are not shown here. The files it changed are in Changes, and the full record is saved with the session.',
          })
        }
        if (envelope.event === 'model.delta') {
          deltaBufferRef.current += String(envelope.content || '')
          if (deltaFrameRef.current === null) deltaFrameRef.current = window.requestAnimationFrame(flushDeltas)
          return
        }
        // Anything else is dispatched immediately — a tool result, an approval
        // or a terminal event must not wait on a frame — but the buffered text
        // has to land first or it would render after the work it preceded.
        if (deltaFrameRef.current !== null) {
          window.cancelAnimationFrame(deltaFrameRef.current)
          flushDeltas()
        }
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
        }
      } catch {
        dispatch({ event: 'session.error', message: 'Camelid returned an unreadable Code-mode event.' })
        closeStream()
      }
    })
    source.onerror = () => {
      if (intentionalClosuresRef.current.has(source) || eventSourceRef.current !== source) return
      // A dropped stream no longer ends the run: the turn is decoupled from the
      // socket and keeps going with nobody attached. So let EventSource do what
      // it does — reconnect, carrying Last-Event-ID, and pick the transcript up
      // where this reader stopped. Only a stream that cannot come back at all
      // falls through to the status poller. Do NOT bound the retries: giving up
      // on a live run means no approval card, and an approval nobody can answer
      // self-aborts the turn five minutes later.
      if (source.readyState === EventSource.CONNECTING) return
      closeStream()
      followDetachedSession(created)
    }
  }

  // The outcome the SERVER recorded for the turn, which is the only account of a
  // run whose event stream we lost. It is written before the session leaves its
  // running state, and a Code session id is also its thread id.
  // The outcome the SERVER recorded, read without a workspace-path round trip:
  // the activity snapshot already carries `terminal_outcome` and is served by
  // /activity with no arguments. Returning null rather than 'aborted' matters
  // now — after this change a stream we lost usually belongs to a turn that
  // ANSWERED, and one failed fetch must not stamp it "Stopped".
  const readRecordedOutcome = async () => {
    try {
      const snapshot = await getWorkspaceActivity(apiBase)
      const outcome = snapshot?.terminal_outcome
      return outcome ? String(outcome) : null
    } catch {
      return null
    }
  }

  const followDetachedSession = async (created) => {
    dispatch({
      event: 'session.notice',
      content: 'Lost the live activity stream. The turn keeps running on the server — following its recorded status until it ends.',
    })
    try {
      await waitForWorkspaceSessionTerminal(apiBase, created.id, {
        timeoutMs: DETACHED_FOLLOW_TIMEOUT_MS,
        pollMs: DETACHED_FOLLOW_POLL_MS,
      })
      terminalHandledRef.current = true
      dispatch({ event: 'session.finished', outcome: (await readRecordedOutcome()) || 'answered' })
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

  // Latched after every commit, never during render: the handlers close over
  // state that moves each turn, and the memoized props above must not.
  useEffect(() => {
    decideRef.current = decide
    inspectorActionsRef.current = { refresh: () => refreshChanges(), undo, stop }
  })

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
              {historicalTurns.map((turn, index) => <HistoricalTurn turn={turn} key={`history-${index}-${turn.user}`} />)}
              {feed.entries.map((entry) => (
                <ActivityEvent
                  event={entry.event}
                  pairedResult={entry.pairedResult}
                  timing={entry.timing}
                  activeApproval={state.approval}
                  decisionBusy={decisionBusy}
                  onDecision={onDecision}
                  key={entry.key}
                />
              ))}
              {running && !feed.liveVisible ? (
                <div className="code-agent-working"><span className="code-live-dot" /><span>Working…</span></div>
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
                {contextWindow ? (
                  <span
                    className="code-context-chip"
                    title={`${contextWindowModeLabel(contextWindow)} context: ${contextWindow.effectiveTokens.toLocaleString()} tokens${contextWindow.pagedWorkingSetTokens ? ` · active paged working set ${contextWindow.pagedWorkingSetTokens.toLocaleString()}` : ''}${contextWindow.validatedMaxTokens ? ` · agent ceiling ${contextWindow.validatedMaxTokens.toLocaleString()}` : ''}${contextWindow.modelMaxTokens ? ` · model max ${contextWindow.modelMaxTokens.toLocaleString()}` : ''}${contextWindow.limitingFactor ? ` · limited by ${contextLimitingFactorLabel(contextWindow.limitingFactor)}` : ''}`}
                  >
                    {contextWindowModeLabel(contextWindow)} · {formatContextTokens(contextWindow.effectiveTokens)}
                  </span>
                ) : null}
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
          activity={inspectorActivity}
          agents={agents}
          agentSeen={state.agentSeen}
          approvalMode={session?.approval_mode || approvalMode}
          allowNetwork={session?.allow_network ?? allowNetwork}
          changes={changes}
          context={state.context}
          contextWindow={contextWindow}
          modelName={modelName}
          modelSteps={state.modelSteps}
          planSteps={state.planSteps}
          running={running}
          session={session}
          tool={inspectorTool}
          totals={state.runTotals}
          undoBusy={undoBusy}
          workspacePath={workspacePath}
          {...inspectorActions}
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
