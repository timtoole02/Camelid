const MAX_WORKSPACE_ACTIVITY_EVENTS = 240

export const WORKSPACE_IDLE_STATE = Object.freeze({
  phase: 'idle',
  events: [],
  turns: [],
  totalTurns: 0,
  error: '',
  approval: null,
  // Monotonic counter bumped on every accepted event. A streamed delta is
  // MERGED into the tail rather than appended, so `events.length` does not
  // change while text arrives — anything that must react per token (the
  // autoscroll) has to watch this instead.
  revision: 0,
  // Per-arrival identity stamped onto every appended envelope as `uid`. The
  // server's own `sequence` is per-EVENT-STREAM and a follow-up turn opens a
  // new stream that restarts it at 1, so `sequence` collides across the turns
  // of one conversation and React reconciles two different cards onto one key.
  eventSeq: 0,
  // Turns whose activity is being rendered from `events` rather than as a
  // finished history pair. Counted, never inferred from `events`: the ring
  // buffer drops a turn's `turn.user` marker first, and Code mode has no step
  // cap to bound how much a single turn can push through it.
  liveTurns: 0,
  // Maintained here so the view never has to scan the event list on render.
  latestTool: null,
  latestResult: null,
  liveActivity: null,
  agents: [],
})

function appendActivity(events, event) {
  const next = [...events, event]
  return next.length > MAX_WORKSPACE_ACTIVITY_EVENTS
    ? next.slice(-MAX_WORKSPACE_ACTIVITY_EVENTS)
    : next
}

function reduceAgentEvent(state, envelope, allowApprovals) {
  const event = envelope?.event
  if (!event) return state
  if (event === 'session.reset') return { ...WORKSPACE_IDLE_STATE, events: [], turns: [], totalTurns: 0 }
  if (event === 'thread.restored') {
    const turns = (Array.isArray(envelope.turns) ? envelope.turns : []).map((turn) => ({
      user: String(turn.user_text || ''),
      assistant: String(turn.assistant_text || ''),
      outcome: String(turn.terminal_outcome || 'answered'),
    }))
    return {
      ...state,
      phase: 'idle',
      events: [],
      turns,
      totalTurns: Math.max(turns.length, Number(envelope.turnCount || 0)),
      // A restore replaces the event list, so every restored turn is history.
      liveTurns: 0,
      liveActivity: null,
      agents: [],
      error: '',
    }
  }
  if (event === 'session.starting' || event === 'turn.starting') {
    const task = String(envelope.task || '')
    const activity = {
      phase: 'starting',
      stage: 'starting',
      detail: 'Starting coding session',
      task,
      started_at_ms: Date.now(),
      updated_at_ms: Date.now(),
      agents: [{ id: 'main', parent_id: null, label: 'Camelid', status: 'starting', task, detail: 'Preparing the coding session' }],
    }
    return { ...state, phase: 'starting', error: '', approval: null, liveActivity: activity, agents: activity.agents }
  }
  if (event === 'turn.stopping') {
    const liveActivity = state.liveActivity
      ? { ...state.liveActivity, phase: 'cancelling', stage: 'stopping', detail: 'Stopping the active coding turn', updated_at_ms: Date.now() }
      : null
    return { ...state, phase: 'cancelling', liveActivity, error: '' }
  }
  if (event === 'turn.stop_failed') {
    const message = String(envelope.message || 'Workspace could not confirm that the turn stopped.')
    const liveActivity = state.liveActivity
      ? { ...state.liveActivity, phase: 'cancel_error', stage: 'warning', detail: message, updated_at_ms: Date.now() }
      : null
    return { ...state, phase: 'cancel_error', liveActivity, error: message }
  }
  if (event === 'activity.snapshot') {
    const activity = envelope.activity || null
    if (!activity) return state
    if (state.liveActivity?.updated_at_ms === activity.updated_at_ms
      && state.liveActivity?.phase === activity.phase) return state
    return {
      ...state,
      phase: String(activity.phase || state.phase || 'idle'),
      liveActivity: activity,
      agents: Array.isArray(activity.agents) ? activity.agents : state.agents,
      error: activity.phase === 'error' ? String(activity.detail || 'Workspace stopped.') : state.error,
    }
  }
  const eventSeq = (state.eventSeq || 0) + 1
  const arrival = { ...envelope, uid: eventSeq }
  if (event === 'turn.user') {
    return {
      ...state,
      phase: 'running',
      events: appendActivity(state.events, arrival),
      turns: [...state.turns, { user: String(envelope.content || ''), assistant: '', outcome: '' }],
      // `turns` already includes the turn being appended, so this is the count
      // itself — the old `max(total, turns.length) + 1` double-counted every
      // turn after a restore and drifted the header away from reality.
      totalTurns: Math.max(state.totalTurns || 0, state.turns.length + 1),
      eventSeq,
      liveTurns: (state.liveTurns || 0) + 1,
    }
  }
  const events = [...state.events]
  const withoutLiveTail = () => {
    if (events.at(-1)?.event === 'model.live') events.pop()
  }

  if (event === 'model.delta') {
    const content = String(envelope.content || '')
    if (!content) return state
    const tail = events.at(-1)
    const merged = tail?.event === 'model.live'
    if (merged) {
      events[events.length - 1] = { ...tail, content: `${tail.content}${content}` }
    } else {
      events.push({ ...arrival, event: 'model.live', content })
    }
    return {
      ...state,
      phase: state.phase === 'cancel_error' ? state.phase : 'running',
      events: events.slice(-MAX_WORKSPACE_ACTIVITY_EVENTS),
      liveActivity: advanceLiveActivity(state.liveActivity, envelope),
      // A merged delta reuses the tail entry's identity; only a new one consumes.
      eventSeq: merged ? (state.eventSeq || 0) : eventSeq,
    }
  }

  if (event === 'tool.call' || event === 'model.answer') withoutLiveTail()
  events.push(arrival)
  if (events.length > MAX_WORKSPACE_ACTIVITY_EVENTS) {
    events.splice(0, events.length - MAX_WORKSPACE_ACTIVITY_EVENTS)
  }

  let turns = state.turns
  let liveTurns = state.liveTurns || 0
  if (event === 'model.answer') {
    turns = [...state.turns]
    const last = turns.at(-1)
    if (last && !last.assistant) turns[turns.length - 1] = { ...last, assistant: String(envelope.content || ''), outcome: 'answered' }
    else {
      turns.push({ user: '', assistant: String(envelope.content || ''), outcome: 'answered' })
      liveTurns += 1
    }
  }
  // Every branch below keeps the appended event, so they share one base.
  const appended = {
    ...state,
    events,
    turns,
    eventSeq,
    liveTurns,
    liveActivity: advanceLiveActivity(state.liveActivity, envelope),
  }

  if (event === 'approval.required') {
    if (allowApprovals) {
      return { ...appended, phase: 'awaiting_approval', approval: envelope, error: '' }
    }
    return { ...appended, phase: 'error', error: 'Read-only Workspace received an unexpected approval request.' }
  }
  if (event === 'approval.resolved') {
    return { ...appended, phase: 'running', approval: null, error: '' }
  }
  if (event === 'tool.result') {
    return { ...appended, phase: state.phase === 'cancel_error' ? state.phase : 'running', approval: null }
  }
  if (event === 'agent.updated') {
    const incoming = {
      id: String(envelope.agent_id || 'agent'),
      parent_id: envelope.parent_id || null,
      label: String(envelope.label || envelope.agent_id || 'Agent'),
      status: String(envelope.status || 'running'),
      task: String(envelope.task || ''),
      detail: String(envelope.detail || ''),
    }
    const agents = [...(state.agents || [])]
    const index = agents.findIndex((agent) => agent.id === incoming.id
      || (agent.parent_id && incoming.parent_id && agent.label === incoming.label))
    if (index === -1) agents.push(incoming)
    else agents[index] = { ...agents[index], ...incoming, task: incoming.task || agents[index].task }
    return { ...appended, agents }
  }
  if (event === 'session.finished') {
    if (envelope.outcome !== 'answered' && turns.length) {
      turns = [...turns]
      turns[turns.length - 1] = { ...turns.at(-1), outcome: envelope.outcome }
    }
    const terminalStatus = envelope.outcome === 'answered' ? 'completed' : 'stopped'
    const agents = (state.agents || []).map((agent) => ({
      ...agent,
      status: ['starting', 'running', 'waiting'].includes(agent.status) ? terminalStatus : agent.status,
      detail: agent.id === 'main' ? terminalActivityDetail(envelope.outcome) : agent.detail,
    }))
    return { ...appended, turns, agents, phase: envelope.outcome === 'answered' ? 'finished' : envelope.outcome, approval: null }
  }
  if (event === 'session.error') {
    const agents = (state.agents || []).map((agent) => ({
      ...agent,
      status: ['starting', 'running', 'waiting'].includes(agent.status) ? 'failed' : agent.status,
    }))
    return { ...appended, agents, phase: 'error', error: String(envelope.message || 'Workspace stopped.') }
  }
  return {
    ...appended,
    phase: event === 'session.started' && state.phase !== 'cancel_error' ? 'running' : state.phase,
  }
}

/// Bump the revision and keep the derived tool pointers current. Returns the
/// SAME object when the inner reducer ignored the event, so React can skip the
/// re-render entirely.
function withDerived(previous, next, envelope) {
  if (next === previous) return previous
  const event = envelope?.event
  // A reset is a return to the idle state, revision counter included.
  if (event === 'session.reset') return next
  const derived = { ...next, revision: (previous.revision || 0) + 1 }
  if (event === 'tool.call') derived.latestTool = envelope
  else if (event === 'tool.result') {
    derived.latestTool = null
    derived.latestResult = envelope
  }
  else if (event === 'thread.restored') {
    derived.latestTool = null
    derived.latestResult = null
  }
  return derived
}

export function reduceWorkspaceEvent(state, envelope) {
  return withDerived(state, reduceAgentEvent(state, envelope, false), envelope)
}

export function reduceCodeEvent(state, envelope) {
  return withDerived(state, reduceAgentEvent(state, envelope, true), envelope)
}

export function workspaceEndpoint(apiBase, suffix = '') {
  const base = String(apiBase || '').replace(/\/$/, '')
  return `${base}/api/agent/workspace/sessions${suffix}`
}

function formatActivityTool(tool) {
  return String(tool || 'tool').replaceAll('_', ' ')
}

function terminalActivityDetail(outcome) {
  if (outcome === 'answered') return 'The coding turn completed'
  if (outcome === 'repeated') return 'Stopped because the model repeated actions without making progress'
  if (outcome === 'step_capped') return 'Stopped after reaching the step limit'
  if (outcome === 'aborted' || outcome === 'cancelled') return 'The coding turn was stopped'
  if (outcome === 'driver_error' || outcome === 'failed') return 'The model or runtime failed'
  return 'The coding turn ended'
}

function advanceLiveActivity(current, envelope) {
  if (!current || !envelope?.event) return current
  const next = { ...current, updated_at_ms: Date.now() }
  const event = envelope.event
  if (event === 'session.started') {
    Object.assign(next, { phase: 'running', stage: 'starting', detail: 'Preparing the model and workspace tools' })
  } else if (event === 'turn.started') {
    Object.assign(next, { phase: 'running', stage: 'context', detail: 'Building the model context' })
  } else if (event === 'memory.updated') {
    Object.assign(next, {
      phase: 'running',
      stage: 'context',
      detail: `Prepared ${envelope.prompt_tokens || 0} prompt tokens inside a ${envelope.budget_total || 0}-token context`,
    })
  } else if (event === 'model.delta') {
    Object.assign(next, { phase: 'running', stage: 'generating', detail: 'The model is generating its next action', current_tool: null })
  } else if (event === 'model.timing') {
    Object.assign(next, {
      output_tokens: Number.isFinite(envelope.output_tokens) ? envelope.output_tokens : next.output_tokens,
      total_model_ms: Number.isFinite(envelope.total_ms) ? envelope.total_ms : next.total_model_ms,
      ttft_ms: Number.isFinite(envelope.ttft_ms) ? envelope.ttft_ms : next.ttft_ms,
      prefill_ms: Number.isFinite(envelope.prefill_ms) ? envelope.prefill_ms : next.prefill_ms,
      reused_tokens: Number.isFinite(envelope.reused_tokens) ? envelope.reused_tokens : next.reused_tokens,
      prefilled_tokens: Number.isFinite(envelope.prefilled_tokens) ? envelope.prefilled_tokens : next.prefilled_tokens,
      engine_rebuilt: typeof envelope.engine_rebuilt === 'boolean' ? envelope.engine_rebuilt : next.engine_rebuilt,
      resident_backend: envelope.resident_backend || next.resident_backend,
      detail: `${Number.isFinite(envelope.output_tokens) ? `${envelope.output_tokens} output tokens` : 'Generation step'}${Number.isFinite(envelope.reused_tokens) && envelope.reused_tokens > 0 ? ` · reused ${envelope.reused_tokens} prompt tokens` : ''}`,
    })
  } else if (event === 'model.answer') {
    Object.assign(next, { phase: 'running', stage: 'finishing', detail: 'Reviewing and saving the final answer', current_tool: null })
  } else if (event === 'index.timing') {
    Object.assign(next, {
      index_refresh_ms: Number(envelope.elapsed_ms || 0),
      index_files_hashed: Number(envelope.files_hashed || 0),
      index_files_reused: Number(envelope.files_reused || 0),
    })
  } else if (event === 'tool.call') {
    Object.assign(next, { phase: 'running', stage: 'tool', detail: String(envelope.detail || 'Running a tool'), current_tool: envelope.detail || null })
  } else if (event === 'tool.result') {
    Object.assign(next, {
      phase: 'running',
      stage: 'tool_result',
      detail: `${formatActivityTool(envelope.tool)} ${envelope.outcome === 'error' ? 'failed' : 'completed'}`,
      current_tool: null,
    })
  } else if (event === 'tool.timing') {
    next.last_tool_ms = Number(envelope.elapsed_ms || 0)
  } else if (event === 'approval.required') {
    Object.assign(next, { phase: 'awaiting_approval', stage: 'approval', detail: `Waiting for approval to run ${formatActivityTool(envelope.tool)}` })
  } else if (event === 'approval.resolved') {
    Object.assign(next, { phase: 'running', stage: 'approval', detail: 'Approval received; resuming the agent' })
  } else if (event === 'session.notice') {
    next.detail = String(envelope.content || next.detail)
  } else if (event === 'agent.updated' && envelope.agent_id === 'main') {
    if (envelope.task) next.task = String(envelope.task)
    if (envelope.detail) next.detail = String(envelope.detail)
  } else if (event === 'session.finished') {
    Object.assign(next, {
      phase: String(envelope.outcome || 'finished'),
      stage: 'finished',
      current_tool: null,
      terminal_outcome: String(envelope.outcome || 'finished'),
      detail: terminalActivityDetail(envelope.outcome),
    })
  } else if (event === 'session.error') {
    Object.assign(next, { phase: 'error', stage: 'finished', current_tool: null, terminal_outcome: 'error', detail: String(envelope.message || 'Workspace stopped.') })
  }
  return next
}

export function workspaceActivityEndpoint(apiBase) {
  const base = String(apiBase || '').replace(/\/$/, '')
  return `${base}/api/agent/workspace/activity`
}

export async function getWorkspaceActivity(apiBase, { signal } = {}) {
  const response = await fetch(workspaceActivityEndpoint(apiBase), { signal })
  if (!response.ok) throw new Error(await readError(response, `Workspace activity failed (${response.status}).`))
  const payload = await response.json()
  return payload?.activity || null
}

export function workspaceModelsEndpoint(apiBase) {
  const base = String(apiBase || '').replace(/\/$/, '')
  return `${base}/api/agent/workspace/models`
}

export function workspaceThreadsEndpoint(apiBase, workspace, threadId = '', mode = 'read_only') {
  const base = String(apiBase || '').replace(/\/$/, '')
  const suffix = threadId ? `/${encodeURIComponent(threadId)}` : ''
  const modeQuery = mode === 'read_only' ? '' : `&mode=${encodeURIComponent(mode)}`
  return `${base}/api/agent/workspace/threads${suffix}?workspace=${encodeURIComponent(workspace)}${modeQuery}`
}

export function workspaceCompactionEndpoint(apiBase, workspace, threadId) {
  const base = String(apiBase || '').replace(/\/$/, '')
  return `${base}/api/agent/workspace/threads/${encodeURIComponent(threadId)}/compact?workspace=${encodeURIComponent(workspace)}`
}

export async function getWorkspaceThreads(apiBase, workspace, { signal, mode = 'read_only' } = {}) {
  const response = await fetch(workspaceThreadsEndpoint(apiBase, workspace, '', mode), { signal })
  if (!response.ok) throw new Error(await readError(response, `Saved Workspace threads failed (${response.status}).`))
  const payload = await response.json()
  return Array.isArray(payload?.threads) ? payload.threads : []
}

export async function getRecentCodeThreads(apiBase, { signal } = {}) {
  const base = String(apiBase || '').replace(/\/$/, '')
  const response = await fetch(`${base}/api/agent/workspace/threads/recent?mode=code`, { signal })
  if (!response.ok) throw new Error(await readError(response, `Coding history failed (${response.status}).`))
  const payload = await response.json()
  return Array.isArray(payload?.threads) ? payload.threads : []
}

export async function getWorkspaceThread(apiBase, workspace, threadId, { signal } = {}) {
  const response = await fetch(workspaceThreadsEndpoint(apiBase, workspace, threadId), { signal })
  if (!response.ok) throw new Error(await readError(response, `Saved Workspace thread failed (${response.status}).`))
  return response.json()
}

export async function deleteWorkspaceThread(apiBase, workspace, threadId) {
  const response = await fetch(workspaceThreadsEndpoint(apiBase, workspace, threadId), { method: 'DELETE' })
  if (!response.ok) throw new Error(await readError(response, `Delete saved Workspace thread failed (${response.status}).`))
}

export async function compactWorkspaceThread(apiBase, workspace, threadId, undo = false) {
  const response = await fetch(workspaceCompactionEndpoint(apiBase, workspace, threadId), { method: undo ? 'DELETE' : 'POST' })
  if (!response.ok) throw new Error(await readError(response, `Workspace compaction failed (${response.status}).`))
  return response.json()
}

async function readError(response, fallback) {
  try {
    const payload = await response.json()
    return payload?.error?.message || fallback
  } catch {
    return fallback
  }
}

export async function getWorkspaceCompatibleModels(apiBase, { signal } = {}) {
  const response = await fetch(workspaceModelsEndpoint(apiBase), { signal })
  if (!response.ok) throw new Error(await readError(response, `Workspace model check failed (${response.status}).`))
  const contentType = String(response.headers.get('content-type') || '').toLowerCase()
  if (!contentType.includes('application/json')) {
    throw new Error('Workspace model recommendations are unavailable from this running backend.')
  }
  const payload = await response.json()
  return Array.isArray(payload?.models) ? payload.models : []
}

export function workspaceBrowseEndpoint(apiBase, path = null) {
  const base = String(apiBase || '').replace(/\/$/, '')
  const query = path ? `?path=${encodeURIComponent(path)}` : ''
  return `${base}/api/agent/workspace/browse${query}`
}

export async function browseWorkspaceFolders(apiBase, path = null, { signal } = {}) {
  const response = await fetch(workspaceBrowseEndpoint(apiBase, path), { signal })
  if (!response.ok) throw new Error(await readError(response, `Could not open that folder (${response.status}).`))
  const contentType = String(response.headers.get('content-type') || '').toLowerCase()
  if (!contentType.includes('application/json')) {
    throw new Error('Folder browsing is unavailable from this running backend.')
  }
  const payload = await response.json()
  return {
    path: payload?.path ?? null,
    parent: payload?.parent ?? null,
    hasRoots: Boolean(payload?.has_roots),
    separator: typeof payload?.separator === 'string' ? payload.separator : '/',
    entries: Array.isArray(payload?.entries) ? payload.entries : [],
    truncated: Boolean(payload?.truncated),
  }
}

export async function createWorkspaceSession(apiBase, input) {
  const response = await fetch(workspaceEndpoint(apiBase), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
  if (!response.ok) throw new Error(await readError(response, `Workspace start failed (${response.status}).`))
  return response.json()
}

export async function sendWorkspaceMessage(apiBase, sessionId, text, clientMessageId) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}/messages`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, client_message_id: clientMessageId }),
  })
  if (!response.ok) throw new Error(await readError(response, `Workspace follow-up failed (${response.status}).`))
  return response.json()
}

export async function getWorkspaceSession(apiBase, sessionId) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}`))
  if (!response.ok) throw new Error(await readError(response, `Workspace status failed (${response.status}).`))
  return response.json()
}

export async function waitForWorkspaceSessionTerminal(apiBase, sessionId, { timeoutMs = 10000, pollMs = 100 } = {}) {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    const session = await getWorkspaceSession(apiBase, sessionId)
    if (!['waiting_for_events', 'running', 'cancelling'].includes(session.state)) return session
    if (Date.now() >= deadline) throw new Error('Workspace is still stopping. Retry Stop before sending another request.')
    await new Promise((resolve) => globalThis.setTimeout(resolve, pollMs))
  }
}

export async function cancelWorkspaceSession(apiBase, sessionId) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}`), { method: 'DELETE' })
  if (!response.ok && response.status !== 404) throw new Error(await readError(response, `Stop failed (${response.status}).`))
  return response.status
}

export async function decideWorkspaceApproval(apiBase, sessionId, approvalId, decision) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}/decisions`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approval_id: approvalId, decision }),
  })
  if (!response.ok) throw new Error(await readError(response, `Approval decision failed (${response.status}).`))
}

export async function getWorkspaceChanges(apiBase, sessionId) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}/changes`))
  if (!response.ok) throw new Error(await readError(response, `File changes failed (${response.status}).`))
  return response.json()
}

export async function undoWorkspaceChange(apiBase, sessionId, force = false) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}/undo`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force }),
  })
  if (!response.ok) throw new Error(await readError(response, `Undo failed (${response.status}).`))
  return response.json()
}
