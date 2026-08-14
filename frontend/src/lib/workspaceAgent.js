const MAX_WORKSPACE_ACTIVITY_EVENTS = 240

/// Rows kept for the inspector's model-step table. The aggregate lives in
/// `runTotals`, which is a counter, so capping rows never changes the total.
const MAX_MODEL_STEP_ROWS = 60
/// Plan steps are model-authored and nothing bounds how many lines
/// `update_plan` can emit, so the list is capped before it reaches a render.
const MAX_PLAN_STEPS = 120
const PLAN_STEP_LINE = /^\[([x~ ])\]\s+(.+)$/

/// The only structured plan the agent publishes. Exported so the transcript
/// card and the inspector's Plan group read one parse, not two that can drift.
export function parsePlanSteps(content) {
  const steps = []
  for (const line of String(content || '').split(/\r?\n/)) {
    const match = PLAN_STEP_LINE.exec(line.trim())
    if (!match) continue
    steps.push({ status: match[1] === 'x' ? 'done' : match[1] === '~' ? 'active' : 'pending', text: match[2] })
    if (steps.length >= MAX_PLAN_STEPS) break
  }
  return steps
}

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
  // Monotonic run totals. Accumulated HERE rather than recomputed by scanning
  // `events`, for the same reason `liveTurns` is: that array is a ring capped
  // at MAX_WORKSPACE_ACTIVITY_EVENTS, so a long turn evicts its oldest
  // `model.timing` and `tool.result` entries and any re-scan silently
  // UNDERCOUNTS. A sidebar readout that drifts downward as a run gets longer
  // is worse than no readout.
  runTotals: Object.freeze({ steps: 0, outputTokens: 0, elapsedMs: 0, tools: 0, toolFailures: 0 }),
  // Latest `memory.updated` snapshot: the live context-budget position and its
  // composition. Replaced wholesale, never appended to — this is the current
  // state of the window, not a history of it.
  context: null,
  // The last `update_plan` result, already parsed. Held here because the event
  // that carried it is among the first the 240-entry ring evicts.
  planSteps: [],
  // One row per completed model step, capped at MAX_MODEL_STEP_ROWS.
  modelSteps: [],
  // Client-observed first/last sighting per agent id. Deliberately NOT stored
  // on the agent objects: `activity.snapshot` replaces `agents` wholesale about
  // once a second with the six server fields, so anything derived onto an agent
  // is wiped every poll. The server reports no per-agent clock at all.
  agentSeen: {},
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
      planSteps: [],
      modelSteps: [],
      agentSeen: {},
      error: '',
    }
  }
  if (event === 'session.starting' || event === 'turn.starting') {
    const task = String(envelope.task || '')
    const main = { id: 'main', parent_id: null, label: 'Camelid', status: 'starting', task, detail: 'Preparing the coding session' }
    // A follow-up turn KEEPS the agents earlier turns finished — the panel
    // groups them under Finished, and wiping the list every turn made that
    // group unreachable in the common case. A new session starts empty.
    const carried = event === 'turn.starting' ? (state.agents || []).filter((agent) => agent.id !== 'main') : []
    const agents = [main, ...carried]
    const activity = {
      phase: 'starting',
      stage: 'starting',
      detail: 'Starting coding session',
      task,
      started_at_ms: Date.now(),
      updated_at_ms: Date.now(),
      agents,
    }
    return {
      ...state,
      phase: 'starting',
      error: '',
      approval: null,
      liveActivity: activity,
      agents,
      // The plan belongs to the turn that published it.
      planSteps: [],
      agentSeen: event === 'turn.starting' ? (state.agentSeen || {}) : {},
    }
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
    const snapshotAgents = Array.isArray(activity.agents) ? activity.agents : state.agents
    // The poll fires every second and `updated_at_ms` moves on every applied
    // event, so the early return above never holds during a run. Reusing the
    // previous array when nothing the panel shows has changed is what lets the
    // memoized inspector skip that re-render.
    const sameAgents = snapshotAgents.length === state.agents.length
      && snapshotAgents.every((agent, index) => {
        const previous = state.agents[index]
        return previous && agent.id === previous.id && agent.status === previous.status
          && agent.label === previous.label && agent.task === previous.task && agent.detail === previous.detail
      })
    let agentSeen = state.agentSeen || {}
    const seenAt = Date.now()
    for (const agent of snapshotAgents) {
      const id = String(agent?.id || '')
      if (id && !agentSeen[id]) agentSeen = { ...agentSeen, [id]: { firstSeenAt: seenAt, lastSeenAt: seenAt } }
    }
    return {
      ...state,
      phase: String(activity.phase || state.phase || 'idle'),
      liveActivity: activity,
      agents: sameAgents ? state.agents : snapshotAgents,
      agentSeen,
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
    const plan = envelope.tool === 'update_plan' && envelope.outcome !== 'error'
      ? parsePlanSteps(envelope.content)
      : null
    return {
      ...appended,
      phase: state.phase === 'cancel_error' ? state.phase : 'running',
      approval: null,
      // Only replace on a parse that produced steps: a plan the agent wrote as
      // free prose must not wipe the last good one.
      planSteps: plan && plan.length ? plan : state.planSteps,
    }
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
    const seenAt = Date.now()
    const seen = (state.agentSeen || {})[incoming.id]
    return {
      ...appended,
      agents,
      agentSeen: { ...(state.agentSeen || {}), [incoming.id]: { firstSeenAt: seen?.firstSeenAt ?? seenAt, lastSeenAt: seenAt } },
    }
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

  // ---- Monotonic totals (see `runTotals` in WORKSPACE_IDLE_STATE) ----
  // A restore replaces the event list wholesale, so its totals restart too;
  // anything else only ever adds.
  if (event === 'thread.restored') {
    derived.runTotals = WORKSPACE_IDLE_STATE.runTotals
    derived.context = null
    derived.modelSteps = []
  } else if (event === 'model.timing') {
    const base = previous.runTotals || WORKSPACE_IDLE_STATE.runTotals
    derived.runTotals = {
      ...base,
      steps: base.steps + 1,
      outputTokens: base.outputTokens + numeric(envelope.output_tokens),
      elapsedMs: base.elapsedMs + numeric(envelope.total_ms),
    }
    derived.modelSteps = [...(previous.modelSteps || []), {
      index: base.steps + 1,
      totalMs: Number.isFinite(envelope.total_ms) ? envelope.total_ms : null,
      ttftMs: Number.isFinite(envelope.ttft_ms) ? envelope.ttft_ms : null,
      outputTokens: Number.isFinite(envelope.output_tokens) ? envelope.output_tokens : null,
    }].slice(-MAX_MODEL_STEP_ROWS)
  } else if (event === 'tool.result') {
    const base = previous.runTotals || WORKSPACE_IDLE_STATE.runTotals
    derived.runTotals = {
      ...base,
      tools: base.tools + 1,
      toolFailures: base.toolFailures + (envelope.outcome === 'error' ? 1 : 0),
    }
  } else if (event === 'memory.updated') {
    derived.context = {
      promptTokens: numeric(envelope.prompt_tokens),
      generationTokens: numeric(envelope.generation_tokens),
      budgetTotal: numeric(envelope.budget_total),
      // Composition, in the order the window is actually assembled.
      parts: [
        { key: 'system', label: 'System', tokens: numeric(envelope.system_tokens_estimate) },
        { key: 'tools', label: 'Tool schemas', tokens: numeric(envelope.tool_definition_tokens_estimate) },
        { key: 'messages', label: 'Messages', tokens: numeric(envelope.message_tokens_estimate) },
        { key: 'recent', label: 'Recent memory', tokens: numeric(envelope.recent_memory_tokens_estimate) },
        { key: 'retrieved', label: 'Retrieved memory', tokens: numeric(envelope.retrieved_memory_tokens_estimate) },
        { key: 'evidence', label: 'Evidence', tokens: numeric(envelope.evidence_memory_tokens_estimate) },
        { key: 'results', label: 'Tool results', tokens: numeric(envelope.tool_result_tokens_estimate) },
      ],
    }
  }
  return derived
}

/** Event payloads are JSON off the wire; a missing or malformed field must
 *  contribute zero to a running total rather than poison it with NaN. */
function numeric(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
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
      detail: Number.isFinite(envelope.output_tokens)
        ? `The model finished a ${envelope.output_tokens}-token generation step`
        : 'The model finished a generation step',
    })
  } else if (event === 'model.answer') {
    Object.assign(next, { phase: 'running', stage: 'finishing', detail: 'Reviewing and saving the final answer', current_tool: null })
  } else if (event === 'tool.call') {
    Object.assign(next, { phase: 'running', stage: 'tool', detail: String(envelope.detail || 'Running a tool'), current_tool: envelope.detail || null })
  } else if (event === 'tool.result') {
    Object.assign(next, {
      phase: 'running',
      stage: 'tool_result',
      detail: `${formatActivityTool(envelope.tool)} ${envelope.outcome === 'error' ? 'failed' : 'completed'}`,
      current_tool: null,
    })
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

/// Delete one saved coding session. The server refuses to delete the thread of a
/// session that is still running (it returns a 409 naming the live turn), so the
/// message it sends back is surfaced rather than replaced with a generic one.
export async function deleteCodeThread(apiBase, threadId, workspace) {
  const base = String(apiBase || '').replace(/\/$/, '')
  // `workspace` is REQUIRED by the endpoint, which cross-checks it against the
  // thread's stored canonical_root and rejects the request outright without it
  // ("missing field `workspace`"). Pass the thread's own root.
  const query = new URLSearchParams({ workspace: String(workspace || ''), mode: 'code' })
  const response = await fetch(
    `${base}/api/agent/workspace/threads/${encodeURIComponent(threadId)}?${query}`,
    { method: 'DELETE' },
  )
  if (!response.ok) {
    throw new Error(await readError(response, `Coding session could not be deleted (${response.status}).`))
  }
  return true
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
