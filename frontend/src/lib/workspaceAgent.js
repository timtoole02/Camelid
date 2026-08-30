const MAX_WORKSPACE_ACTIVITY_EVENTS = 240

export const WORKSPACE_IDLE_STATE = Object.freeze({
  phase: 'idle',
  events: [],
  turns: [],
  totalTurns: 0,
  error: '',
})

function appendActivity(events, event) {
  const next = [...events, event]
  return next.length > MAX_WORKSPACE_ACTIVITY_EVENTS
    ? next.slice(-MAX_WORKSPACE_ACTIVITY_EVENTS)
    : next
}

export function reduceWorkspaceEvent(state, envelope) {
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
      error: '',
    }
  }
  if (event === 'session.starting') return { ...state, phase: 'starting', error: '' }
  if (event === 'turn.starting') return { ...state, phase: 'starting', error: '' }
  if (event === 'turn.stopping') return { ...state, phase: 'cancelling', error: '' }
  if (event === 'session.recovering') {
    return {
      ...state,
      phase: 'recovering',
      error: String(envelope.message || 'Workspace is reconciling an interrupted turn.'),
    }
  }
  if (event === 'turn.stop_failed') {
    return { ...state, phase: 'cancel_error', error: String(envelope.message || 'Workspace could not confirm that the turn stopped.') }
  }
  if (event === 'turn.user') {
    return {
      ...state,
      phase: 'running',
      events: appendActivity(state.events, envelope),
      turns: [...state.turns, { user: String(envelope.content || ''), assistant: '', outcome: '' }],
      totalTurns: Math.max(state.totalTurns || 0, state.turns.length) + 1,
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
    if (tail?.event === 'model.live') {
      events[events.length - 1] = { ...tail, content: `${tail.content}${content}` }
    } else {
      events.push({ ...envelope, event: 'model.live', content })
    }
    return {
      ...state,
      phase: state.phase === 'cancel_error' ? state.phase : 'running',
      events: events.slice(-MAX_WORKSPACE_ACTIVITY_EVENTS),
    }
  }

  if (event === 'tool.call' || event === 'model.answer') withoutLiveTail()
  events.push(envelope)
  if (events.length > MAX_WORKSPACE_ACTIVITY_EVENTS) {
    events.splice(0, events.length - MAX_WORKSPACE_ACTIVITY_EVENTS)
  }

  let turns = state.turns
  if (event === 'model.answer') {
    turns = [...state.turns]
    const last = turns.at(-1)
    if (last && !last.assistant) turns[turns.length - 1] = { ...last, assistant: String(envelope.content || ''), outcome: 'answered' }
    else turns.push({ user: '', assistant: String(envelope.content || ''), outcome: 'answered' })
  }

  if (event === 'approval.required') {
    return { ...state, phase: 'recovering', events, turns, error: 'Read-only Workspace received an unexpected approval request.' }
  }
  if (event === 'tool.result') {
    return { ...state, phase: state.phase === 'cancel_error' ? state.phase : 'running', events, turns }
  }
  if (event === 'session.finished') {
    if (envelope.outcome !== 'answered' && turns.length) {
      turns = [...turns]
      turns[turns.length - 1] = { ...turns.at(-1), outcome: envelope.outcome }
    }
    return { ...state, phase: envelope.outcome === 'answered' ? 'finished' : envelope.outcome, events, turns }
  }
  if (event === 'session.error') {
    return { ...state, phase: 'error', events, turns, error: String(envelope.message || 'Workspace stopped.') }
  }
  return {
    ...state,
    phase: event === 'session.started' && state.phase !== 'cancel_error' ? 'running' : state.phase,
    events,
    turns,
  }
}

export function workspaceEndpoint(apiBase, suffix = '') {
  const base = String(apiBase || '').replace(/\/$/, '')
  return `${base}/api/agent/workspace/sessions${suffix}`
}

export function workspaceModelsEndpoint(apiBase) {
  const base = String(apiBase || '').replace(/\/$/, '')
  return `${base}/api/agent/workspace/models`
}

export function workspaceThreadsEndpoint(apiBase, workspace, threadId = '') {
  const base = String(apiBase || '').replace(/\/$/, '')
  const suffix = threadId ? `/${encodeURIComponent(threadId)}` : ''
  return `${base}/api/agent/workspace/threads${suffix}?workspace=${encodeURIComponent(workspace)}`
}

export function workspaceCompactionEndpoint(apiBase, workspace, threadId) {
  const base = String(apiBase || '').replace(/\/$/, '')
  return `${base}/api/agent/workspace/threads/${encodeURIComponent(threadId)}/compact?workspace=${encodeURIComponent(workspace)}`
}

export async function getWorkspaceThreads(apiBase, workspace, { signal } = {}) {
  const response = await fetch(workspaceThreadsEndpoint(apiBase, workspace), { signal })
  if (!response.ok) throw new Error(await readError(response, `Saved Workspace threads failed (${response.status}).`))
  const payload = await response.json()
  return Array.isArray(payload?.threads) ? payload.threads : []
}

export async function getWorkspaceThread(apiBase, workspace, threadId, { signal } = {}) {
  const response = await fetch(workspaceThreadsEndpoint(apiBase, workspace, threadId), { signal })
  if (!response.ok) throw new Error(await readError(response, `Saved Workspace thread failed (${response.status}).`))
  return response.json()
}

export async function deleteWorkspaceThread(apiBase, workspace, threadId, { signal } = {}) {
  const response = await fetch(workspaceThreadsEndpoint(apiBase, workspace, threadId), { method: 'DELETE', signal })
  if (!response.ok) throw new Error(await readError(response, `Delete saved Workspace thread failed (${response.status}).`))
}

export async function compactWorkspaceThread(apiBase, workspace, threadId, undo = false, { signal } = {}) {
  const response = await fetch(workspaceCompactionEndpoint(apiBase, workspace, threadId), { method: undo ? 'DELETE' : 'POST', signal })
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

export async function createWorkspaceSession(apiBase, input, { signal } = {}) {
  const response = await fetch(workspaceEndpoint(apiBase), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
    signal,
  })
  if (!response.ok) throw new Error(await readError(response, `Workspace start failed (${response.status}).`))
  return response.json()
}

export async function sendWorkspaceMessage(apiBase, sessionId, text, clientMessageId, { signal } = {}) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}/messages`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, client_message_id: clientMessageId }),
    signal,
  })
  if (!response.ok) throw new Error(await readError(response, `Workspace follow-up failed (${response.status}).`))
  return response.json()
}

export async function getWorkspaceSession(apiBase, sessionId, { signal } = {}) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}`), { signal })
  if (!response.ok) throw new Error(await readError(response, `Workspace status failed (${response.status}).`))
  return response.json()
}

function abortableDelay(delayMs, signal) {
  if (!signal) return new Promise((resolve) => globalThis.setTimeout(resolve, delayMs))
  if (signal.aborted) return Promise.reject(signal.reason || new DOMException('The request was aborted.', 'AbortError'))
  return new Promise((resolve, reject) => {
    const timer = globalThis.setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, delayMs)
    const onAbort = () => {
      globalThis.clearTimeout(timer)
      reject(signal.reason || new DOMException('The request was aborted.', 'AbortError'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

export async function waitForWorkspaceSessionTerminal(apiBase, sessionId, { timeoutMs = 10000, pollMs = 100, signal } = {}) {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    const session = await getWorkspaceSession(apiBase, sessionId, { signal })
    if (!['waiting_for_events', 'running', 'cancelling'].includes(session.state)) return session
    if (Date.now() >= deadline) throw new Error('Workspace is still stopping. Retry Stop before sending another request.')
    await abortableDelay(pollMs, signal)
  }
}

export async function cancelWorkspaceSession(apiBase, sessionId, { signal } = {}) {
  const response = await fetch(workspaceEndpoint(apiBase, `/${encodeURIComponent(sessionId)}`), { method: 'DELETE', signal })
  if (!response.ok && response.status !== 404) throw new Error(await readError(response, `Stop failed (${response.status}).`))
  return response.status
}

export function workspaceFollowUpDisposition(response, expectedSessionId) {
  if (!response || String(response.session_id || '') !== String(expectedSessionId || '')) {
    throw new Error('Workspace follow-up returned a mismatched session identity.')
  }
  const state = String(response.state || '')
  if (state === 'waiting_for_events') return 'stream'
  if (response.duplicate && ['idle', 'cancelled', 'failed', 'error'].includes(state)) return 'restore'
  return 'recover'
}

export function workspaceSessionMatchesRuntime(session, runtime, toolCapable) {
  return Boolean(
    session
    && toolCapable
    && runtime?.status === 'online'
    && runtime?.loaded_now
    && runtime?.generation_ready
    && runtime?.active_model_id
    && String(runtime.active_model_id) === String(session.model_id),
  )
}
