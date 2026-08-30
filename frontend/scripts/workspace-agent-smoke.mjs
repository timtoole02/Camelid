import assert from 'node:assert/strict'
import {
  cancelWorkspaceSession,
  createWorkspaceSession,
  getWorkspaceSession,
  reduceWorkspaceEvent,
  sendWorkspaceMessage,
  waitForWorkspaceSessionTerminal,
  WORKSPACE_IDLE_STATE,
  workspaceBrowseEndpoint,
  workspaceCompactionEndpoint,
  workspaceEndpoint,
  workspaceFollowUpDisposition,
  workspaceModelsEndpoint,
  workspaceSessionMatchesRuntime,
  workspaceThreadsEndpoint,
} from '../src/lib/workspaceAgent.js'

let state = { ...WORKSPACE_IDLE_STATE, events: [] }
state = reduceWorkspaceEvent(state, { event: 'session.started', model_id: 'tool-model', sequence: 1 })
state = reduceWorkspaceEvent(state, { event: 'model.delta', content: '<tool_', sequence: 2 })
state = reduceWorkspaceEvent(state, { event: 'model.delta', content: 'call>', sequence: 3 })
assert.equal(state.events.at(-1).event, 'model.live')
assert.equal(state.events.at(-1).content, '<tool_call>')

state = reduceWorkspaceEvent(state, { event: 'tool.call', detail: 'read_file(a.txt)', sequence: 4 })
assert.equal(state.events.at(-1).event, 'tool.call')
assert.equal(state.events.some((event) => event.event === 'model.live'), false, 'raw streamed tool syntax must be removed')

state = reduceWorkspaceEvent(state, { event: 'model.delta', content: 'Done', sequence: 6 })
state = reduceWorkspaceEvent(state, { event: 'model.answer', content: 'Done', sequence: 7 })
assert.equal(state.events.at(-1).event, 'model.answer')
assert.equal(state.events.filter((event) => event.event === 'model.live').length, 0)

state = reduceWorkspaceEvent(state, {
  event: 'memory.compacted', compacted_through_turn: 3, archived_turns: 4,
  compaction_count: 1, trigger_tokens: 3072, budget_total: 4096, sequence: 8,
})
assert.equal(state.events.at(-1).event, 'memory.compacted')
assert.equal(state.events.at(-1).archived_turns, 4)

state = reduceWorkspaceEvent(state, { event: 'session.finished', outcome: 'answered', sequence: 9 })
assert.equal(state.phase, 'finished')

const impossibleApproval = reduceWorkspaceEvent(
  { ...WORKSPACE_IDLE_STATE, events: [] },
  { event: 'approval.required', approval_id: 'unexpected' },
)
assert.equal(impossibleApproval.phase, 'recovering')
assert.match(impossibleApproval.error, /unexpected approval request/)
const recovering = reduceWorkspaceEvent(
  { ...WORKSPACE_IDLE_STATE, events: [], turns: [] },
  { event: 'session.recovering', message: 'stream interrupted' },
)
assert.equal(recovering.phase, 'recovering')
assert.equal(recovering.error, 'stream interrupted')

state = reduceWorkspaceEvent(state, { event: 'session.reset' })
assert.deepEqual(state, { ...WORKSPACE_IDLE_STATE, events: [] })
state = reduceWorkspaceEvent(state, {
  event: 'thread.restored',
  turnCount: 7,
  turns: [{ user_text: 'Where is login?', assistant_text: 'In src/auth.rs.' }],
})
assert.deepEqual(state.turns, [{ user: 'Where is login?', assistant: 'In src/auth.rs.', outcome: 'answered' }])
assert.equal(state.totalTurns, 7)
assert.equal(state.events.length, 0)
state = reduceWorkspaceEvent(state, { event: 'turn.starting' })
assert.equal(state.phase, 'starting')
assert.equal(state.turns.length, 1, 'starting a follow-up must preserve restored turns')
state = reduceWorkspaceEvent(state, { event: 'turn.stopping' })
assert.equal(state.phase, 'cancelling')
assert.equal(state.turns.length, 1, 'stopping must preserve durable turns')
state = reduceWorkspaceEvent(state, { event: 'turn.stop_failed', message: 'still running' })
assert.equal(state.phase, 'cancel_error')
assert.equal(state.error, 'still running')
state = reduceWorkspaceEvent(state, { event: 'session.started', model_id: 'late-model' })
state = reduceWorkspaceEvent(state, { event: 'model.delta', content: 'late delta' })
state = reduceWorkspaceEvent(state, { event: 'tool.result', tool: 'read_file', outcome: 'ok', content: 'late result' })
assert.equal(state.phase, 'cancel_error', 'late nonterminal SSE events must not erase a failed Stop')

const originalFetch = globalThis.fetch
const terminalStates = ['running', 'cancelling', 'cancelled']
let statusReads = 0
const waitController = new AbortController()
globalThis.fetch = async (_url, options) => {
  assert.equal(options.signal, waitController.signal, 'terminal polling must forward its AbortSignal')
  return new Response(JSON.stringify({ state: terminalStates[statusReads++] }), {
  status: 200,
  headers: { 'Content-Type': 'application/json' },
  })
}
try {
  const settled = await waitForWorkspaceSessionTerminal('http://127.0.0.1:8181', 'thread-1', { timeoutMs: 1000, pollMs: 0, signal: waitController.signal })
  assert.equal(settled.state, 'cancelled')
  assert.equal(statusReads, 3, 'follow-up must wait through running and cancelling states')
} finally {
  globalThis.fetch = originalFetch
}

const abortController = new AbortController()
globalThis.fetch = async () => new Response(JSON.stringify({ state: 'running' }), {
  status: 200,
  headers: { 'Content-Type': 'application/json' },
})
try {
  const pending = waitForWorkspaceSessionTerminal('http://127.0.0.1:8181', 'thread-1', {
    timeoutMs: 20000,
    pollMs: 10000,
    signal: abortController.signal,
  })
  abortController.abort()
  await assert.rejects(pending, (error) => error?.name === 'AbortError')
} finally {
  globalThis.fetch = originalFetch
}

const requestController = new AbortController()
const apiCalls = []
globalThis.fetch = async (url, options = {}) => {
  apiCalls.push({ url: String(url), options })
  assert.equal(options.signal, requestController.signal, 'Workspace lifecycle request must forward its AbortSignal')
  if (options.method === 'DELETE') return new Response(null, { status: 204 })
  const payload = String(url).endsWith('/messages')
    ? { session_id: 'thread-1', turn_index: 1, state: 'waiting_for_events', duplicate: false }
    : { id: 'thread-1', workspace: '/work', model_id: 'model-1', state: 'waiting_for_events' }
  return new Response(JSON.stringify(payload), { status: options.method === 'POST' ? 201 : 200, headers: { 'Content-Type': 'application/json' } })
}
try {
  await createWorkspaceSession('http://127.0.0.1:8181', { workspace: '/work', goal: 'inspect' }, { signal: requestController.signal })
  await sendWorkspaceMessage('http://127.0.0.1:8181', 'thread-1', 'follow up', 'message-1', { signal: requestController.signal })
  await getWorkspaceSession('http://127.0.0.1:8181', 'thread-1', { signal: requestController.signal })
  await cancelWorkspaceSession('http://127.0.0.1:8181', 'thread-1', { signal: requestController.signal })
  assert.equal(apiCalls.length, 4)
  assert.equal(JSON.parse(apiCalls[1].options.body).client_message_id, 'message-1')
} finally {
  globalThis.fetch = originalFetch
}

assert.equal(workspaceFollowUpDisposition({ session_id: 'thread-1', state: 'waiting_for_events', duplicate: false }, 'thread-1'), 'stream')
assert.equal(workspaceFollowUpDisposition({ session_id: 'thread-1', state: 'waiting_for_events', duplicate: true }, 'thread-1'), 'stream')
assert.equal(workspaceFollowUpDisposition({ session_id: 'thread-1', state: 'idle', duplicate: true }, 'thread-1'), 'restore')
assert.equal(workspaceFollowUpDisposition({ session_id: 'thread-1', state: 'running', duplicate: true }, 'thread-1'), 'recover')
assert.throws(
  () => workspaceFollowUpDisposition({ session_id: 'thread-2', state: 'waiting_for_events', duplicate: false }, 'thread-1'),
  /mismatched session identity/,
)

const boundSession = { id: 'thread-1', model_id: 'model-1' }
const readyRuntime = { status: 'online', loaded_now: true, generation_ready: true, active_model_id: 'model-1' }
assert.equal(workspaceSessionMatchesRuntime(boundSession, readyRuntime, true), true)
assert.equal(workspaceSessionMatchesRuntime(boundSession, { ...readyRuntime, active_model_id: 'model-2' }, true), false)
assert.equal(workspaceSessionMatchesRuntime(boundSession, { ...readyRuntime, generation_ready: false }, true), false)
assert.equal(workspaceSessionMatchesRuntime(boundSession, readyRuntime, false), false)

let bounded = { ...WORKSPACE_IDLE_STATE, events: [], turns: [] }
for (let index = 0; index < 300; index += 1) {
  bounded = reduceWorkspaceEvent(bounded, { event: 'session.notice', content: `event-${index}` })
}
assert.equal(bounded.events.length, 240, 'activity history must remain bounded during long sessions')
assert.equal(bounded.events[0].content, 'event-60')
assert.equal(workspaceEndpoint('http://127.0.0.1:8181/', '/abc/events'), 'http://127.0.0.1:8181/api/agent/workspace/sessions/abc/events')
assert.equal(workspaceModelsEndpoint('http://127.0.0.1:8181/'), 'http://127.0.0.1:8181/api/agent/workspace/models')
assert.equal(workspaceBrowseEndpoint('http://127.0.0.1:8181/'), 'http://127.0.0.1:8181/api/agent/workspace/browse')
assert.equal(workspaceBrowseEndpoint('http://127.0.0.1:8181/', 'C:/data'), 'http://127.0.0.1:8181/api/agent/workspace/browse?path=C%3A%2Fdata')
assert.equal(workspaceThreadsEndpoint('http://127.0.0.1:8181/', 'C:/data', 'thread/1'), 'http://127.0.0.1:8181/api/agent/workspace/threads/thread%2F1?workspace=C%3A%2Fdata')
assert.equal(workspaceCompactionEndpoint('http://127.0.0.1:8181/', 'C:/data', 'thread/1'), 'http://127.0.0.1:8181/api/agent/workspace/threads/thread%2F1/compact?workspace=C%3A%2Fdata')

console.log('workspace-agent-smoke: PASS')
