import assert from 'node:assert/strict'
import { reduceCodeEvent, reduceWorkspaceEvent, waitForWorkspaceSessionTerminal, WORKSPACE_IDLE_STATE, workspaceEndpoint, workspaceModelsEndpoint, workspaceBrowseEndpoint, workspaceThreadsEndpoint, workspaceCompactionEndpoint } from '../src/lib/workspaceAgent.js'

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
assert.equal(impossibleApproval.phase, 'error')
assert.match(impossibleApproval.error, /unexpected approval request/)

let codeState = reduceCodeEvent(
  { ...WORKSPACE_IDLE_STATE, events: [] },
  { event: 'approval.required', approval_id: 'approval-1', tool: 'write_file', risk: 'write', detail: 'write_file → src/a.rs' },
)
assert.equal(codeState.phase, 'awaiting_approval')
assert.equal(codeState.approval.approval_id, 'approval-1')
codeState = reduceCodeEvent(codeState, { event: 'approval.resolved' })
assert.equal(codeState.phase, 'running')
assert.equal(codeState.approval, null)

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
globalThis.fetch = async () => new Response(JSON.stringify({ state: terminalStates[statusReads++] }), {
  status: 200,
  headers: { 'Content-Type': 'application/json' },
})
try {
  const settled = await waitForWorkspaceSessionTerminal('http://127.0.0.1:8181', 'thread-1', { timeoutMs: 1000, pollMs: 0 })
  assert.equal(settled.state, 'cancelled')
  assert.equal(statusReads, 3, 'follow-up must wait through running and cancelling states')
} finally {
  globalThis.fetch = originalFetch
}

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
assert.equal(workspaceThreadsEndpoint('http://127.0.0.1:8181/', 'C:/data', '', 'code'), 'http://127.0.0.1:8181/api/agent/workspace/threads?workspace=C%3A%2Fdata&mode=code')
assert.equal(workspaceCompactionEndpoint('http://127.0.0.1:8181/', 'C:/data', 'thread/1'), 'http://127.0.0.1:8181/api/agent/workspace/threads/thread%2F1/compact?workspace=C%3A%2Fdata')

console.log('workspace-agent-smoke: PASS')
