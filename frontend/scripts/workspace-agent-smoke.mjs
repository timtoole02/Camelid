import assert from 'node:assert/strict'
import {
  contextLimitingFactorLabel,
  contextWindowModeLabel,
  formatContextTokens,
  normalizeContextWindow,
  reduceCodeEvent,
  reduceWorkspaceEvent,
  waitForWorkspaceSessionTerminal,
  WORKSPACE_IDLE_STATE,
  workspaceEndpoint,
  workspaceModelsEndpoint,
  workspaceBrowseEndpoint,
  workspaceThreadsEndpoint,
  workspaceCompactionEndpoint,
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
codeState = reduceCodeEvent(codeState, { event: 'tool.call', detail: 'write_file(a.txt, 4 bytes)' })
assert.equal(codeState.latestTool.detail, 'write_file(a.txt, 4 bytes)')
codeState = reduceCodeEvent(codeState, { event: 'tool.result', tool: 'write_file', outcome: 'ok', content: 'created a.txt' })
assert.equal(codeState.latestTool, null, 'a completed tool must not remain marked active')
assert.equal(codeState.latestResult.tool, 'write_file')

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

let timed = reduceCodeEvent(
  { ...WORKSPACE_IDLE_STATE, events: [], turns: [] },
  { event: 'session.starting', task: 'Measure the local model' },
)
timed = reduceCodeEvent(timed, {
  event: 'model.timing',
  total_ms: 2480,
  ttft_ms: 190,
  output_tokens: 92,
  prefill_ms: 120,
  server_first_content_ms: 165,
  decode_ms: 2200,
  prompt_cache_hit: true,
  reused_tokens: 2884,
  prefilled_tokens: 302,
  prompt_cache_decision: 'block_prefix_hit',
  common_prefix_tokens: 2884,
  divergent_suffix_tokens: 302,
  candidate_tokens: 3000,
  cache_block_tokens: 64,
  matched_cache_blocks: 45,
})
assert.deepEqual(timed.modelSteps[0], {
  index: 1,
  totalMs: 2480,
  ttftMs: 190,
  outputTokens: 92,
  prefillMs: 120,
  serverFirstContentMs: 165,
  decodeMs: 2200,
  promptCacheHit: true,
  reusedTokens: 2884,
  prefilledTokens: 302,
  promptCacheDecision: 'block_prefix_hit',
  commonPrefixTokens: 2884,
  divergentSuffixTokens: 302,
  candidateTokens: 3000,
  cacheBlockTokens: 64,
  matchedCacheBlocks: 45,
})
assert.equal(timed.liveActivity.server_first_content_ms, 165)
assert.equal(timed.liveActivity.prompt_cache_hit, true)
assert.equal(timed.liveActivity.prompt_cache_decision, 'block_prefix_hit')
assert.equal(timed.liveActivity.common_prefix_tokens, 2884)
assert.match(timed.liveActivity.detail, /190ms TTFT/)
assert.match(timed.liveActivity.detail, /165ms server first content/)
assert.match(timed.liveActivity.detail, /prompt-cache hit/)
assert.match(timed.liveActivity.detail, /2,884 prompt tokens reused/)

// The activity poll exposes the durable core metrics but not every resident
// engine diagnostic. It must not erase the richer SSE timing between frames.
timed = reduceCodeEvent(timed, {
  event: 'activity.snapshot',
  activity: {
    phase: 'running',
    stage: 'tool',
    detail: 'Running a tool',
    updated_at_ms: (timed.liveActivity.updated_at_ms || 0) + 1,
    total_model_ms: 2480,
    ttft_ms: 190,
    prefill_ms: 120,
    prompt_cache_hit: true,
    agents: [],
  },
})
assert.equal(timed.liveActivity.decode_ms, 2200)
assert.equal(timed.liveActivity.prompt_cache_decision, 'block_prefix_hit')
assert.equal(timed.liveActivity.divergent_suffix_tokens, 302)

const adaptiveContext = normalizeContextWindow({
  mode: 'auto',
  effective_tokens: 16_384,
  recommended_max_tokens: 12_288,
  memory_safe_max_tokens: 12_288,
  model_max_tokens: 40_960,
  validated_max_tokens: 8_192,
  paged_target_tokens: 16_384,
  paged_working_set_tokens: 8_000,
  kv_owner_slots: 3,
  limiting_factor: 'paged_model_target',
})
assert.equal(contextWindowModeLabel(adaptiveContext), 'Auto')
assert.equal(formatContextTokens(adaptiveContext.effectiveTokens), '16K')
assert.equal(formatContextTokens(adaptiveContext.pagedWorkingSetTokens), '7.8K')
assert.equal(formatContextTokens(adaptiveContext.modelMaxTokens), '40K')
assert.equal(formatContextTokens(adaptiveContext.validatedMaxTokens), '8K')
assert.equal(adaptiveContext.kvOwnerSlots, 3)
assert.equal(formatContextTokens(5500), '5.4K')
assert.equal(normalizeContextWindow({ effective_tokens: 8192, memory_safe_max_tokens: 0 }).memorySafeMaxTokens, 0)
assert.equal(formatContextTokens(0), '0')
assert.equal(contextLimitingFactorLabel(adaptiveContext.limitingFactor), 'Qwen 4B paged target')
assert.deepEqual(normalizeContextWindow(null, 8192), {
  mode: 'auto',
  effectiveTokens: 8192,
  recommendedMaxTokens: null,
  memorySafeMaxTokens: null,
  modelMaxTokens: null,
  validatedMaxTokens: null,
  kvOwnerSlots: null,
  availableMemoryBytes: null,
  kvBytesPerToken: null,
  residentCapacityTokens: null,
  configuredMaxTokens: null,
  pagedTargetTokens: null,
  pagedWorkingSetTokens: null,
  limitingFactor: null,
})
assert.equal(normalizeContextWindow({ effective_tokens: 'not-a-number' }, 0), null)

assert.equal(workspaceEndpoint('http://127.0.0.1:8181/', '/abc/events'), 'http://127.0.0.1:8181/api/agent/workspace/sessions/abc/events')
assert.equal(workspaceModelsEndpoint('http://127.0.0.1:8181/'), 'http://127.0.0.1:8181/api/agent/workspace/models')
assert.equal(workspaceBrowseEndpoint('http://127.0.0.1:8181/'), 'http://127.0.0.1:8181/api/agent/workspace/browse')
assert.equal(workspaceBrowseEndpoint('http://127.0.0.1:8181/', 'C:/data'), 'http://127.0.0.1:8181/api/agent/workspace/browse?path=C%3A%2Fdata')
assert.equal(workspaceThreadsEndpoint('http://127.0.0.1:8181/', 'C:/data', 'thread/1'), 'http://127.0.0.1:8181/api/agent/workspace/threads/thread%2F1?workspace=C%3A%2Fdata')
assert.equal(workspaceThreadsEndpoint('http://127.0.0.1:8181/', 'C:/data', '', 'code'), 'http://127.0.0.1:8181/api/agent/workspace/threads?workspace=C%3A%2Fdata&mode=code')
assert.equal(workspaceCompactionEndpoint('http://127.0.0.1:8181/', 'C:/data', 'thread/1'), 'http://127.0.0.1:8181/api/agent/workspace/threads/thread%2F1/compact?workspace=C%3A%2Fdata')

console.log('workspace-agent-smoke: PASS')
