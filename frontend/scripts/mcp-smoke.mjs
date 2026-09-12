import assert from 'node:assert/strict'
import { runMcpTurn, completeToolHistory, selectedMcpTools, mcpToolDefinitions } from '../src/lib/mcp.js'
import { compactForSend } from '../src/lib/conversationCompaction.js'

const tool = { key: 'mcp_echo', name: 'echo', connection_name: 'Fixture', input_schema: { type: 'object' } }
const toolCall = (text = 'hello', id = 'call_1') => ({ id, type: 'function', function: { name: tool.key, arguments: JSON.stringify({ text }) } })
async function scenario({ approve = true, calls = [toolCall()], cancel = false, repeat = false } = {}) {
  const controller = new AbortController()
  const requests = [], sent = [], stored = [], phases = []
  let runs = 0
  const request = async (path, options = {}) => {
    requests.push([path, options.method, options.body])
    if (path === '/calls') return { id: 'receipt', connection_name: 'Fixture', tool: 'echo', arguments: options.body.arguments }
    if (path.endsWith('/decision')) {
      if (options.body.approved) runs += 1
      return { id: 'receipt', status: options.body.approved ? 'complete' : 'denied', ...(options.body.approved ? { result: { content: [{ type: 'text', text: 'hello' }] } } : {}) }
    }
    return { status: 'cancelled' }
  }
  let error
  try {
    await runMcpTurn({ initialOptions: { overrideContent: 'Use echo' }, tools: [tool],
      send: async options => { sent.push(options); return { conversationId: 'thread', message: { role: 'assistant', tool_calls: (repeat || sent.length === 1) ? calls : [] } } },
      request, approve: async () => { if (cancel) controller.abort(); return approve },
      recordResults: (id, results) => stored.push({ id, results }), activity: state => phases.push(state.phase), signal: controller.signal,
    })
  } catch (e) { error = e }
  return { error, requests, sent, stored, phases, runs }
}
const allowed = await scenario()
assert.equal(allowed.error, undefined)
assert.equal(allowed.runs, 1)
assert.equal(allowed.sent.length, 2)
assert.equal(allowed.sent[1].mcpConversationId, 'thread')
assert.ok(allowed.sent[0].mcpSignal instanceof AbortSignal)
assert.equal(allowed.sent[0].mcpSignal, allowed.sent[1].mcpSignal)
assert.equal(allowed.stored[0].results[0].tool_call_id, 'call_1')
assert.deepEqual(allowed.sent[0].connectedTools, mcpToolDefinitions([tool]))
const denied = await scenario({ approve: false })
assert.equal(denied.runs, 0)
assert.equal(denied.sent.length, 2)
assert.match(denied.stored[0].results[0].content, /denied/)
const stopped = await scenario({ cancel: true })
assert.equal(stopped.runs, 0)
assert.equal(stopped.error.name, 'AbortError')
assert.ok(stopped.requests.some(r => r[1] === 'DELETE'))
assert.equal(stopped.stored[0].results[0].mcp.status, 'interrupted')
const repeated = await scenario({ repeat: true })
assert.equal(repeated.runs, 1)
assert.match(repeated.error.message, /repeated tool call/)
const foreign = await scenario({ calls: [{ ...toolCall(), function: { name: 'not_selected', arguments: '{}' } }] })
assert.equal(foreign.runs, 0)
assert.match(foreign.error.message, /not selected/)
const malformed = await scenario({ calls: [{ ...toolCall(), function: { name: tool.key, arguments: '{' } }] })
assert.equal(malformed.runs, 0)
assert.match(malformed.error.message, /invalid tool arguments/)
const many = await scenario({ calls: Array.from({ length: 17 }, (_, i) => toolCall('hi', `call_${i}`)) })
assert.equal(many.runs, 0)
assert.match(many.error.message, /too many/)
for (const calls of [[toolCall(), toolCall('different')], [{ ...toolCall(), id: '' }]]) {
  const invalidIds = await scenario({ calls })
  assert.equal(invalidIds.runs, 0)
  assert.match(invalidIds.error.message, /missing or duplicate/)
}
assert.equal(selectedMcpTools([{ connected: false, config: { id: 'a' }, tools: [tool] }], [tool.key]).length, 0)
const history = [{ role: 'user', content: 'Question' }, { role: 'assistant', content: '', tool_calls: [toolCall()] }, { role: 'tool', tool_call_id: 'call_1', content: 'result' }, { role: 'assistant', content: 'Answer' }]
assert.deepEqual(completeToolHistory(history), history)
assert.deepEqual(completeToolHistory(history.filter(m => m.role !== 'tool')), [history[0], history[3]])
assert.deepEqual(completeToolHistory([history[2], history[3]]), [history[3]])
const compacted = compactForSend([{ role: 'user', content: 'old' }, { role: 'assistant', content: 'old answer' }, ...history], { keepRecent: 2 })
assert.equal(compacted.messages.filter(m => m.tool_calls).length, 1)
assert.equal(compacted.messages.filter(m => m.role === 'tool').length, 1)
assert.deepEqual(completeToolHistory(compacted.messages), compacted.messages)
console.log('MCP smoke passed: approval, denial, cancellation, repeated/foreign/malformed calls, tool history, and compaction.')
