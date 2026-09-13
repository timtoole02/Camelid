/* Connected tool orchestration. No command, credential, or endpoint comes from
 * model output: a call must match the tool catalog selected for this turn. */
export const MAX_MCP_ROUNDS = 8
export const MAX_MCP_TOOLS = 16

export async function mcpRequest(apiBase, path, { method = 'GET', body, signal } = {}) {
  const response = await fetch(`${String(apiBase || '').replace(/\/$/, '')}/api/mcp${path}`, {
    method, signal,
    headers: { 'Content-Type': 'application/json', 'X-Camelid-Mcp': '1' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  const data = await response.json().catch(() => null)
  if (!response.ok) throw new Error(data?.error?.message || `MCP request failed (${response.status}).`)
  return data
}
export function selectedMcpTools(connections, keys) {
  const selected = new Set(keys || [])
  return (connections || []).filter(c => c.connected).flatMap(c => (c.tools || [])
    .filter(t => selected.has(t.key))
    .map(t => ({ ...t, connection_id: c.config.id, connection_name: c.config.name }))).slice(0, MAX_MCP_TOOLS)
}
export function mcpToolDefinitions(tools) {
  return tools.map(t => ({ type: 'function', function: { name: t.key, description: `${t.connection_name}: ${t.name}. ${t.description || ''}`, parameters: t.input_schema } }))
}
export function toolHistoryMessage(message) {
  const payload = { role: message.role, content: message.content || '' }
  if (message.role === 'assistant' && message.tool_calls?.length) payload.tool_calls = message.tool_calls
  if (message.role === 'tool') payload.tool_call_id = message.tool_call_id
  return payload
}
// An interrupted or imported transcript must never send an orphan result, nor
// advertise an unfulfilled call as if it had executed.
export function completeToolHistory(messages) {
  const out = []
  for (let i = 0; i < messages.length; i += 1) {
    const m = messages[i]
    if (m.role === 'tool') continue
    if (!m.tool_calls?.length) { out.push(m); continue }
    const results = []
    for (let j = i + 1; j < messages.length && messages[j].role === 'tool'; j += 1) results.push(messages[j])
    const ids = new Set(m.tool_calls.map(c => c.id))
    if (ids.size === m.tool_calls.length && results.length === ids.size && results.every(r => ids.delete(r.tool_call_id))) {
      out.push(m, ...results)
      i += results.length
    } else if (m.content?.trim()) {
      const { tool_calls, ...plain } = m
      out.push(plain)
    }
  }
  return out
}
export function mcpResultText(call) {
  if (call.status === 'denied') return 'The user denied this tool call. Do not retry it or use another tool to perform the same action.'
  if (!call.result) return `Tool ${call.status}. No successful result was received. Do not assume the action completed.`
  return JSON.stringify(call.result)
}
const wait = (ms, signal) => new Promise((resolve, reject) => {
  if (signal.aborted) { reject(new DOMException('Stopped', 'AbortError')); return }
  const abort = () => { clearTimeout(timer); reject(new DOMException('Stopped', 'AbortError')) }
  const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve() }, ms)
  signal.addEventListener('abort', abort, { once: true })
})

export async function runMcpTurn({ initialOptions, tools, send, request, approve, recordResults, activity, signal, pause = wait }) {
  let options = { ...initialOptions, mcpSignal: signal, connectedTools: mcpToolDefinitions(tools) }
  const allowed = new Map(tools.map(t => [t.key, t]))
  const seen = new Set()
  for (let round = 0; round <= MAX_MCP_ROUNDS; round += 1) {
    signal.throwIfAborted()
    activity({ phase: 'generating', round })
    const generated = await send(options)
    if (!generated) return
    const calls = generated.message.tool_calls || []
    if (!calls.length) return
    if (round === MAX_MCP_ROUNDS) throw new Error('Reached the limit of 8 tool rounds. Review the results and send a follow-up to continue.')
    if (calls.length > MAX_MCP_TOOLS) throw new Error('The model requested too many tools in one turn.')
    if (calls.some(c => typeof c.id !== 'string' || !c.id) || new Set(calls.map(c => c.id)).size !== calls.length) throw new Error('The model returned missing or duplicate tool call IDs. Nothing was executed.')
    const results = []
    let nextHistory
    try {
      for (const call of calls) {
        signal.throwIfAborted()
        const tool = allowed.get(call.function?.name)
        if (!tool) throw new Error('The model requested a tool that was not selected for this turn.')
        let argumentsObject
        try { argumentsObject = JSON.parse(call.function.arguments) } catch { throw new Error('The model returned invalid tool arguments. Nothing was executed.') }
        if (!argumentsObject || typeof argumentsObject !== 'object' || Array.isArray(argumentsObject)) throw new Error('Tool arguments must be a JSON object.')
        const signature = JSON.stringify([tool.key, argumentsObject])
        if (seen.has(signature)) throw new Error('Stopped a repeated tool call with identical arguments. Review the result before asking to retry.')
        seen.add(signature)
        const pending = await request('/calls', { method: 'POST', body: { tool_key: tool.key, arguments: argumentsObject }, signal })
        let receipt
        try {
          signal.throwIfAborted()
          activity({ phase: 'approval', tool: tool.name, connection: tool.connection_name, round })
          const approved = await approve(pending, signal)
          signal.throwIfAborted()
          receipt = await request(`/calls/${pending.id}/decision`, { method: 'POST', body: { approved }, signal })
          while (receipt.status === 'running') {
            activity({ phase: 'executing', tool: tool.name, connection: tool.connection_name, round })
            await pause(400, signal)
            receipt = await request(`/calls/${pending.id}`, { signal })
          }
          results.push({ role: 'tool', tool_call_id: call.id, content: mcpResultText(receipt), mcp: { tool: tool.name, connection: tool.connection_name, status: receipt.status, is_error: Boolean(receipt.result?.isError) } })
          if (!['complete', 'denied'].includes(receipt.status)) throw new Error(`Tool ${receipt.status}. Review its outcome before retrying.`)
        } finally {
          if (!receipt || receipt.status === 'running') await request(`/calls/${pending.id}`, { method: 'DELETE' }).catch(() => {})
        }
      }
    } finally {
      // Complete every call/result group, even on Stop, so later requests never
      // contain a dangling call or silently execute pending work on reload.
      const complete = calls.map(call => results.find(r => r.tool_call_id === call.id) || {
        role: 'tool', tool_call_id: call.id,
        content: 'Tool execution was stopped or unavailable. No successful result was received. Do not assume completion or retry without asking the user.',
        mcp: { tool: call.function?.name || 'Tool', status: 'interrupted', is_error: true },
      })
      recordResults(generated.conversationId, complete)
      nextHistory = [...(generated.history || []), ...complete]
    }
    options = { mcpSignal: signal, connectedTools: mcpToolDefinitions(tools), mcpConversationId: generated.conversationId, mcpHistory: nextHistory, overrideContent: 'Continue using the tool results.' }
  }
}
