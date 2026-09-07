/* Tool-calling contract lane.

   A tool-capable model can answer a turn by asking to CALL something instead of
   writing prose: the reply comes back with `finish_reason: "tool_calls"`, empty
   content, and a list of requested calls. This module owns whether that may be
   offered, how the request is composed, and — the part that needs care — what the
   result means.

   THREE RULES SHAPE EVERYTHING HERE.

   1. A TOOL CALL IS A REQUEST, NOT AN ACTION. Nothing in this lane executes
      anything. The model asked; that is all that happened. Copy never says a tool
      "ran", and the surface offers no way to run one — a browser that executed
      model-chosen calls against the user's machine would be a different and much
      larger security proposition than showing what was asked.

   2. THE ARGUMENTS ARE MODEL-GENERATED TEXT. `function.arguments` is a JSON
      *string* the model produced. It can be malformed, it can disagree with the
      schema that was offered, and it is not trustworthy input. It is parsed
      defensively and rendered as data.

   3. MODELS LOOP. Verified live: given a tool result, Llama 3.2 3B re-issued the
      identical call rather than using it. A surface that offers a continuation
      without detecting that will spin. `detectRepeatedCall` exists for exactly
      this and is asserted in the smoke.

   Gating is doubly conditional: the ENGINE must advertise `streaming_tool_calls`,
   and the LOADED MODEL must carry a `tool_capable` compatibility row.

   That second condition is STRICTER THAN THE ENGINE'S, on purpose, and the
   distinction matters. `POST /v1/chat/completions` gates on the chat TEMPLATE, not
   on `tool_capable` — it never reads that bit. A model whose row says
   `tool_capable: false` but whose template happens to carry tools is accepted with
   a 200 and returns degraded calls. (Templates with no tools branch DO fail closed,
   with a typed 422 `unsupported_chat_template`, but that is a different gate.)
   `tool_capable` is the receipt saying those calls were actually checked, so this
   lane requires it — and nothing here claims the engine does. */

import { findCompatibilityHint, isSupportedCapabilityStatus, displayCapabilityCopy } from './capabilities.js'

const ROW_ID = 'streaming_tool_calls'

export const DEFAULT_TOOLS = `[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city",
      "parameters": {
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
      }
    }
  }
]`

function findRow(rows, id) {
  return (rows || []).find((row) => row?.id === id) || null
}

/* The engine half of the gate: does this build speak the tool-call protocol? */
export function readToolContract(capabilities) {
  const conformance = findRow(capabilities?.api_conformance, ROW_ID)
  const feature = findRow(capabilities?.api_features, ROW_ID)
  const row = conformance || feature
  if (!row) {
    return { present: false, supported: false, rowId: ROW_ID, status: null, note: null }
  }
  return {
    present: true,
    supported: isSupportedCapabilityStatus(String(row.status || '')),
    rowId: ROW_ID,
    status: row.status || null,
    note: feature?.notes ? displayCapabilityCopy(feature.notes) : null,
  }
}

/* The model half of the gate.

   `tool_capable` is a per-row receipt, not a family trait: a model is tool-capable
   because that exact artifact has a certified tool template, and a neighbouring
   quantization of the same model may not. Resolved by exact row, matching the
   discipline the rest of the compatibility surface uses. */
export function readModelToolCapability(capabilities, model, runtime) {
  if (!runtime?.loaded_now) return { capable: false, reason: 'No model is loaded.', rowId: null }
  /* Resolved through the shared compatibility resolver rather than by matching an
     id string. `runtime.active_model_id` is a DISPLAY name ("Llama 3.2 3B
     Instruct") while a compatibility row is a slug (`llama32_3b_instruct_q8_0`),
     so a naive comparison never matches and every model looks uncertified. This
     is the same call WorkspaceView makes for the same question. */
  const compatibility = findCompatibilityHint(capabilities, model, null)
  const row = compatibility?.target || null
  if (!row || !compatibility?.exact) {
    return { capable: false, reason: 'The loaded model has no exact compatibility row, so no tool receipt applies to it.', rowId: row?.id || null }
  }
  const supported = isSupportedCapabilityStatus(String(row.status || ''))
  if (!row.tool_capable) {
    /* Deliberately STRICTER than the engine. POST /v1/chat/completions gates on
       the chat TEMPLATE, not on this bit — a row with tool_capable:false whose
       template happens to carry tools will be accepted with a 200 and return
       degraded calls. `tool_capable` is the receipt saying the calls were checked;
       offering the control without it would invite exactly that degraded result.
       So this is a product choice, and the copy must not claim the engine
       enforces it. */
    return {
      capable: false,
      reason: 'This model has no tool receipt. The engine may still accept tools for it, but the results are unchecked, so Camelid does not offer them here.',
      rowId: row.id || null,
    }
  }
  if (!supported) {
    return {
      capable: false,
      reason: 'This model claims a tool template but its row is not marked supported, so the claim is not carried here.',
      rowId: row.id || null,
    }
  }
  return { capable: true, reason: null, rowId: row.id || null }
}

/* Validate a tools array. The engine rejects a malformed one, so catching it here
   turns a typed 400 at send time into an explanation at edit time. */
export function parseToolDefinitions(text) {
  const raw = String(text || '').trim()
  if (!raw) return { ok: false, error: 'Define at least one tool.', value: null }
  let value
  try {
    value = JSON.parse(raw)
  } catch (error) {
    return { ok: false, error: `Not valid JSON: ${error.message}`, value: null }
  }
  if (!Array.isArray(value)) return { ok: false, error: 'Tools must be a JSON array.', value: null }
  if (!value.length) return { ok: false, error: 'Define at least one tool.', value: null }
  for (const [index, tool] of value.entries()) {
    if (tool?.type !== 'function') {
      return { ok: false, error: `Tool ${index + 1}: "type" must be "function".`, value: null }
    }
    const name = tool?.function?.name
    if (typeof name !== 'string' || !name.trim()) {
      return { ok: false, error: `Tool ${index + 1}: "function.name" is required.`, value: null }
    }
  }
  return { ok: true, error: null, value }
}

export function toolRequestFields({ enabled, contract, capability, toolsText } = {}) {
  if (!enabled) return {}
  if (!contract?.supported || !capability?.capable) return {}
  const parsed = parseToolDefinitions(toolsText)
  if (!parsed.ok) return {}
  return { tools: parsed.value }
}

export function toolReadiness({ enabled, contract, capability, toolsText } = {}) {
  if (!enabled) return { ready: false, reason: null }
  if (!contract?.present) return { ready: false, reason: 'This engine does not advertise tool calling.' }
  if (!contract.supported) return { ready: false, reason: 'The tool-calling capability row is not marked supported on this engine.' }
  if (!capability?.capable) return { ready: false, reason: capability?.reason || 'The loaded model is not tool-capable.' }
  const parsed = parseToolDefinitions(toolsText)
  if (!parsed.ok) return { ready: false, reason: parsed.error }
  return { ready: true, reason: null }
}

/* Normalize the tool calls on a reply.

   `arguments` is a JSON string the MODEL wrote. Parsing it is best-effort and its
   failure is reported rather than hidden: a malformed argument object is a real
   thing to know about a model, and swallowing the error would present a broken
   call as a clean one. */
export function normalizeToolCalls(toolCalls) {
  const list = Array.isArray(toolCalls) ? toolCalls : []
  if (!list.length) return null
  return list.map((call, index) => {
    const rawArguments = call?.function?.arguments
    let parsedArguments = null
    let parseError = null
    if (typeof rawArguments === 'string' && rawArguments.trim()) {
      try {
        parsedArguments = JSON.parse(rawArguments)
      } catch (error) {
        parseError = error.message
      }
    } else if (rawArguments && typeof rawArguments === 'object') {
      parsedArguments = rawArguments
    }
    return {
      index,
      id: typeof call?.id === 'string' ? call.id : null,
      name: typeof call?.function?.name === 'string' ? call.function.name : null,
      rawArguments: typeof rawArguments === 'string' ? rawArguments : JSON.stringify(rawArguments ?? null),
      parsedArguments,
      parseError,
    }
  })
}

/* A signature that identifies "the same call again".

   Name plus arguments, with the arguments normalized through a parse where
   possible so that whitespace differences do not read as a different call. */
export function toolCallSignature(call) {
  if (!call) return ''
  const args = call.parsedArguments !== null && call.parsedArguments !== undefined
    ? JSON.stringify(call.parsedArguments)
    : String(call.rawArguments || '')
  return `${call.name || ''}::${args}`
}

/* Detect a model that is not consuming tool results.

   Observed live: given "12C, light rain" for get_weather(Paris), Llama 3.2 3B
   asked for get_weather(Paris) again. Without this, a continuation UI loops
   forever and looks like a hang. Returns the repeated signature so the surface can
   name what is repeating rather than saying something vague. */
export function detectRepeatedCall(previousSignatures, calls) {
  const seen = new Set(previousSignatures || [])
  for (const call of calls || []) {
    const signature = toolCallSignature(call)
    if (signature && seen.has(signature)) {
      return { repeated: true, signature, name: call.name || null }
    }
  }
  return { repeated: false, signature: null, name: null }
}

/* Build the tool-result message that continues the conversation.

   The engine accepts the standard shape: an assistant turn carrying the calls,
   then one `role:"tool"` message per call carrying `tool_call_id`. Verified live
   against a running engine. */
export function toolResultMessage(call, content) {
  return { role: 'tool', tool_call_id: call?.id || '', content: String(content ?? '') }
}

export function assistantToolCallMessage(rawToolCalls) {
  return { role: 'assistant', content: '', tool_calls: rawToolCalls }
}

/* Whether the reply text still carries the model's raw tool-call envelope.

   The capability row claims calls are surfaced "without leaking the model's raw
   tool-call envelope", and on the dense lane that holds — content comes back
   empty. The runnable lane (qwen35/Ornith) is different: it lifts the structured
   calls out of the text but KEEPS the text, so the same reply carries both. A
   surface that rendered that content as prose would show the user a wall of
   markup next to a tidy card and look broken. Detected rather than assumed, so a
   lane that stops doing it needs no change here. */
export function replyCarriesRawEnvelope(content) {
  const text = String(content || '')
  if (!text.trim()) return false
  return /<function\s*=|<\/?tool_call>|\[TOOL_CALLS\]|<\|python_tag\|>/.test(text)
}

export function formatArguments(call) {
  if (!call) return ''
  if (call.parsedArguments !== null && call.parsedArguments !== undefined) {
    return JSON.stringify(call.parsedArguments, null, 2)
  }
  return String(call.rawArguments || '')
}
