/* Structured-output contract lane.

   The engine can constrain decoding to a JSON schema or a Lark grammar, masking
   the token distribution at every step so the reply cannot leave the grammar.
   This module owns every decision about that: whether the contract permits it,
   which of the mutually-exclusive request forms to send, and — the hard part —
   what may honestly be claimed about a reply that came back.

   THE CENTRAL PROBLEM. A constrained response is byte-for-byte the same SHAPE as
   an unconstrained one. There is no field, flag, header or finish_reason
   asserting that a constraint was compiled or applied. So a 200 proves the
   request was accepted, and nothing more. Any surface that renders "constrained"
   on the strength of a 200 is asserting something it cannot see.

   What CAN be shown, in descending order of strength:

     1. DIVERTED TOKENS. The mask is applied to a clone of the logits, and the
        raw distribution is what gets returned. So with `logprobs:true` on a
        greedy turn, a position whose emitted token is NOT the argmax of its own
        top-N is direct evidence the mask moved the decode — the model wanted
        something else and was not allowed it. This is observed, not asserted.
     2. SCHEMA CONFORMANCE. The reply parses, and validates against the schema
        the user submitted. Necessary but not sufficient: an unconstrained model
        can emit valid JSON by luck.
     3. THE NEGATIVE-SPACE GUARANTEE. Every path that could silently drop a
        constraint is a typed refusal before generation — an ambiguous request
        form, an uncompilable schema, a lane that cannot enforce, and streaming.
        So a 200 means no such path was taken. Worth stating; not proof of a mask.

   Copy in this lane says which of those it is standing on, and never rounds 2 up
   to 1. */

import { isSupportedCapabilityStatus, displayCapabilityCopy } from './capabilities.js'

const ROW_ID = 'llguidance_structured_outputs'

export const STRUCTURED_MODES = {
  OFF: 'off',
  JSON_OBJECT: 'json_object',
  JSON_SCHEMA: 'json_schema',
  GRAMMAR: 'grammar',
}

/* A schema that exercises the interesting cases — required keys, an enum, a
   bounded number, a nested array — so the first run shows the mask doing real
   work rather than agreeing with the model. */
export const DEFAULT_SCHEMA = `{
  "type": "object",
  "properties": {
    "title": { "type": "string" },
    "sentiment": { "type": "string", "enum": ["positive", "neutral", "negative"] },
    "score": { "type": "integer" },
    "tags": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["title", "sentiment", "score", "tags"]
}`

export const DEFAULT_GRAMMAR = `start: answer
answer: "YES" | "NO" | "MAYBE"`

function findRow(rows, id) {
  return (rows || []).find((row) => row?.id === id) || null
}

/* Resolve the contract by EXACT row id.

   `api_conformance` is preferred for two reasons: it carries the machine-readable
   mode lists, and its projection ships no `notes` string — so it cannot smuggle a
   vendor name into rendered copy. `api_features` is the fallback for an engine
   that predates the conformance array; there the modes are unknown, and unknown
   fails closed. */
export function readStructuredOutputContract(capabilities) {
  const conformance = findRow(capabilities?.api_conformance, ROW_ID)
  const feature = findRow(capabilities?.api_features, ROW_ID)
  const row = conformance || feature
  if (!row) {
    return {
      present: false,
      supported: false,
      modesKnown: false,
      nonStreamingSupported: false,
      rowId: ROW_ID,
      status: null,
      note: null,
    }
  }
  const supported = isSupportedCapabilityStatus(String(row.status || ''))
  const modesKnown = Array.isArray(conformance?.supported_modes)
  const supportedModes = modesKnown ? conformance.supported_modes : []
  return {
    present: true,
    supported,
    modesKnown,
    nonStreamingSupported: supported && modesKnown && supportedModes.includes('chat_nonstreaming'),
    rowId: ROW_ID,
    status: row.status || null,
    note: feature?.notes ? displayCapabilityCopy(feature.notes) : null,
  }
}

export function parseSchemaText(schemaText) {
  const text = String(schemaText || '').trim()
  if (!text) return { ok: false, error: 'Enter a JSON schema.', value: null }
  let value
  try {
    value = JSON.parse(text)
  } catch (error) {
    return { ok: false, error: `Not valid JSON: ${error.message}`, value: null }
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return { ok: false, error: 'A JSON schema must be an object.', value: null }
  }
  return { ok: true, error: null, value }
}

/* Build the request fields for exactly ONE constraint form.

   The engine refuses an ambiguous request — two constraint forms in one body is a
   typed 400 before generation — so this returns at most one key and never
   composes them. It returns {} whenever the mode is off or the contract does not
   permit constrained decoding, so a caller can spread it unconditionally. */
export function structuredOutputRequestFields({ enabled, mode, contract, schemaText, grammarText } = {}) {
  if (!enabled || mode === STRUCTURED_MODES.OFF) return {}
  if (!contract?.nonStreamingSupported) return {}
  if (mode === STRUCTURED_MODES.JSON_OBJECT) {
    return { response_format: { type: 'json_object' } }
  }
  if (mode === STRUCTURED_MODES.JSON_SCHEMA) {
    const parsed = parseSchemaText(schemaText)
    if (!parsed.ok) return {}
    /* The nested `json_schema.schema` envelope is the chat shape. The top-level
       llama.cpp `json_schema` field takes a RAW schema, and the Responses route
       uses `text.format.schema` — three different locations for the same object,
       and sending the wrong one is a 400 with no hint. */
    return { response_format: { type: 'json_schema', json_schema: { schema: parsed.value } } }
  }
  if (mode === STRUCTURED_MODES.GRAMMAR) {
    const grammar = String(grammarText || '').trim()
    if (!grammar) return {}
    return { response_format: { type: 'grammar', grammar } }
  }
  return {}
}

/* A constraint and stream:true is a hard 400, and the streaming decode job never
   builds a grammar state at all — that route-level refusal is the ONLY thing
   standing between a streamed constrained request and silently unconstrained
   output. Derived from one predicate so the stream flag and the constraint fields
   cannot drift apart. */
export function structuredOutputForcesNonStreaming({ enabled, mode, contract } = {}) {
  return Boolean(enabled && mode !== STRUCTURED_MODES.OFF && contract?.nonStreamingSupported)
}

/* Whether the composed request is actually sendable, and if not, why — so the
   composer can explain rather than silently sending an unconstrained turn. */
export function structuredOutputReadiness({ enabled, mode, contract, schemaText, grammarText } = {}) {
  if (!enabled || mode === STRUCTURED_MODES.OFF) return { ready: false, reason: null }
  if (!contract?.present) {
    return { ready: false, reason: 'This engine does not advertise constrained decoding.' }
  }
  if (!contract.nonStreamingSupported) {
    return { ready: false, reason: 'This engine advertises constrained decoding but does not describe a supported non-streaming mode for it.' }
  }
  if (mode === STRUCTURED_MODES.JSON_SCHEMA) {
    const parsed = parseSchemaText(schemaText)
    if (!parsed.ok) return { ready: false, reason: parsed.error }
  }
  if (mode === STRUCTURED_MODES.GRAMMAR && !String(grammarText || '').trim()) {
    return { ready: false, reason: 'Enter a grammar.' }
  }
  return { ready: true, reason: null }
}

/* Minimal structural validation of a reply against the submitted schema.

   Deliberately NOT a JSON Schema implementation — it checks the handful of
   keywords the default schema uses and reports everything else as unchecked. A
   validator that silently ignored the keywords it does not implement would report
   "valid" for a document it never examined, which is the same class of lie this
   whole surface exists to avoid. */
function validateAgainstSchema(value, schema, path = '$') {
  const problems = []
  const unchecked = []
  const type = schema?.type
  const typeOf = (v) => (Array.isArray(v) ? 'array' : v === null ? 'null' : typeof v)
  if (type === 'object') {
    if (typeOf(value) !== 'object') return { problems: [`${path} should be an object, got ${typeOf(value)}`], unchecked }
    for (const key of schema.required || []) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) problems.push(`${path}.${key} is required but missing`)
    }
    for (const [key, sub] of Object.entries(schema.properties || {})) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) continue
      const nested = validateAgainstSchema(value[key], sub, `${path}.${key}`)
      problems.push(...nested.problems)
      unchecked.push(...nested.unchecked)
    }
    return { problems, unchecked }
  }
  if (type === 'array') {
    if (!Array.isArray(value)) return { problems: [`${path} should be an array, got ${typeOf(value)}`], unchecked }
    value.forEach((item, index) => {
      if (!schema.items) return
      const nested = validateAgainstSchema(item, schema.items, `${path}[${index}]`)
      problems.push(...nested.problems)
      unchecked.push(...nested.unchecked)
    })
    return { problems, unchecked }
  }
  if (Array.isArray(schema?.enum)) {
    if (!schema.enum.includes(value)) problems.push(`${path} should be one of ${schema.enum.join(', ')}, got ${JSON.stringify(value)}`)
    return { problems, unchecked }
  }
  if (type === 'string' && typeOf(value) !== 'string') problems.push(`${path} should be a string, got ${typeOf(value)}`)
  else if (type === 'integer' && !Number.isInteger(value)) problems.push(`${path} should be an integer, got ${JSON.stringify(value)}`)
  else if (type === 'number' && typeof value !== 'number') problems.push(`${path} should be a number, got ${typeOf(value)}`)
  else if (type === 'boolean' && typeof value !== 'boolean') problems.push(`${path} should be a boolean, got ${typeOf(value)}`)
  else if (type === undefined) unchecked.push(path)
  return { problems, unchecked }
}

/* Assess a reply. `divertedPositions` comes from the token record when one was
   captured: the count of positions whose emitted token was not the highest-scoring
   one. Under a greedy turn that is the mask's fingerprint. */
export function assessStructuredReply({ content, mode, schemaText, divertedPositions = null, greedy = true }) {
  const text = String(content ?? '')
  const wantsJson = mode === STRUCTURED_MODES.JSON_OBJECT || mode === STRUCTURED_MODES.JSON_SCHEMA
  const result = {
    mode,
    parsed: null,
    parses: false,
    parseError: null,
    problems: [],
    unchecked: [],
    schemaChecked: false,
    diverted: divertedPositions,
    /* The honest headline. `enforced` is never true from a 200 alone — only a
       diverted position observed on a greedy turn earns it. */
    evidence: 'accepted',
  }
  if (wantsJson) {
    try {
      result.parsed = JSON.parse(text)
      result.parses = true
    } catch (error) {
      result.parseError = error.message
    }
    if (result.parses && mode === STRUCTURED_MODES.JSON_SCHEMA) {
      const schema = parseSchemaText(schemaText)
      if (schema.ok) {
        const checked = validateAgainstSchema(result.parsed, schema.value)
        result.problems = checked.problems
        result.unchecked = checked.unchecked
        result.schemaChecked = true
      }
    }
  }
  if (greedy && Number.isFinite(Number(divertedPositions)) && Number(divertedPositions) > 0) {
    result.evidence = 'diverted'
  } else if (result.parses && result.problems.length === 0 && result.schemaChecked) {
    result.evidence = 'conforms'
  } else if (result.parses) {
    result.evidence = 'parses'
  }
  return result
}

export const EVIDENCE_COPY = {
  diverted: {
    label: 'Mask observed',
    detail: 'At one or more positions the emitted token was not the highest-scoring one. Because the returned scores are the model’s unmasked distribution, that is direct evidence the constraint moved this decode.',
  },
  conforms: {
    label: 'Matches the schema',
    detail: 'The reply parsed and satisfied every keyword this page checks. That is consistent with the constraint being applied, but a model can also produce conforming output on its own — it is not proof the mask acted.',
  },
  parses: {
    label: 'Parsed',
    detail: 'The reply is valid JSON. Nothing here shows whether the constraint shaped it.',
  },
  accepted: {
    label: 'Request accepted',
    detail: 'The engine accepted the constraint and returned a reply. It reports no field saying a mask was applied, so this page cannot claim more than that from the response alone.',
  },
}
