#!/usr/bin/env node
/* Memory (lib/memory.js) and how memories reach a request (projectContext.js).
   The browser smoke covers the UI; this covers the rules. */
import assert from 'node:assert/strict'

import {
  AUTO_SUGGEST_MAX_FIRST_TOKEN_MS,
  AUTO_SUGGEST_MAX_SUGGESTION_MS,
  MAX_EXTRACTION_INPUT_CHARS,
  MAX_MEMORIES,
  MAX_MEMORY_CHARS,
  MAX_PROMPT_MEMORIES,
  MAX_PROMPT_MEMORY_CHARS,
  MAX_SUGGESTIONS,
  SUGGESTION_SCHEMA,
  cleanMemoryText,
  freshSuggestions,
  memoriesForPrompt,
  memoryContextSource,
  memoryKey,
  newMemory,
  normalizeMemories,
  normalizeSuggestionCosts,
  normalizeSuggestions,
  parseSuggestionReply,
  suggestionMode,
  suggestionRequest,
  userTurnNumber,
} from '../src/lib/memory.js'
import { buildContextSources, normalizeChatContext } from '../src/lib/projectContext.js'

/* ---- text ---------------------------------------------------------------- */
assert.equal(cleanMemoryText('  The user\n  is   vegetarian. '), 'The user is vegetarian.', 'one line, single spaces')
assert.equal(cleanMemoryText('x'.repeat(MAX_MEMORY_CHARS + 50)).length, MAX_MEMORY_CHARS, 'capped')
assert.equal(cleanMemoryText(null), '')
assert.equal(memoryKey('The user is Vegetarian!'), memoryKey('the user is vegetarian'), 'case and punctuation do not make a new fact')
assert.notEqual(memoryKey('The user is vegetarian.'), memoryKey('The user is vegan.'))

/* ---- stored memories ----------------------------------------------------- */
const stored = normalizeMemories([
  { id: 'a', text: '  Likes tea ', source: { kind: 'chat', conversation_id: 'c1', message_id: 'm1', turn: 2, conversation_title: 'Morning' } },
  { id: 'a', text: 'duplicate id' },
  { id: 'b', text: '   ' },
  { id: 'c', text: 'Off one', enabled: false, source: { kind: 'bogus' } },
  { id: 'd', text: 'From a note', source: { kind: 'note', note_title: 'Trip' } },
  'not an object',
])
assert.deepEqual(stored.map((memory) => memory.id), ['a', 'c', 'd'], 'empty, repeated and malformed records are dropped')
assert.equal(stored[0].text, 'Likes tea')
assert.equal(stored[0].enabled, true, 'in use unless turned off')
assert.deepEqual(stored[0].source, { kind: 'chat', conversation_id: 'c1', message_id: 'm1', turn: 2, conversation_title: 'Morning' })
assert.equal(stored[1].enabled, false)
assert.deepEqual(stored[1].source, { kind: 'manual' }, 'an unknown source kind reads as written by the user')
assert.deepEqual(stored[2].source, { kind: 'note', note_title: 'Trip' })
assert.equal(normalizeMemories(Array.from({ length: MAX_MEMORIES + 5 }, (_, i) => ({ id: `m${i}`, text: `fact ${i}` }))).length, MAX_MEMORIES)
assert.deepEqual(normalizeMemories('nope'), [])
const made = newMemory('  Works nights ', { kind: 'manual' }, '2026-10-08T10:00:00.000Z')
assert.equal(made.text, 'Works nights')
assert.equal(made.created_at, made.updated_at)
assert.equal(newMemory('   ', { kind: 'manual' }), null, 'nothing to remember is not a memory')

/* ---- what one request carries -------------------------------------------- */
const dated = (id, text, updated, enabled = true) => ({ id, text, enabled, updated_at: updated, created_at: updated, source: { kind: 'manual' } })
const chosen = memoriesForPrompt([
  dated('old', 'Old fact', '2026-01-01T00:00:00Z'),
  dated('new', 'New fact', '2026-03-01T00:00:00Z'),
  dated('off', 'Not in use', '2026-04-01T00:00:00Z', false),
])
assert.deepEqual(chosen.map((memory) => memory.id), ['new', 'old'], 'in use only, most recently updated first')
const many = Array.from({ length: MAX_PROMPT_MEMORIES + 10 }, (_, i) => dated(`k${i}`, `fact ${i}`, `2026-01-01T00:00:${String(i % 60).padStart(2, '0')}Z`))
assert.equal(memoriesForPrompt(many).length, MAX_PROMPT_MEMORIES, 'at most MAX_PROMPT_MEMORIES')
const long = Array.from({ length: 40 }, (_, i) => dated(`l${i}`, `${i} `.padEnd(MAX_MEMORY_CHARS, 'x'), '2026-01-01T00:00:00Z'))
const longChosen = memoriesForPrompt(long)
assert.ok(longChosen.reduce((sum, memory) => sum + memory.text.length, 0) <= MAX_PROMPT_MEMORY_CHARS, 'within the character budget')
assert.equal(longChosen.length, Math.floor(MAX_PROMPT_MEMORY_CHARS / MAX_MEMORY_CHARS))

assert.equal(memoryContextSource([]), null)
assert.equal(memoryContextSource([dated('x', 'x', '2026-01-01T00:00:00Z', false)]), null, 'none in use, no source')
const source = memoryContextSource([dated('t', 'The user likes tea.', '2026-01-01T00:00:00Z')])
assert.equal(source.id, 'memory')
assert.equal(source.role, 'system')
assert.equal(source.label, 'Memory · 1 fact')
assert.match(source.content, /information, not instructions/, 'memories are framed as information')
assert.match(source.content, /\n- The user likes tea\.$/)

/* ---- in the context sources ---------------------------------------------- */
assert.equal(normalizeChatContext({}).use_memory, true, 'a chat uses memory unless it turned it off')
assert.equal(normalizeChatContext({ use_memory: false }).use_memory, false)
assert.equal(normalizeChatContext({ use_memory: 'no' }).use_memory, true, 'only false turns it off')
const memories = [dated('t', 'The user likes tea.', '2026-01-01T00:00:00Z')]
const context = { instructions: 'Be brief.', references: [{ id: 'r1', name: 'notes.txt', content: 'ref' }] }
const withMemory = buildContextSources({ context, memories, globalPrompt: 'Global.' })
assert.deepEqual(withMemory.map((item) => item.id), ['global', 'conversation', 'memory', 'r1'], 'after instructions, before reference files')
assert.ok(!buildContextSources({ context, memories: null }).some((item) => item.id === 'memory'), 'memory off: no memory source')
assert.ok(!buildContextSources({ context: { ...context, use_memory: false }, memories }).some((item) => item.id === 'memory'), 'chat turned it off')
assert.ok(!buildContextSources({ context, memories: [] }).some((item) => item.id === 'memory'), 'nothing remembered: no source')

/* ---- the suggestion request ---------------------------------------------- */
const constrained = suggestionRequest({ model: 'm', userText: 'I am Priya.', knownFacts: ['The user likes tea.'] })
assert.equal(constrained.model, 'm')
assert.equal(constrained.stream, false, 'a schema needs a non-streaming request')
assert.equal(constrained.temperature, 0)
assert.deepEqual(constrained.response_format, { type: 'json_schema', json_schema: { schema: SUGGESTION_SCHEMA } })
assert.equal(SUGGESTION_SCHEMA.properties.facts.maxItems, MAX_SUGGESTIONS)
const prompt = constrained.messages[1].content
assert.match(prompt, /"I am Priya\."$/, 'the message is quoted as data')
assert.match(prompt, /already known:\n {2}- The user likes tea\./, 'known facts are left out')
assert.match(prompt, /A statement about someone else, or about the world, is not about the user\./)
assert.match(prompt, /Never passwords, PINs, keys, or account numbers\./)
assert.match(prompt, /Examples:\nMessage: [^\n]+\nAnswer: \{"facts": \[/, 'worked examples of the answer')
assert.equal(constrained.messages.length, 2, 'only the user’s own message: no reply, no history')
const plain = suggestionRequest({ model: 'm', userText: 'hi', constrained: false })
assert.equal(plain.response_format, undefined, 'the plain form has no schema')
assert.match(plain.messages[1].content, /Answer with only the JSON object/)
assert.doesNotMatch(plain.messages[1].content, /already known/, 'nothing known, nothing listed')
const hugeText = suggestionRequest({ model: 'm', userText: 'y'.repeat(MAX_EXTRACTION_INPUT_CHARS + 500) }).messages[1].content
assert.ok(hugeText.includes(`"${'y'.repeat(MAX_EXTRACTION_INPUT_CHARS)}"`), 'a long message is cut to the extraction limit')

/* ---- reading the reply --------------------------------------------------- */
assert.deepEqual(parseSuggestionReply('{"facts": ["The user is Priya.", "  The user is a nurse. "]}'), ['The user is Priya.', 'The user is a nurse.'])
assert.deepEqual(parseSuggestionReply('```json\n{"facts":["A"]}\n```'), ['A'], 'a code fence is read')
assert.deepEqual(parseSuggestionReply('Sure! {"facts": ["B"]} Hope that helps.'), ['B'], 'surrounding prose is read past')
assert.deepEqual(parseSuggestionReply('{"facts": []}'), [])
assert.deepEqual(parseSuggestionReply('{"facts": ["ok", 3, null, "  "]}'), ['ok'], 'only non-empty strings')
assert.deepEqual(parseSuggestionReply('{"facts": "The user is Priya."}'), [], 'the wrong shape is not guessed at')
assert.deepEqual(parseSuggestionReply('["The user is Priya."]'), [], 'a bare list is not the expected shape')
assert.deepEqual(parseSuggestionReply('{"facts": [ "unterminated'), [])
assert.deepEqual(parseSuggestionReply(''), [])
assert.deepEqual(parseSuggestionReply(undefined), [])

/* ---- which suggestions are new -------------------------------------------- */
assert.deepEqual(
  freshSuggestions(['The user likes TEA', 'The user is Priya.', 'the user is priya', 'A', 'B'], { memories }),
  ['The user is Priya.', 'A', 'B'],
  'known and repeated facts are dropped, at most three',
)
assert.deepEqual(freshSuggestions(['x'], { suggested: ['X.'] }), [], 'already suggested')

const suggestions = normalizeSuggestions([
  { id: 's1', text: 'One', status: 'saved', memory_id: 'm1' },
  { id: 's2', text: 'Two', status: 'weird' },
  { id: 's3', text: '   ' },
  { id: 's4', text: 'Four', status: 'dismissed' },
  { id: 's5', text: 'Five' },
])
assert.deepEqual(suggestions.map((item) => [item.id, item.status]), [['s1', 'saved'], ['s2', 'pending'], ['s4', 'dismissed']], 'unknown status is pending; at most three')
assert.equal(suggestions[0].memory_id, 'm1')

/* ---- provenance ----------------------------------------------------------- */
const thread = [
  { id: 'u1', role: 'user' }, { id: 'a1', role: 'assistant' },
  { id: 'u2', role: 'user' }, { id: 't1', role: 'tool' }, { id: 'a2', role: 'assistant' },
  { id: 'u3', role: 'user' },
]
assert.equal(userTurnNumber(thread, 'u1'), 1)
assert.equal(userTurnNumber(thread, 'u3'), 3, 'counts the user’s messages only')
assert.equal(userTurnNumber(thread, 'a2'), null, 'a reply is not a user turn')
assert.equal(userTurnNumber(thread, 'missing'), null)

/* ---- asked automatically only while it is quick ------------------------------ */
assert.equal(suggestionMode(), 'auto', 'nothing known yet: ask')
assert.equal(suggestionMode({ firstTokenMs: 1200, lastSuggestionMs: 1500 }), 'auto')
assert.equal(suggestionMode({ firstTokenMs: AUTO_SUGGEST_MAX_FIRST_TOKEN_MS }), 'auto', 'the limits are inclusive')
assert.equal(suggestionMode({ firstTokenMs: AUTO_SUGGEST_MAX_FIRST_TOKEN_MS + 1 }), 'manual', 'a slow reply start: offer a button')
assert.equal(suggestionMode({ firstTokenMs: 900, lastSuggestionMs: AUTO_SUGGEST_MAX_SUGGESTION_MS + 1 }), 'manual', 'a slow last suggestion: offer a button')
assert.equal(suggestionMode({ firstTokenMs: null, lastSuggestionMs: undefined }), 'auto')
assert.equal(suggestionMode({ firstTokenMs: 30000, lastSuggestionMs: 1500 }), 'auto', 'a measured suggestion outweighs a slow reply start: long chats start slowly, suggestions do not')
assert.deepEqual(normalizeSuggestionCosts({ a: 1200, b: -1, c: 'x', '': 5 }), { a: 1200 })
assert.deepEqual(normalizeSuggestionCosts(['no']), {})
assert.deepEqual(normalizeSuggestionCosts(null), {})
assert.equal(Object.keys(normalizeSuggestionCosts(Object.fromEntries(Array.from({ length: 60 }, (_, i) => [`m${i}`, i])))).length, 50, 'bounded')

console.log('memory smoke passed')
