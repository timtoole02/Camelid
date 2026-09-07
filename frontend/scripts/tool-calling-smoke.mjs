#!/usr/bin/env node
/* Tool calling — gating, normalization, loop detection, and copy that does not
 * claim more than happened.
 *
 * The defects that matter here are claims and silent mismatches, not crashes:
 *
 *   1. SAYING SOMETHING RAN. Nothing in this lane executes anything. A card that
 *      implies a tool was invoked would misdescribe the single most consequential
 *      fact about the turn.
 *
 *   2. CLAIMING THE ENGINE ENFORCES THIS GATE. It does not. POST
 *      /v1/chat/completions gates on the chat TEMPLATE and never reads
 *      `tool_capable`; a row with tool_capable:false whose template carries tools
 *      is accepted with a 200 and returns unchecked calls. This lane is
 *      deliberately stricter, and the copy must own that as a product choice.
 *
 *   3. A LOOPING MODEL RENDERED AS PROGRESS. Verified live: given a tool result,
 *      Llama 3.2 3B re-issued the identical call. Detection must ignore the call
 *      id, because the runnable lane mints non-unique ids (call_0, call_1).
 *
 * The response fixture is the engine's own committed capture, not a hand-written
 * literal.
 *
 * SSR component test — no browser, no dist build required.
 */
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')
const repoRoot = resolve(frontendRoot, '..')

let checks = 0
function check(label, fn) {
  fn()
  checks += 1
  console.log(`  ok  ${label}`)
}

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const mod = await server.ssrLoadModule('/src/lib/toolCalling.js')
  const { ToolCallsCard } = await server.ssrLoadModule('/src/components/chat/render/ToolCalls.jsx')
  const {
    DEFAULT_TOOLS,
    readToolContract,
    parseToolDefinitions,
    toolRequestFields,
    toolReadiness,
    normalizeToolCalls,
    toolCallSignature,
    detectRepeatedCall,
    replyCarriesRawEnvelope,
  } = mod

  const live = {
    api_features: [{ id: 'streaming_tool_calls', status: 'supported_current_gate', notes: 'Tool-capable chat streams emit tool_calls deltas.' }],
    api_conformance: [{ id: 'streaming_tool_calls', status: 'supported_current_gate', supported_modes: ['chat_completions_sse', 'responses_sse'], unsupported_modes: ['uncertified_model_tool_templates'] }],
  }
  const capable = { capable: true, reason: null, rowId: 'llama32_3b_instruct_q8_0' }

  console.log('tool calling — contract gating')

  check('a live contract is supported; an absent row fails closed', () => {
    assert.equal(readToolContract(live).supported, true)
    assert.equal(readToolContract({}).present, false)
    assert.deepEqual(toolRequestFields({ enabled: true, contract: readToolContract({}), capability: capable, toolsText: DEFAULT_TOOLS }), {})
  })

  check('resemblance is not evidence — a near-miss row id does not count', () => {
    assert.equal(readToolContract({ api_conformance: [{ id: 'streaming_tool_calls_v2', status: 'supported' }] }).present, false)
  })

  check('an engine row alone does not license tools — the model gate also applies', () => {
    const fields = toolRequestFields({
      enabled: true, contract: readToolContract(live),
      capability: { capable: false, reason: 'no receipt' }, toolsText: DEFAULT_TOOLS,
    })
    assert.deepEqual(fields, {}, 'a model without a tool receipt must contribute no tools field')
  })

  check('the guarded reason does not claim the engine enforces this gate', () => {
    // The engine gates on the chat template and never reads tool_capable. Copy
    // asserting otherwise would be a false statement about the backend.
    const readiness = toolReadiness({
      enabled: true, contract: readToolContract(live),
      capability: { capable: false, reason: 'This model has no tool receipt. The engine may still accept tools for it, but the results are unchecked, so Camelid does not offer them here.' },
      toolsText: DEFAULT_TOOLS,
    })
    assert.equal(readiness.ready, false)
    assert.doesNotMatch(readiness.reason, /refused by the engine|engine refuses/i)
    assert.match(readiness.reason, /may still accept|unchecked/i)
  })

  console.log('tool calling — request composition')

  check('the default tool definition is valid', () => {
    const parsed = parseToolDefinitions(DEFAULT_TOOLS)
    assert.equal(parsed.ok, true)
    assert.equal(parsed.value[0].function.name, 'get_weather')
  })

  check('a malformed tools array is explained at edit time, not sent', () => {
    for (const [text, pattern] of [
      ['{ not json', /valid JSON/i],
      ['{}', /array/i],
      ['[]', /at least one/i],
      ['[{"type":"x"}]', /"type" must be "function"/],
      ['[{"type":"function","function":{}}]', /name" is required/],
    ]) {
      const parsed = parseToolDefinitions(text)
      assert.equal(parsed.ok, false, `expected ${text} to be rejected`)
      assert.match(parsed.error, pattern)
      assert.deepEqual(
        toolRequestFields({ enabled: true, contract: readToolContract(live), capability: capable, toolsText: text }),
        {},
        'an invalid definition must contribute no tools field',
      )
    }
  })

  check('tools ride on the ordinary send and never force it off the stream', () => {
    // Unlike receipts and constrained decoding, tool calling is supported on the
    // streaming path — so nothing here may touch the stream flag.
    const fields = toolRequestFields({ enabled: true, contract: readToolContract(live), capability: capable, toolsText: DEFAULT_TOOLS })
    assert.deepEqual(Object.keys(fields), ['tools'])
    assert.ok(!('stream' in fields))
  })

  console.log('tool calling — normalization')

  /* The engine's own captured response, if the repo carries it. */
  const fixturePath = resolve(repoRoot, 'qa/capability/mac_tools_out/llama-3.2-3b/p1.json')
  if (existsSync(fixturePath)) {
    check('the committed engine capture still has the shape this surface reads', () => {
      const captured = JSON.parse(readFileSync(fixturePath, 'utf8'))
      const message = captured.choices[0].message
      assert.equal(captured.choices[0].finish_reason, 'tool_calls')
      const calls = normalizeToolCalls(message.tool_calls)
      assert.ok(calls && calls.length >= 1)
      assert.equal(typeof calls[0].name, 'string')
      assert.ok(calls[0].parsedArguments, 'arguments must parse from the real capture')
    })
  } else {
    console.log('  --  engine capture fixture absent on this branch; skipped')
  }

  check('malformed arguments are reported, never swallowed', () => {
    const calls = normalizeToolCalls([{ id: 'c', type: 'function', function: { name: 'f', arguments: '{not json' } }])
    assert.equal(calls[0].parsedArguments, null)
    assert.ok(calls[0].parseError, 'the parse failure must be surfaced')
    assert.equal(calls[0].rawArguments, '{not json', 'and the raw text preserved')
  })

  check('an empty or absent list yields nothing rather than an empty card', () => {
    assert.equal(normalizeToolCalls(null), null)
    assert.equal(normalizeToolCalls([]), null)
  })

  console.log('tool calling — loop detection')

  check('a repeated call is detected across differing ids and whitespace', () => {
    const first = normalizeToolCalls([{ id: 'call_a', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }])
    // Same request, different id and spacing — the runnable lane mints
    // non-unique ids (call_0, call_1), so identity must not depend on them.
    const again = normalizeToolCalls([{ id: 'call_0', type: 'function', function: { name: 'get_weather', arguments: '{"city": "Paris"}' } }])
    assert.equal(detectRepeatedCall([], first).repeated, false, 'the first call is not a repeat')
    const verdict = detectRepeatedCall(first.map(toolCallSignature), again)
    assert.equal(verdict.repeated, true)
    assert.equal(verdict.name, 'get_weather')
  })

  check('a genuinely different call is not flagged', () => {
    const first = normalizeToolCalls([{ id: 'a', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }])
    const other = normalizeToolCalls([{ id: 'b', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Berlin"}' } }])
    assert.equal(detectRepeatedCall(first.map(toolCallSignature), other).repeated, false)
  })

  console.log('tool calling — the runnable lane leaves its envelope in the text')

  check('a raw envelope in the reply is detected, and ordinary prose is not', () => {
    assert.equal(replyCarriesRawEnvelope('The weather in Paris is mild.'), false)
    assert.equal(replyCarriesRawEnvelope(''), false)
    assert.equal(replyCarriesRawEnvelope('<function=get_weather><parameter=city>Paris</parameter></function>'), true)
    assert.equal(replyCarriesRawEnvelope('<tool_call>{"name":"x"}</tool_call>'), true)
    assert.equal(replyCarriesRawEnvelope('[TOOL_CALLS] foo'), true)
  })

  console.log('tool calling — rendered copy')

  const CALLS = [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }]
  const render = (props) => renderToStaticMarkup(React.createElement(ToolCallsCard, { toolCalls: CALLS, ...props }))

  check('the card never says anything ran', () => {
    const html = render({})
    assert.match(html, /get_weather/)
    assert.match(html, /Nothing\s*has run/i)
    assert.doesNotMatch(html, /\bexecuted\b|\bwas run\b|\bresult:/i)
    assert.doesNotMatch(html, /\bsucceeded\b|\bcompleted successfully\b/i)
  })

  check('a repeat is named as a repeat, not shown as progress', () => {
    const html = render({ repeated: { repeated: true, name: 'get_weather', signature: 's' } })
    assert.match(html, /same call as before/i)
    assert.match(html, /not using the result/i)
  })

  check('the duplicated-envelope case is explained rather than looking like two requests', () => {
    const html = render({ replyContent: '<function=get_weather><parameter=city>Paris</parameter></function>' })
    assert.match(html, /same request, not two requests/i)
  })

  check('nothing renders without calls', () => {
    assert.equal(renderToStaticMarkup(React.createElement(ToolCallsCard, { toolCalls: null })), '')
  })

  check('the lane executes nothing — asserted at the source', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/lib/toolCalling.js'), 'utf8')
    const component = readFileSync(resolve(frontendRoot, 'src/components/chat/render/ToolCalls.jsx'), 'utf8')
    for (const [name, text] of [['lib', source], ['component', component]]) {
      assert.doesNotMatch(text, /\beval\(|new Function\(|child_process|\bexec\(/, `${name} must contain no execution path`)
    }
  })

  console.log(`\n${checks} checks passed`)
} finally {
  await server.close()
}
