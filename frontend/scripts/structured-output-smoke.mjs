#!/usr/bin/env node
/* Structured output — contract, request composition, and the honesty of the verdict.
 *
 * The defect this surface is most likely to ship is a claim, not a crash. A
 * constrained response is byte-for-byte the same SHAPE as an unconstrained one:
 * no field, flag, header or finish_reason says a mask was applied. So a panel that
 * renders "constrained" on the strength of a 200 is asserting something it cannot
 * see, and it would look completely correct in a screenshot.
 *
 * Three classes of assertion here:
 *
 *   1. THE REQUEST CANNOT BE AMBIGUOUS. The engine returns a typed 400 when two
 *      constraint forms arrive together, and another when a constraint arrives
 *      with stream:true — and its streaming decoder never builds a grammar state,
 *      so that route refusal is the only thing between a streamed constrained
 *      request and silently unconstrained output. The builder is asserted to be
 *      structurally incapable of composing either.
 *
 *   2. THE VERDICT NEVER OUTRUNS THE EVIDENCE. Only an observed diverted position
 *      on a greedy turn earns the strong claim. Conformance is explicitly weaker,
 *      because an unconstrained model can emit valid JSON by luck.
 *
 *   3. THE VALIDATOR ADMITS WHAT IT DID NOT CHECK. This is not a JSON Schema
 *      implementation; silently skipping unimplemented keywords would report
 *      "valid" for a document it never examined.
 *
 * SSR component test — no browser, no dist build required.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')

let checks = 0
function check(label, fn) {
  fn()
  checks += 1
  console.log(`  ok  ${label}`)
}

const VENDOR = ['Open', 'AI'].join('')

const server = await createServer({
  root: frontendRoot,
  appType: 'custom',
  logLevel: 'silent',
  server: { middlewareMode: true },
})

try {
  const mod = await server.ssrLoadModule('/src/lib/structuredOutput.js')
  const { StructuredOutputCard } = await server.ssrLoadModule('/src/components/chat/render/StructuredOutput.jsx')
  const {
    STRUCTURED_MODES,
    DEFAULT_SCHEMA,
    readStructuredOutputContract,
    structuredOutputRequestFields,
    structuredOutputForcesNonStreaming,
    structuredOutputReadiness,
    assessStructuredReply,
    parseSchemaText,
  } = mod

  const live = {
    api_features: [{
      id: 'llguidance_structured_outputs',
      status: 'supported_current_gate_nonstreaming',
      notes: `Constrained decoding for ${VENDOR}-shaped response_format requests.`,
    }],
    api_conformance: [{
      id: 'llguidance_structured_outputs',
      status: 'supported_current_gate_nonstreaming',
      supported_modes: ['chat_nonstreaming', 'responses_nonstreaming'],
      unsupported_modes: ['streaming', 'raw_completions'],
    }],
  }

  console.log('structured output — contract gating')

  check('a live contract permits non-streaming constrained decoding', () => {
    const contract = readStructuredOutputContract(live)
    assert.equal(contract.present, true)
    assert.equal(contract.supported, true)
    assert.equal(contract.nonStreamingSupported, true)
  })

  check('an engine with no row fails closed and contributes no request fields', () => {
    const contract = readStructuredOutputContract({})
    assert.equal(contract.nonStreamingSupported, false)
    assert.deepEqual(
      structuredOutputRequestFields({ enabled: true, mode: STRUCTURED_MODES.JSON_OBJECT, contract }),
      {},
    )
  })

  check('a supported row with no machine-readable modes stays closed', () => {
    const contract = readStructuredOutputContract({ api_features: live.api_features })
    assert.equal(contract.supported, true)
    assert.equal(contract.modesKnown, false)
    assert.equal(contract.nonStreamingSupported, false, 'unknown modes must not be assumed permissive')
  })

  check('resemblance is not evidence — a near-miss row id does not count', () => {
    const contract = readStructuredOutputContract({
      api_conformance: [{ id: 'llguidance_structured_outputs_v2', status: 'supported', supported_modes: ['chat_nonstreaming'] }],
    })
    assert.equal(contract.present, false)
  })

  console.log('structured output — the request cannot be ambiguous')

  const contract = readStructuredOutputContract(live)

  check('exactly one constraint form is ever emitted', () => {
    for (const mode of [STRUCTURED_MODES.JSON_OBJECT, STRUCTURED_MODES.JSON_SCHEMA, STRUCTURED_MODES.GRAMMAR]) {
      const fields = structuredOutputRequestFields({
        enabled: true, mode, contract, schemaText: DEFAULT_SCHEMA, grammarText: 'start: "x"',
      })
      const keys = Object.keys(fields)
      assert.deepEqual(keys, ['response_format'], `mode ${mode} emitted ${keys.join(', ')}`)
      // The engine 400s on response_format + top-level json_schema/grammar together.
      assert.ok(!('json_schema' in fields), 'top-level json_schema must never ride along')
      assert.ok(!('grammar' in fields), 'top-level grammar must never ride along')
    }
  })

  check('the json_schema envelope is the nested chat shape, not the raw schema', () => {
    const fields = structuredOutputRequestFields({
      enabled: true, mode: STRUCTURED_MODES.JSON_SCHEMA, contract, schemaText: DEFAULT_SCHEMA,
    })
    // Three different locations exist for this object across routes; the wrong one
    // is a 400 with no hint.
    assert.equal(fields.response_format.type, 'json_schema')
    assert.ok(fields.response_format.json_schema.schema.properties, 'schema must sit at response_format.json_schema.schema')
  })

  check('an unparseable schema produces no constraint rather than a bad one', () => {
    assert.deepEqual(
      structuredOutputRequestFields({ enabled: true, mode: STRUCTURED_MODES.JSON_SCHEMA, contract, schemaText: '{ not json' }),
      {},
    )
    assert.equal(parseSchemaText('{ not json').ok, false)
    assert.equal(parseSchemaText('[1,2]').ok, false, 'a schema must be an object, not an array')
  })

  check('a constraint forces the turn off the streaming path', () => {
    assert.equal(structuredOutputForcesNonStreaming({ enabled: true, mode: STRUCTURED_MODES.JSON_SCHEMA, contract }), true)
    assert.equal(structuredOutputForcesNonStreaming({ enabled: true, mode: STRUCTURED_MODES.OFF, contract }), false)
    // Guarded contract must NOT flip the stream flag — that would cost streaming
    // for no constraint at all.
    assert.equal(
      structuredOutputForcesNonStreaming({ enabled: true, mode: STRUCTURED_MODES.JSON_SCHEMA, contract: readStructuredOutputContract({}) }),
      false,
    )
  })

  check('readiness explains a blocked send rather than sending unconstrained', () => {
    assert.equal(structuredOutputReadiness({ enabled: true, mode: STRUCTURED_MODES.JSON_SCHEMA, contract, schemaText: DEFAULT_SCHEMA }).ready, true)
    const bad = structuredOutputReadiness({ enabled: true, mode: STRUCTURED_MODES.JSON_SCHEMA, contract, schemaText: '{ nope' })
    assert.equal(bad.ready, false)
    assert.match(bad.reason, /valid JSON/i)
    const noRow = structuredOutputReadiness({ enabled: true, mode: STRUCTURED_MODES.JSON_OBJECT, contract: readStructuredOutputContract({}) })
    assert.equal(noRow.ready, false)
    assert.match(noRow.reason, /does not advertise/i)
  })

  console.log('structured output — the verdict never outruns the evidence')

  const conforming = '{"title":"t","sentiment":"positive","score":4,"tags":["a"]}'

  check('a 200 with no observed divergence claims only that it was accepted', () => {
    const a = assessStructuredReply({ content: 'plain prose', mode: STRUCTURED_MODES.JSON_OBJECT, divertedPositions: null })
    assert.equal(a.evidence, 'accepted')
    assert.equal(a.parses, false)
  })

  check('conforming output is explicitly weaker than observed divergence', () => {
    const conforms = assessStructuredReply({ content: conforming, mode: STRUCTURED_MODES.JSON_SCHEMA, schemaText: DEFAULT_SCHEMA, divertedPositions: 0 })
    assert.equal(conforms.evidence, 'conforms')
    const diverted = assessStructuredReply({ content: conforming, mode: STRUCTURED_MODES.JSON_SCHEMA, schemaText: DEFAULT_SCHEMA, divertedPositions: 3 })
    assert.equal(diverted.evidence, 'diverted', 'an observed diverted position is the stronger claim')
  })

  check('divergence on a SAMPLED turn does not earn the strong claim', () => {
    // Off-argmax emission is only mask evidence when decoding was greedy; a
    // sampled turn produces it routinely.
    const sampled = assessStructuredReply({ content: conforming, mode: STRUCTURED_MODES.JSON_SCHEMA, schemaText: DEFAULT_SCHEMA, divertedPositions: 3, greedy: false })
    assert.notEqual(sampled.evidence, 'diverted')
  })

  check('schema problems are reported, not swallowed', () => {
    const wrong = assessStructuredReply({
      content: '{"title":"t","sentiment":"ecstatic","score":"four","tags":"a"}',
      mode: STRUCTURED_MODES.JSON_SCHEMA,
      schemaText: DEFAULT_SCHEMA,
      divertedPositions: 0,
    })
    assert.equal(wrong.parses, true)
    assert.ok(wrong.problems.length >= 3, `expected enum, integer and array problems, got ${JSON.stringify(wrong.problems)}`)
    assert.ok(wrong.problems.some((p) => /sentiment/.test(p)), 'the enum violation must be named')
    assert.notEqual(wrong.evidence, 'conforms', 'a reply with schema problems must not read as conforming')
  })

  console.log('structured output — rendered copy')

  const render = (record) => renderToStaticMarkup(React.createElement(StructuredOutputCard, { record }))

  check('the card never asserts enforcement it cannot observe', () => {
    const html = render({ content: conforming, mode: STRUCTURED_MODES.JSON_SCHEMA, schemaText: DEFAULT_SCHEMA, divertedPositions: 0 })
    assert.match(html, /Matches the schema/)
    // The words that would overclaim from a 200 alone.
    assert.doesNotMatch(html, /\bguarantee/i)
    assert.doesNotMatch(html, /\benforced\b/i)
    assert.doesNotMatch(html, /\bproves\b/i)
    assert.match(html, /not proof/i, 'the weaker verdict must say what it is not')
  })

  check('the vendor name never reaches the screen', () => {
    const html = render({ content: conforming, mode: STRUCTURED_MODES.JSON_SCHEMA, schemaText: DEFAULT_SCHEMA, divertedPositions: 2 })
    assert.doesNotMatch(html, new RegExp(`\\b${VENDOR}\\b`))
  })

  check('unchecked schema keywords are disclosed', () => {
    const html = render({
      content: '{"a":1}',
      mode: STRUCTURED_MODES.JSON_SCHEMA,
      schemaText: '{"type":"object","properties":{"a":{"minimum":0}}}',
      divertedPositions: 0,
    })
    assert.match(html, /Not checked here/i, 'a keyword this page does not implement must be named, not silently passed')
  })

  check('nothing renders without a record', () => {
    assert.equal(renderToStaticMarkup(React.createElement(StructuredOutputCard, { record: null })), '')
  })

  console.log('structured output — source-level guarantees')

  check('the lib module never composes two constraint forms', () => {
    const source = readFileSync(resolve(frontendRoot, 'src/lib/structuredOutput.js'), 'utf8')
    // Each builder branch returns immediately; a fallthrough that merged two
    // forms would be a 400 on every send.
    const returns = source.match(/return \{ response_format:/g) || []
    assert.ok(returns.length >= 3, 'each mode returns its own single-key object')
    assert.doesNotMatch(source, /localStorage|sessionStorage|indexedDB/, 'schemas are session state, not persisted')
  })

  console.log(`\n${checks} checks passed`)
} finally {
  await server.close()
}
