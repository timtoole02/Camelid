#!/usr/bin/env node
/* Unit coverage for the divergence view's pure layer.
 *
 * This screen is the one most able to mislead: two answers side by side invite
 * a reader to decide which is right. These checks pin the rules that stop the
 * page from helping them do that.
 *
 * No browser and no network: every input is a literal proxy answer.
 */
import assert from 'node:assert/strict'
import {
  advertisedTemplateNote,
  buildCompareRequest,
  comparisonCaveats,
  describeComparison,
  identityStatement,
  isAttributable,
  modelChoices,
  modelLooksAbsent,
  needsIdentityAssertion,
  renderedPromptComparison,
  reportedModelMismatch,
  templateDivergence,
  tokenCapStatement,
  uncontrolledStatements,
  verdictHeadline,
} from '../src/lib/divergenceModel.js'
import { requestComparison } from '../src/lib/divergenceClient.js'

let checks = 0
function check(name, fn) {
  fn()
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}

async function checkAsync(name, fn) {
  await fn()
  checks += 1
  process.stdout.write(`  ok  ${name}\n`)
}

const CAMELID_TEMPLATE = '{{- bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}'
const OLLAMA_TEMPLATE = '{{ if .System }}Cutting Knowledge Date: December 2023{{ end }}{{ .Prompt }}'

function side(overrides = {}) {
  return {
    label: 'win',
    engine: 'camelid',
    engine_version: 'v0.6.1-267',
    model: 'llama-3.2-1b',
    applied_sampling: { temperature: 'sent', seed: 'sent' },
    samples: [
      { text: '12', sha256: 'aa', elapsed_ms: 30 },
      { text: '12', sha256: 'aa', elapsed_ms: 28 },
    ],
    stability: { kind: 'stable' },
    advertised_template: { kind: 'captured', source: 'GET /props', template: CAMELID_TEMPLATE },
    rendered_prompt: { kind: 'captured', source: 'POST /apply-template', text: '<|user|>What is 7 plus 5?' },
    ...overrides,
  }
}

function body(overrides = {}) {
  return {
    prompt: 'What is 7 plus 5?',
    prompt_sha256: 'f00d',
    plan: { temperature: 0, seed: 0, max_tokens: 64, repetitions: 2 },
    left: side(),
    right: side({
      label: 'studio',
      engine: 'ollama',
      engine_version: '0.33.3',
      samples: [
        { text: '7', sha256: 'bb', elapsed_ms: 41 },
        { text: '7', sha256: 'bb', elapsed_ms: 39 },
      ],
      advertised_template: { kind: 'captured', source: 'POST /api/show', template: OLLAMA_TEMPLATE },
      rendered_prompt: { kind: 'unavailable', reason: "Ollama's documented API has no route that renders a chat prompt without generating" },
    }),
    verdict: { kind: 'divergent' },
    diff: {
      kind: 'lines',
      lines: [{ op: 'removed', text: '12' }, { op: 'added', text: '7' }],
    },
    uncontrolled: [],
    ...overrides,
  }
}

console.log('divergence model')

/* ---- the headline finding ---- */

check('a divergence between two self-consistent nodes is attributable and diffed', () => {
  const comparison = describeComparison(body())
  assert.equal(comparison.verdict.kind, 'divergent')
  assert.ok(isAttributable(comparison))
  assert.equal(comparison.diff.changedLines, 2)
  assert.equal(comparison.left.settledDigest, 'aa')
  assert.equal(comparison.right.settledDigest, 'bb')
  assert.match(verdictHeadline(comparison), /agreed with themselves, and disagreed/)
})

check('the two templates are offered as the explanation only when both were captured', () => {
  const divergence = templateDivergence(describeComparison(body()))
  assert.equal(divergence.differ, true)
  assert.equal(divergence.left.source, 'GET /props')
  assert.equal(divergence.right.source, 'POST /api/show')
})

/* ---- advertised is not applied (C1) ---- */

check('identical advertised templates beside a divergence are said not to explain it', () => {
  const comparison = describeComparison(body({
    right: side({
      label: 'studio',
      engine: 'ollama',
      samples: [{ text: 'Hi. How can I assist you today?', sha256: 'bb', elapsed_ms: 41 }, { text: 'Hi. How can I assist you today?', sha256: 'bb', elapsed_ms: 39 }],
      advertised_template: { kind: 'captured', source: 'POST /api/show', template: CAMELID_TEMPLATE },
    }),
  }))
  const templates = templateDivergence(comparison)
  assert.equal(templates.differ, false)
  assert.equal(templates.unexplained, true)
  assert.match(advertisedTemplateNote(comparison), /does not explain this difference/)
  const agreeing = describeComparison(body({ verdict: { kind: 'identical' }, right: side({ label: 'mac' }) }))
  assert.equal(templateDivergence(agreeing).unexplained, false, 'only a divergence can be left unexplained')
})

check('no note about templates ever calls an advertised template applied', () => {
  const cases = [
    body(),
    body({ right: side({ label: 'mac', advertised_template: { kind: 'captured', source: 'GET /props', template: CAMELID_TEMPLATE } }) }),
    body({ verdict: { kind: 'identical' }, right: side({ label: 'mac' }) }),
  ]
  for (const raw of cases) {
    const note = advertisedTemplateNote(describeComparison(raw))
    assert.ok(note, 'both templates were captured, so something is said')
    assert.doesNotMatch(note, /\bappl(y|ied)\b/i, note)
  }
})

check('an older proxy\'s template is read as advertised, and its rendered prompt as not reported', () => {
  const legacySide = side()
  legacySide.template = legacySide.advertised_template
  delete legacySide.advertised_template
  delete legacySide.rendered_prompt
  const comparison = describeComparison(body({ left: legacySide }))
  assert.equal(comparison.left.advertisedTemplate.kind, 'captured', 'the old capture was always the advertised one')
  assert.equal(comparison.left.advertisedTemplate.source, 'GET /props')
  assert.equal(comparison.left.renderedPrompt.reported, false)
  assert.equal(comparison.left.renderedPrompt.kind, null, 'absent is not "could not be captured"')
})

check('a rendered prompt keeps its source and text, and one not taken keeps its reason', () => {
  const comparison = describeComparison(body())
  assert.deepEqual(
    [comparison.left.renderedPrompt.kind, comparison.left.renderedPrompt.source, comparison.left.renderedPrompt.text],
    ['captured', 'POST /apply-template', '<|user|>What is 7 plus 5?'],
  )
  assert.equal(comparison.right.renderedPrompt.kind, 'unavailable')
  assert.match(comparison.right.renderedPrompt.reason, /no route that renders/)
  assert.equal(renderedPromptComparison(comparison), null, 'one render is not a comparison of two')
  const both = describeComparison(body({ right: side({ label: 'mac' }) }))
  assert.deepEqual(renderedPromptComparison(both), { same: true })
})

/* ---- refusals to conclude are not differences ---- */

check('an unstable side is not attributable and carries no settled digest', () => {
  const comparison = describeComparison(body({
    right: side({
      label: 'studio',
      samples: [
        { text: '7', sha256: 'bb', elapsed_ms: 41 },
        { text: 'seven', sha256: 'cc', elapsed_ms: 39 },
      ],
      stability: { kind: 'unstable', digests: ['bb', 'cc'] },
    }),
    verdict: { kind: 'not_attributable', reason: 'studio did not repeat its own answer' },
    diff: { kind: 'declined', reason: 'the two sides are not comparable, so no diff is shown' },
  }))

  assert.equal(isAttributable(comparison), false)
  assert.equal(comparison.right.settledDigest, null, 'an unstable side represents no single answer')
  assert.equal(comparison.right.stability.distinctAnswers, 2)
  assert.equal(comparison.diff.changedLines, null, 'a declined diff reports no change count')
  assert.match(verdictHeadline(comparison), /Nothing can be concluded/)
})

check('different models are reported as not comparable, never as a disagreement', () => {
  const comparison = describeComparison(body({
    verdict: { kind: 'different_models', left: 'llama-3.2-1b', right: 'qwen3:8b' },
    diff: { kind: 'declined', reason: 'not comparable' },
  }))
  assert.equal(isAttributable(comparison), false)
  const headline = verdictHeadline(comparison)
  assert.match(headline, /Not comparable/)
  assert.match(headline, /llama-3\.2-1b/)
  assert.match(headline, /qwen3:8b/)
  assert.doesNotMatch(headline, /disagree/, 'a model swap is not a disagreement between engines')
})

check('a verdict this build does not know is unknown, never read as agreement', () => {
  const comparison = describeComparison(body({ verdict: { kind: 'probably_fine' } }))
  assert.equal(comparison.verdict.kind, null)
  assert.equal(isAttributable(comparison), false)
  assert.match(verdictHeadline(comparison), /does not recognise/)
})

check('no headline this build can produce judges either side', () => {
  const verdicts = [
    { kind: 'identical' },
    { kind: 'divergent' },
    { kind: 'different_models', left: 'a', right: 'b' },
    { kind: 'not_attributable', reason: 'studio did not repeat its own answer' },
    { kind: 'something_new' },
  ]
  for (const verdict of verdicts) {
    const headline = verdictHeadline(describeComparison(body({ verdict })))
    for (const banned of ['correct', 'incorrect', 'wrong', 'better', 'worse', 'accurate', 'winner']) {
      assert.ok(!headline.toLowerCase().includes(banned), `${banned} in "${headline}"`)
    }
  }
})

/* ---- what must still be said when a verdict WAS reached ---- */

check('an engine with no seed parameter is disclosed even on an attributable verdict', () => {
  const comparison = describeComparison(body({
    right: side({
      label: 'desk',
      engine: 'lmstudio',
      engine_version: null,
      runtime: 'llama.cpp-win-x86_64-cuda12 2.34.0',
      applied_sampling: { temperature: 'sent', seed: 'unsupported' },
      samples: [
        { text: '7', sha256: 'bb', elapsed_ms: 41 },
        { text: '7', sha256: 'bb', elapsed_ms: 39 },
      ],
      advertised_template: { kind: 'not_exposed', detail: "LM Studio's documented API exposes no prompt template" },
    }),
    uncontrolled: ['seed'],
  }))

  assert.ok(isAttributable(comparison), 'an uncontrolled seed is disclosed, not a reason to withhold')
  assert.equal(comparison.right.appliedSampling.seed, 'unsupported')
  assert.equal(comparison.right.engineVersion, null, 'LM Studio publishes no version')
  assert.equal(
    comparison.right.runtime,
    'llama.cpp-win-x86_64-cuda12 2.34.0',
    'the runtime is reported under its own name, never as the engine version',
  )
  const caveats = comparisonCaveats(comparison)
  assert.deepEqual(uncontrolledStatements(comparison), [
    { name: 'seed', text: 'at least one engine has no such parameter.' },
  ], 'an older proxy that named seed gave no other reason than a missing parameter')
  assert.ok(caveats.some((c) => /exposes no prompt template/.test(c)), caveats.join(' | '))
  assert.equal(templateDivergence(comparison), null, 'one template is not a comparison of two')
})

check('a single run per side is called out as never having been tested', () => {
  const once = side({ samples: [{ text: '12', sha256: 'aa', elapsed_ms: 30 }], stability: { kind: 'unmeasured' } })
  const caveats = comparisonCaveats(describeComparison(body({ left: once })))
  assert.ok(caveats.some((c) => /run once/.test(c)), caveats.join(' | '))
})

check('identical templates are reported as not differing rather than as an explanation', () => {
  const comparison = describeComparison(body({
    right: side({ label: 'mac', samples: [{ text: '12', sha256: 'aa', elapsed_ms: 9 }, { text: '12', sha256: 'aa', elapsed_ms: 8 }] }),
    verdict: { kind: 'identical' },
    diff: { kind: 'identical' },
  }))
  assert.equal(templateDivergence(comparison).differ, false)
  assert.match(verdictHeadline(comparison), /same bytes/)
})

/* ---- malformed input is unknown, never a default ---- */

check('a malformed comparison yields nulls rather than invented structure', () => {
  assert.equal(describeComparison(null), null)
  assert.equal(describeComparison('nope'), null)
  const empty = describeComparison({})
  assert.equal(empty.verdict.kind, null)
  assert.equal(empty.left, null)
  assert.equal(empty.diff.kind, null)
  assert.deepEqual(empty.uncontrolled, [])
  assert.equal(isAttributable(empty), false)
})

check('an unrecognised stability or template kind is null rather than the nearest known one', () => {
  const comparison = describeComparison(body({
    left: side({ stability: { kind: 'probably_stable' }, advertised_template: { kind: 'inferred', template: 'x' } }),
  }))
  assert.equal(comparison.left.stability.kind, null)
  assert.equal(comparison.left.advertisedTemplate.kind, null)
  assert.equal(comparison.left.settledDigest, null, 'only a known-stable side has a settled digest')
  assert.equal(templateDivergence(comparison), null)
})

check('a diff line with an unknown op keeps its text but not a guessed op', () => {
  const comparison = describeComparison(body({
    diff: { kind: 'lines', lines: [{ op: 'maybe', text: '12' }, { op: 'added', text: '7' }] },
  }))
  assert.equal(comparison.diff.lines[0].op, null)
  assert.equal(comparison.diff.lines[0].text, '12')
})

/* ---- choosing a model, without inventing one ---- */

const READY = {
  label: 'studio', engine: 'ollama', state: 'ready',
  models: ['llama-3.2-1b-instruct:latest', 'qwen3:8b'], reason: null,
}

check('a ready node offers exactly the models the proxy said it holds', () => {
  const choices = modelChoices({ nodes: [READY] }, 'studio')
  assert.equal(choices.kind, 'listed')
  assert.deepEqual(choices.models, ['llama-3.2-1b-instruct:latest', 'qwen3:8b'])
  assert.equal(choices.engine, 'ollama')
})

check('a proxy that withholds its node detail offers no list, and is not an empty one', () => {
  // Off-loopback the proxy discloses no node_detail. Rendering an empty picker
  // would tell the operator the node holds nothing.
  assert.equal(modelChoices({ nodes: null }, 'studio').kind, 'withheld')
})

check('a fabric that could not be read is neither withheld nor empty', () => {
  // Not having looked is its own fact. This used to be reported as "withheld",
  // which told an operator whose proxy was down that it was hiding its nodes.
  assert.equal(modelChoices(null, 'studio').kind, 'unread')
  assert.equal(modelChoices({ nodes: null, problem: { code: 'unreachable' } }, 'studio').kind, 'unread')
})

check('a node that is not ready has no list rather than an empty one, and says why', () => {
  const choices = modelChoices({
    nodes: [{ label: 'mac', state: 'not_ready', reason: 'no model loaded', models: null }],
  }, 'mac')
  assert.equal(choices.kind, 'not_ready')
  assert.equal(choices.reason, 'no model loaded')
})

check('an unknown label and an unreadable list are distinguished from each other', () => {
  assert.equal(modelChoices({ nodes: [READY] }, 'nowhere').kind, 'no_node')
  assert.equal(modelChoices({ nodes: [READY] }, '').kind, 'no_node')
  assert.equal(
    modelChoices({ nodes: [{ label: 'x', state: 'ready', models: null }] }, 'x').kind,
    'unknown',
  )
})

check('a typed model absent from the list is a warning, never a verdict', () => {
  const choices = modelChoices({ nodes: [READY] }, 'studio')
  assert.equal(modelLooksAbsent(choices, 'qwen3:8b'), false)
  assert.equal(modelLooksAbsent(choices, 'not-there'), true)
  // Nothing to compare against means no warning: the proxy remains the
  // authority and refuses by name if it really is absent.
  assert.equal(modelLooksAbsent({ kind: 'withheld' }, 'anything'), false)
  assert.equal(modelLooksAbsent(choices, ''), false)
})

check('two ids that differ require an explicit assertion, and identical ones do not', () => {
  // The picker may suggest a pairing; it must never assert one. Stripping
  // `:latest` here would be exactly the fuzzy matching the CLI refuses.
  assert.equal(needsIdentityAssertion('a', 'a'), false)
  assert.equal(
    needsIdentityAssertion('llama-3.2-1b-instruct:latest', 'llama-3.2-1b-instruct'),
    true,
    'a `:latest` suffix is not evidence that two names are the same weights',
  )
  assert.equal(needsIdentityAssertion('', 'a'), false, 'an unfilled field asserts nothing')
  assert.equal(needsIdentityAssertion('a', ''), false)
})

/* ---- the claim a result rests on travels with it ---- */

check('an asserted identity is read from the result and stated as unverified, naming both ids', () => {
  const comparison = describeComparison(body({
    left: side({ model: 'llama-3.2-1b-instruct:latest' }),
    right: side({ label: 'desk', model: 'llama-3.2-1b-instruct' }),
    model_identity: 'asserted_by_operator',
    uncontrolled: ['model identity'],
  }))
  assert.equal(comparison.modelIdentity, 'asserted_by_operator')
  const statement = identityStatement(comparison)
  assert.equal(statement.kind, 'asserted_by_operator')
  assert.match(statement.text, /not verified/)
  assert.match(statement.text, /llama-3\.2-1b-instruct:latest/)
  assert.match(statement.text, /llama-3\.2-1b-instruct was|and llama-3\.2-1b-instruct /)
})

check('a same-id result says the id is all that was compared', () => {
  const statement = identityStatement(describeComparison(body({ model_identity: 'same_id' })))
  assert.equal(statement.kind, 'same_id')
  assert.match(statement.text, /same id, llama-3\.2-1b/)
  assert.match(statement.text, /digest/)
})

/* ---- model identity by weights digest (C4) ---- */

const DIGEST = '432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3'
const published = (source) => ({ kind: 'published', digest: DIGEST, source })

check('an identity the proxy verified by digest is stated with the digest it rests on', () => {
  const comparison = describeComparison(body({
    model_identity: 'verified_by_digest',
    left: side({ weights_digest: published('GET /v1/models gguf_sha256'), weights_check: { kind: 'enforced', expected: DIGEST } }),
    right: side({ label: 'studio', weights_digest: published('POST /api/show modelfile FROM blob') }),
  }))
  const statement = identityStatement(comparison)
  assert.equal(statement.kind, 'verified_by_digest')
  assert.match(statement.text, new RegExp(`Verified by digest.*GGUF file, sha256 ${DIGEST}`))
  assert.doesNotMatch(statement.text, /asserted|not verified/i)
  assert.deepEqual(
    [comparison.left.weightsDigest.kind, comparison.left.weightsDigest.digest, comparison.left.weightsCheck.kind],
    ['published', DIGEST, 'enforced'],
  )
})

check('this build never concludes identity from two digests the proxy did not call verified', () => {
  // Two equal strings under a proxy that said same_id stay same_id: only the
  // proxy knows whether they were weights digests or something else.
  const comparison = describeComparison(body({
    model_identity: 'same_id',
    left: side({ weights_digest: published('x') }),
    right: side({ label: 'studio', weights_digest: published('y') }),
  }))
  assert.equal(identityStatement(comparison).kind, 'same_id')
})

check('a refused weights check and an unreported digest are each their own fact', () => {
  const refused = describeComparison(body({
    left: side({
      weights_digest: { kind: 'unavailable', reason: '/v1/models answered HTTP 404' },
      weights_check: { kind: 'refused', expected: DIGEST, detail: 'model_artifact_mismatch: other bytes' },
    }),
  }))
  assert.equal(refused.left.weightsDigest.kind, 'unavailable')
  assert.equal(refused.left.weightsCheck.kind, 'refused')
  assert.equal(refused.left.weightsCheck.expected, DIGEST)
  const legacy = describeComparison(body())
  assert.equal(legacy.left.weightsDigest.reported, false, 'an older proxy read no digest; that is not "unpublished"')
  assert.equal(legacy.left.weightsCheck, null)
})

check('a proxy that records no identity is never read as "same id"', () => {
  const legacy = describeComparison(body())
  assert.equal(legacy.modelIdentity, null)
  assert.equal(identityStatement(legacy).kind, 'not_reported')
  assert.equal(identityStatement(describeComparison(body({ model_identity: 'digest_match' }))).kind, 'unrecognised',
    'a future identity kind must not read as one this build knows')
})

check('an older proxy\'s unverified identity is never described as a missing engine parameter', () => {
  const statements = uncontrolledStatements(describeComparison(body({ uncontrolled: ['seed', 'model identity'] })))
  const byName = Object.fromEntries(statements.map((item) => [item.name, item.text]))
  assert.doesNotMatch(byName['model identity'], /parameter/, 'no engine has a "model identity" parameter to lack')
  assert.match(byName['model identity'], /did not say why/, 'an unstated reason is not invented')
  assert.match(byName.seed, /no such parameter/)
})

check('every uncontrolled item carries the reason the proxy gave it, not a shared one', () => {
  const comparison = describeComparison(body({
    uncontrolled: ['seed', 'model identity'],
    uncontrolled_detail: [
      { name: 'seed', reason: 'desk (lmstudio) runs an engine whose documented completion API has no seed parameter, so its runs were sent none' },
      { name: 'model identity', reason: 'the operator declared `a:latest` and `a` to be the same weights, and nothing here checked it' },
    ],
  }))
  const statements = uncontrolledStatements(comparison)
  assert.deepEqual(statements.map((item) => item.name), ['seed', 'model identity'])
  assert.match(statements[0].text, /desk \(lmstudio\).*no seed parameter/)
  assert.match(statements[1].text, /operator declared/)
  assert.notEqual(statements[0].text, statements[1].text)
  assert.ok(!comparisonCaveats(comparison).some((c) => /seed|model identity/.test(c)),
    'each item is said once, with its reason, not again as a caveat')
})

check('an item the proxy gave no reason for says so, never borrowing another item\'s', () => {
  const statements = uncontrolledStatements(describeComparison(body({
    uncontrolled: ['seed', 'request history'],
    uncontrolled_detail: [{ name: 'seed', reason: 'x lacks it' }, { name: 'request history' }],
  })))
  assert.deepEqual(statements, [
    { name: 'seed', text: 'x lacks it' },
    { name: 'request history', text: 'this proxy did not say why.' },
  ])
})

check('an absent uncontrolled list is "not reported", never "nothing uncontrolled"', () => {
  const legacy = { ...body() }
  delete legacy.uncontrolled
  assert.equal(describeComparison(legacy).uncontrolledReported, false)
  assert.equal(describeComparison(body()).uncontrolledReported, true)
})

/* ---- what the bytes cover ---- */

check('a line ending is carried per line, so a terminator-only difference can be shown', () => {
  const comparison = describeComparison(body({
    diff: {
      kind: 'lines',
      lines: [
        { op: 'removed', text: '12', eol: 'crlf' },
        { op: 'added', text: '12', eol: 'none' },
        { op: 'same', text: 'x', eol: 'lf' },
        { op: 'same', text: 'y' },
        { op: 'same', text: 'z', eol: 'cr' },
      ],
    },
  }))
  const lines = comparison.diff.lines
  assert.deepEqual(lines.map((line) => line.eol), ['crlf', 'none', 'lf', null, null])
  assert.equal(lines[0].text, lines[1].text, 'the text is identical; only the terminator differs')
  assert.equal(lines[3].eolUnrecognised, false, 'an older proxy that sends no eol renders as before')
  assert.equal(lines[4].eolUnrecognised, true, 'an eol this build does not know is unknown, not LF')
})

check('the token cap is stated, and a clamped request says what was applied', () => {
  assert.match(tokenCapStatement(describeComparison(body())), /capped at 64 tokens, so this compares at most the first 64 tokens/)
  const clamped = tokenCapStatement(describeComparison(body({ plan: { temperature: 0, seed: 0, max_tokens: 1024, repetitions: 2 } })), 5000)
  assert.match(clamped, /5000 were asked for; the proxy applied 1024/)
  assert.match(tokenCapStatement(describeComparison(body({ plan: { max_tokens: 1 } }))), /capped at 1 token,/)
  assert.equal(tokenCapStatement(describeComparison(body({ plan: {} }))), null, 'an unreported cap is unknown, not 64')
})

check('a node answering under another model name is flagged; absent and unnamed are not', () => {
  const named = describeComparison(body({ right: side({ label: 'desk', reported_model: 'llama-3.2-3b-instruct' }) }))
  assert.equal(reportedModelMismatch(named.right), true)
  assert.ok(comparisonCaveats(named).some((c) => /desk answered as llama-3\.2-3b-instruct, not the requested llama-3\.2-1b/.test(c)))
  const same = describeComparison(body({ right: side({ reported_model: 'llama-3.2-1b' }) }))
  assert.equal(reportedModelMismatch(same.right), false)
  const unnamed = describeComparison(body({ right: side({ reported_model: null }) }))
  assert.equal(unnamed.right.reported.state, 'unnamed')
  assert.equal(reportedModelMismatch(unnamed.right), false)
  assert.equal(describeComparison(body()).right.reported.state, 'not_relayed')
})

/* ---- the request ---- */

const LEFT = { node: 'studio', model: 'llama-3.2-1b-instruct:latest' }

check('per-side ids are sent only when the operator paired two different names', () => {
  // Any per-side id makes the proxy skip its alias table, so a same-name
  // request that carried them would be refused for a name only an alias knows.
  const same = buildCompareRequest({ left: LEFT, right: { node: 'desk', model: LEFT.model }, prompt: 'p', repetitions: '2', maxTokens: '64' })
  assert.equal('left_model' in same, false)
  assert.equal('right_model' in same, false)
  assert.equal(same.model, LEFT.model)
  const paired = buildCompareRequest({ left: LEFT, right: { node: 'desk', model: 'llama-3.2-1b-instruct' }, prompt: 'p', repetitions: '2', maxTokens: '64' })
  assert.equal(paired.left_model, 'llama-3.2-1b-instruct:latest')
  assert.equal(paired.right_model, 'llama-3.2-1b-instruct')
})

check('the token cap and run count are always sent, bounded as the proxy bounds them', () => {
  const cap = (maxTokens) => buildCompareRequest({ left: LEFT, right: LEFT, prompt: 'p', repetitions: '2', maxTokens }).max_tokens
  assert.equal(cap('64'), 64)
  assert.equal(cap('200'), 200)
  assert.equal(cap('0'), 1)
  assert.equal(cap('5000'), 1024)
  assert.equal(cap(''), 64, 'an emptied field falls back to the default, never to 0')
  assert.equal(cap('abc'), 64)
  assert.equal(buildCompareRequest({ left: LEFT, right: LEFT, prompt: 'p', repetitions: '9', maxTokens: '64' }).repetitions, 5)
})

/* ---- the transport: every failure is named, none is an empty comparison ---- */

const json = (status, value) => new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } })
const ask = (overrides) => requestComparison({ base: 'http://127.0.0.1:8282', request: { left: 'a' }, ...overrides })

await checkAsync('a client key is sent as a bearer token, and only when one was entered', async () => {
  const seen = []
  const capture = (url, init) => { seen.push(init.headers.authorization ?? null); return Promise.resolve(json(200, body())) }
  await ask({ fetchImpl: capture })
  await ask({ fetchImpl: capture, clientKey: '  k-9f3a  ' })
  assert.deepEqual(seen, [null, 'Bearer k-9f3a'])
})

await checkAsync('a 401 says a key is needed, or that the key sent was not accepted', async () => {
  const deny = () => Promise.resolve(json(401, { error: { message: 'unauthorized' } }))
  assert.equal((await ask({ fetchImpl: deny })).problem.code, 'key_required')
  assert.equal((await ask({ fetchImpl: deny, clientKey: 'wrong' })).problem.code, 'key_refused')
})

await checkAsync('a hung proxy times out with its own answer, and leaving the page is not a failure', async () => {
  const hang = (url, init) => new Promise((_, reject) => {
    init.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
  })
  const timedOut = await ask({ fetchImpl: hang, timeoutMs: 30 })
  assert.equal(timedOut.problem.code, 'timeout')
  assert.equal(timedOut.problem.timeoutMs, 30)

  const left = new AbortController()
  const pending = ask({ fetchImpl: hang, signal: left.signal })
  left.abort()
  assert.equal((await pending).problem.code, 'cancelled', 'an unmount is not reported as the proxy failing')
})

await checkAsync('a network failure, a refusal and a non-comparison are each named', async () => {
  const offline = await ask({ fetchImpl: () => Promise.reject(new TypeError('Failed to fetch')) })
  assert.equal(offline.problem.code, 'unreachable')
  assert.equal(offline.problem.cause, 'network', 'which from a browser may be a CORS refusal')
  const refused = await ask({ fetchImpl: () => Promise.resolve(json(400, { error: { message: 'studio does not hold x' } })) })
  assert.deepEqual(refused.problem, { code: 'refused', detail: 'studio does not hold x' })
  const notAComparison = await ask({ fetchImpl: () => Promise.resolve(json(200, [])) })
  assert.equal(notAComparison.problem.code, 'malformed')
  assert.equal(notAComparison.comparison, undefined, 'never an empty comparison')
  const ok = await ask({ fetchImpl: () => Promise.resolve(json(200, body())) })
  assert.equal(ok.comparison.verdict.kind, 'divergent')
})

console.log(`\ndivergence model smoke: ${checks} checks passed`)
