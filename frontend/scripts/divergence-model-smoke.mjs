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
  comparisonCaveats,
  describeComparison,
  isAttributable,
  modelChoices,
  modelLooksAbsent,
  needsIdentityAssertion,
  templateDivergence,
  verdictHeadline,
} from '../src/lib/divergenceModel.js'

let checks = 0
function check(name, fn) {
  fn()
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
    template: { kind: 'captured', source: 'GET /props', template: CAMELID_TEMPLATE },
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
      template: { kind: 'captured', source: 'POST /api/show', template: OLLAMA_TEMPLATE },
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
      template: { kind: 'not_exposed', detail: "LM Studio's documented API exposes no prompt template" },
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
  assert.ok(caveats.some((c) => /seed was not controlled/.test(c)), caveats.join(' | '))
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
    left: side({ stability: { kind: 'probably_stable' }, template: { kind: 'inferred', template: 'x' } }),
  }))
  assert.equal(comparison.left.stability.kind, null)
  assert.equal(comparison.left.template.kind, null)
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
  assert.equal(modelChoices(null, 'studio').kind, 'withheld')
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

console.log(`\ndivergence model smoke: ${checks} checks passed`)
