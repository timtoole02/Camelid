/* The pure layer of the divergence view.
 *
 * The rules here are the client half of the ones the proxy enforces, and they
 * exist because this screen is the one most able to mislead. A reader looking
 * at two answers side by side will reach for "which is right" on their own; the
 * page's whole job is to keep saying what was and was not established.
 *
 * Three of them are load-bearing:
 *
 *   - A verdict this build does not recognise is UNKNOWN, never rendered as one
 *     of the ones it does know. A future proxy verdict must not silently read
 *     as agreement.
 *   - `not_attributable` and `different_models` are NOT differences. They are
 *     refusals to conclude, and must never be styled or summarised as "these
 *     two disagree".
 *   - A side that did not agree with itself is reported as such before any
 *     comparison is offered, because that fact invalidates the comparison.
 */

const KNOWN_VERDICTS = ['identical', 'divergent', 'different_models', 'not_attributable']
const KNOWN_STABILITY = ['stable', 'unstable', 'unmeasured']
const KNOWN_TEMPLATE = ['captured', 'not_exposed', 'unavailable']
const KNOWN_DIFF = ['identical', 'lines', 'declined']
const KNOWN_OPS = ['same', 'removed', 'added']
const KNOWN_HONOURED = ['sent', 'unsupported']

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringOrNull(value) {
  return typeof value === 'string' && value.length > 0 ? value : null
}

function numberOrNull(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function oneOf(allowed, value) {
  return typeof value === 'string' && allowed.includes(value) ? value : null
}

function describeStability(raw) {
  if (!isPlainObject(raw)) return { kind: null, distinctAnswers: null }
  const kind = oneOf(KNOWN_STABILITY, raw.kind)
  const digests = Array.isArray(raw.digests)
    ? raw.digests.filter((digest) => typeof digest === 'string')
    : null
  return { kind, distinctAnswers: digests ? digests.length : null }
}

function describeTemplate(raw) {
  if (!isPlainObject(raw)) return { kind: null, source: null, template: null, detail: null }
  return {
    kind: oneOf(KNOWN_TEMPLATE, raw.kind),
    source: stringOrNull(raw.source),
    template: stringOrNull(raw.template),
    detail: stringOrNull(raw.detail),
  }
}

function describeSide(raw) {
  if (!isPlainObject(raw)) return null
  const samples = Array.isArray(raw.samples)
    ? raw.samples.filter(isPlainObject).map((sample) => ({
      text: typeof sample.text === 'string' ? sample.text : null,
      sha256: stringOrNull(sample.sha256),
      elapsedMs: numberOrNull(sample.elapsed_ms),
    }))
    : []
  const stability = describeStability(raw.stability)
  const sampling = isPlainObject(raw.applied_sampling) ? raw.applied_sampling : {}

  return {
    label: stringOrNull(raw.label),
    engine: stringOrNull(raw.engine),
    // LM Studio publishes no version; that must read as unknown, not as a gap.
    engineVersion: stringOrNull(raw.engine_version),
    // The inference runtime, where the engine names one. Never shown as the
    // engine's version: they are different facts about different software.
    runtime: stringOrNull(raw.runtime),
    model: stringOrNull(raw.model),
    samples,
    stability,
    // Only a side that agreed with itself has a digest that represents it.
    settledDigest: stability.kind === 'stable' && samples.length > 0 ? samples[0].sha256 : null,
    appliedSampling: {
      temperature: oneOf(KNOWN_HONOURED, sampling.temperature),
      seed: oneOf(KNOWN_HONOURED, sampling.seed),
    },
    template: describeTemplate(raw.template),
  }
}

function describeDiff(raw) {
  if (!isPlainObject(raw)) return { kind: null, lines: null, reason: null, changedLines: null }
  const kind = oneOf(KNOWN_DIFF, raw.kind)
  const lines = Array.isArray(raw.lines)
    ? raw.lines.filter(isPlainObject).map((line) => ({
      op: oneOf(KNOWN_OPS, line.op),
      text: typeof line.text === 'string' ? line.text : '',
    }))
    : null
  return {
    kind,
    lines: kind === 'lines' ? lines : null,
    reason: stringOrNull(raw.reason),
    changedLines: kind === 'lines' && lines ? lines.filter((line) => line.op !== 'same').length : null,
  }
}

/** Turn a proxy comparison into what the page renders. */
export function describeComparison(body) {
  if (!isPlainObject(body)) return null
  const verdict = isPlainObject(body.verdict) ? body.verdict : {}
  const plan = isPlainObject(body.plan) ? body.plan : {}

  return {
    prompt: typeof body.prompt === 'string' ? body.prompt : null,
    promptSha256: stringOrNull(body.prompt_sha256),
    plan: {
      temperature: numberOrNull(plan.temperature),
      seed: numberOrNull(plan.seed),
      maxTokens: numberOrNull(plan.max_tokens),
      repetitions: numberOrNull(plan.repetitions),
    },
    verdict: {
      kind: oneOf(KNOWN_VERDICTS, verdict.kind),
      reason: stringOrNull(verdict.reason),
      leftModel: stringOrNull(verdict.left),
      rightModel: stringOrNull(verdict.right),
    },
    left: describeSide(body.left),
    right: describeSide(body.right),
    diff: describeDiff(body.diff),
    uncontrolled: Array.isArray(body.uncontrolled)
      ? body.uncontrolled.filter((name) => typeof name === 'string')
      : [],
  }
}

/* Whether this comparison established anything about the engines.
 *
 * False for an unrecognised verdict: a build that treated "something new" as
 * "attributable" would put a claim on screen it cannot support. */
export function isAttributable(comparison) {
  const kind = comparison?.verdict?.kind
  return kind === 'identical' || kind === 'divergent'
}

/** The one-line headline. Never ranks the two sides. */
export function verdictHeadline(comparison) {
  const verdict = comparison?.verdict
  switch (verdict?.kind) {
    case 'identical':
      return 'Both nodes returned the same bytes.'
    case 'divergent':
      return 'Both nodes agreed with themselves, and disagreed with each other.'
    case 'different_models':
      return verdict.leftModel && verdict.rightModel
        ? `Not comparable: these nodes are serving different models (${verdict.leftModel} and ${verdict.rightModel}).`
        : 'Not comparable: these nodes are serving different models.'
    case 'not_attributable':
      return verdict.reason
        ? `Nothing can be concluded: ${verdict.reason}.`
        : 'Nothing can be concluded from this comparison.'
    default:
      return 'This proxy reported a verdict this build does not recognise, so nothing is concluded.'
  }
}

/* What still has to be said even when a verdict was reached: controls that did
 * not reach an engine, and sides that never proved they repeat themselves. */
export function comparisonCaveats(comparison) {
  if (!comparison) return []
  const caveats = []

  for (const name of comparison.uncontrolled) {
    caveats.push(`${name} was not controlled: at least one engine has no such parameter.`)
  }
  for (const side of [comparison.left, comparison.right]) {
    if (!side) continue
    if (side.stability.kind === 'unmeasured') {
      caveats.push(`${side.label || 'a node'} was run once, so it was never shown to repeat itself.`)
    }
    if (side.stability.kind === 'unstable') {
      const count = side.stability.distinctAnswers
      caveats.push(
        `${side.label || 'a node'} did not repeat itself${count ? ` (${count} distinct answers)` : ''}.`,
      )
    }
    if (side.template.kind === 'not_exposed') {
      caveats.push(`${side.label || 'a node'} runs an engine that exposes no prompt template.`)
    }
    if (side.template.kind === 'unavailable') {
      caveats.push(`${side.label || 'a node'} did not return its template.`)
    }
  }
  return caveats
}

/* The two templates, when both were captured and actually differ. This is the
 * explanation the whole screen exists to surface, so it is only offered when
 * there really are two of them to compare. */
export function templateDivergence(comparison) {
  const left = comparison?.left?.template
  const right = comparison?.right?.template
  if (left?.kind !== 'captured' || right?.kind !== 'captured') return null
  if (left.template === right.template) return { differ: false, left, right }
  return { differ: true, left, right }
}

/* What a node can be asked about, for the model picker.
 *
 * Four outcomes, and collapsing any of them into "no models" would be a lie:
 * a proxy that withholds its node detail is not a fabric of empty nodes, and a
 * node that is not ready has no list rather than an empty one. */
export function modelChoices(fabric, label) {
  if (!label) return { kind: 'no_node' }
  // The proxy discloses node detail only on loopback; absent is not empty.
  if (!fabric || fabric.nodes === null) return { kind: 'withheld' }
  const node = fabric.nodes.find((entry) => entry.label === label)
  if (!node) return { kind: 'no_node' }
  if (node.state !== 'ready') return { kind: 'not_ready', reason: node.reason }
  if (!Array.isArray(node.models)) return { kind: 'unknown' }
  return { kind: 'listed', models: node.models, engine: node.engine }
}

/* Whether a typed model looks absent from what the node reported.
 *
 * A warning, never a gate: the proxy is the authority on what a node holds and
 * refuses by name, and blocking here on a stale list would refuse a model the
 * node really has. */
export function modelLooksAbsent(choices, model) {
  if (!model || choices?.kind !== 'listed') return false
  return !choices.models.includes(model)
}

/* Whether running this comparison means asserting two names are the same
 * weights. Engines do not agree on naming and none publishes a comparable
 * digest, so this is a claim only a person can make. */
export function needsIdentityAssertion(leftModel, rightModel) {
  return Boolean(leftModel) && Boolean(rightModel) && leftModel !== rightModel
}
