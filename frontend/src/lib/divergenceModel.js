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
const KNOWN_RENDERED = ['captured', 'unavailable']
const KNOWN_DIFF = ['identical', 'lines', 'declined']
const KNOWN_OPS = ['same', 'removed', 'added']
const KNOWN_HONOURED = ['sent', 'unsupported']
const KNOWN_IDENTITY = ['same_id', 'asserted_by_operator']
const KNOWN_EOL = ['lf', 'crlf', 'none']

/** Bounds the proxy clamps to (MAX_COMPARE_* in src/fabric/server.rs). */
export const COMPARE_MAX_TOKENS = { min: 1, max: 1024, fallback: 64 }
export const COMPARE_REPETITIONS = { min: 1, max: 5, fallback: 2 }

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

/* The prompt an engine rendered from the messages the comparison sent. The
 * only evidence of what it applied. A proxy from before the field sent none,
 * which is "not reported", never "not captured". */
function describeRenderedPrompt(raw, present) {
  if (!present) return { kind: null, reported: false, source: null, text: null, reason: null }
  if (!isPlainObject(raw)) return { kind: null, reported: true, source: null, text: null, reason: null }
  return {
    kind: oneOf(KNOWN_RENDERED, raw.kind),
    reported: true,
    source: stringOrNull(raw.source),
    text: typeof raw.text === 'string' ? raw.text : null,
    reason: stringOrNull(raw.reason),
  }
}

/* The model the node's own response named. Three facts, kept apart: an older
 * proxy relays nothing (`not_relayed`), a response can name no model
 * (`unnamed`), or it names one (`named`). */
function describeReportedModel(raw) {
  if (!('reported_model' in raw)) return { state: 'not_relayed', model: null }
  if (raw.reported_model === null) return { state: 'unnamed', model: null }
  const model = stringOrNull(raw.reported_model)
  return model ? { state: 'named', model } : { state: null, model: null }
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
    // The id this side was asked for.
    model: stringOrNull(raw.model),
    reported: describeReportedModel(raw),
    samples,
    stability,
    // Only a side that agreed with itself has a digest that represents it.
    settledDigest: stability.kind === 'stable' && samples.length > 0 ? samples[0].sha256 : null,
    appliedSampling: {
      temperature: oneOf(KNOWN_HONOURED, sampling.temperature),
      seed: oneOf(KNOWN_HONOURED, sampling.seed),
    },
    // What the engine publishes as the model's template, never what it
    // applied. An older proxy sent the same capture as `template`.
    advertisedTemplate: describeTemplate('advertised_template' in raw ? raw.advertised_template : raw.template),
    renderedPrompt: describeRenderedPrompt(raw.rendered_prompt, 'rendered_prompt' in raw),
  }
}

function describeDiff(raw) {
  if (!isPlainObject(raw)) return { kind: null, lines: null, reason: null, changedLines: null }
  const kind = oneOf(KNOWN_DIFF, raw.kind)
  const lines = Array.isArray(raw.lines)
    ? raw.lines.filter(isPlainObject).map((line) => ({
      op: oneOf(KNOWN_OPS, line.op),
      text: typeof line.text === 'string' ? line.text : '',
      // How the line ended. The text never carries its terminator, so without
      // this two answers differing only in CRLF against LF, or in a final
      // newline, would render as two identical lines. Absent on older proxies.
      eol: oneOf(KNOWN_EOL, line.eol),
      eolUnrecognised: line.eol !== undefined && oneOf(KNOWN_EOL, line.eol) === null,
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
    // How the proxy judged the two sides to be the same weights. Absent from a
    // proxy that predates the field, which is not the same as "same id".
    modelIdentity: oneOf(KNOWN_IDENTITY, body.model_identity),
    modelIdentityReported: body.model_identity !== undefined,
    uncontrolled: Array.isArray(body.uncontrolled)
      ? body.uncontrolled.filter((name) => typeof name === 'string')
      : [],
    // An absent list is "not reported", which must not render as "nothing".
    uncontrolledReported: Array.isArray(body.uncontrolled),
    // Why each item is listed. Absent from an older proxy, which named the
    // items without saying why.
    uncontrolledDetail: Array.isArray(body.uncontrolled_detail)
      ? body.uncontrolled_detail
        .filter(isPlainObject)
        .map((item) => ({ name: stringOrNull(item.name), reason: stringOrNull(item.reason) }))
        .filter((item) => item.name)
      : null,
  }
}

/* One sentence per uncontrolled item, each with the reason the proxy gave.
 *
 * An older proxy gave none. For its seed and temperature the only reason it
 * ever had was an engine lacking the parameter; for anything else — model
 * identity is not a parameter — this build does not invent one. */
export function uncontrolledStatements(comparison) {
  if (!comparison) return []
  if (comparison.uncontrolledDetail) {
    return comparison.uncontrolledDetail.map((item) => ({
      name: item.name,
      text: item.reason || 'this proxy did not say why.',
    }))
  }
  return comparison.uncontrolled.map((name) => ({
    name,
    text: name === 'seed' || name === 'temperature'
      ? 'at least one engine has no such parameter.'
      : 'this proxy did not say why.',
  }))
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

/* The model-identity claim this result rests on, in words.
 *
 * An operator's assertion travels with every result that depends on it, so it
 * is stated on the result itself rather than left in a checkbox the reader
 * never saw. */
export function identityStatement(comparison) {
  const leftId = comparison?.left?.model
  const rightId = comparison?.right?.model
  switch (comparison?.modelIdentity) {
    case 'asserted_by_operator':
      return {
        kind: 'asserted_by_operator',
        text: `Asserted by the operator, not verified: ${leftId || 'the left id'} and ${rightId || 'the right id'} `
          + 'were declared to be the same weights. This result rests on that claim.',
      }
    case 'same_id':
      return {
        kind: 'same_id',
        text: `Both sides were asked for the same id${leftId ? `, ${leftId}` : ''}. The id is all that was `
          + 'compared: no engine publishes a digest this fabric can check.',
      }
    default:
      return comparison?.modelIdentityReported
        ? { kind: 'unrecognised', text: 'This proxy recorded a model identity this build does not recognise.' }
        : { kind: 'not_reported', text: 'This proxy did not record how model identity was established.' }
  }
}

/* How much of each answer the comparison covers. "Same bytes" under a 64-token
 * cap is a claim about a 64-token prefix, and has to say so. */
export function tokenCapStatement(comparison, requestedMaxTokens = null) {
  const cap = comparison?.plan?.maxTokens
  if (cap === null || cap === undefined) return null
  const tokens = `${cap} token${cap === 1 ? '' : 's'}`
  let text = `Each answer was capped at ${tokens}, so this compares at most the first ${tokens} of each.`
  if (typeof requestedMaxTokens === 'number' && requestedMaxTokens !== cap) {
    text += ` ${requestedMaxTokens} were asked for; the proxy applied ${cap}.`
  }
  return text
}

/* Whether a node answered under a different model name than the one it was
 * asked for. Only a named, differing model counts: absent and unnamed are
 * reported separately and are not evidence of a swap. */
export function reportedModelMismatch(side) {
  return Boolean(
    side
    && side.reported?.state === 'named'
    && side.model
    && side.reported.model !== side.model,
  )
}

/* What still has to be said about each side even when a verdict was reached:
 * a node answering under another name, sides that never proved they repeat
 * themselves, templates that could not be read. What was not controlled is
 * listed once, with its reasons, by `uncontrolledStatements`. */
export function comparisonCaveats(comparison) {
  if (!comparison) return []
  const caveats = []

  for (const side of [comparison.left, comparison.right]) {
    if (!side) continue
    if (reportedModelMismatch(side)) {
      caveats.push(`${side.label || 'a node'} answered as ${side.reported.model}, not the requested ${side.model}.`)
    }
    if (side.stability.kind === 'unmeasured') {
      caveats.push(`${side.label || 'a node'} was run once, so it was never shown to repeat itself.`)
    }
    if (side.stability.kind === 'unstable') {
      const count = side.stability.distinctAnswers
      caveats.push(
        `${side.label || 'a node'} did not repeat itself${count ? ` (${count} distinct answers)` : ''}.`,
      )
    }
    if (side.advertisedTemplate.kind === 'not_exposed') {
      caveats.push(`${side.label || 'a node'} runs an engine that exposes no prompt template.`)
    }
    if (side.advertisedTemplate.kind === 'unavailable') {
      caveats.push(`${side.label || 'a node'} did not return its advertised template.`)
    }
  }
  return caveats
}

/* The two advertised templates, when both were captured. `unexplained` marks
 * the case a reader most easily gets wrong: the same template on both sides
 * beside an established difference, which that template cannot explain. */
export function templateDivergence(comparison) {
  const left = comparison?.left?.advertisedTemplate
  const right = comparison?.right?.advertisedTemplate
  if (left?.kind !== 'captured' || right?.kind !== 'captured') return null
  const differ = left.template !== right.template
  return { differ, unexplained: !differ && comparison?.verdict?.kind === 'divergent', left, right }
}

/* What may be said about the advertised templates. An advertised template is
 * what an engine publishes, not the prompt it built, so none of these calls it
 * the template applied. */
export function advertisedTemplateNote(comparison) {
  const templates = templateDivergence(comparison)
  if (!templates) return null
  if (templates.unexplained) {
    return 'Both nodes advertise byte-identical chat templates, so the advertised template does not explain this difference.'
  }
  if (templates.differ) {
    return 'The two nodes advertise different chat templates. An advertised template is not proof of the prompt an '
      + 'engine built; only a rendered prompt shows that.'
  }
  return 'Both nodes advertise the same chat template.'
}

/* The two rendered prompts, when both were captured. */
export function renderedPromptComparison(comparison) {
  const left = comparison?.left?.renderedPrompt
  const right = comparison?.right?.renderedPrompt
  if (left?.kind !== 'captured' || right?.kind !== 'captured') return null
  return { same: left.text === right.text }
}

/* What a node can be asked about, for the model picker.
 *
 * Five outcomes, and collapsing any of them into "no models" would be a lie:
 * a fabric we could not read is not one that withholds its detail, a proxy that
 * withholds its node detail is not a fabric of empty nodes, and a node that is
 * not ready has no list rather than an empty one. */
export function modelChoices(fabric, label) {
  if (!label) return { kind: 'no_node' }
  // Not read yet, or the read failed: there is no list because we could not look.
  if (!fabric || fabric.problem) return { kind: 'unread' }
  // The proxy discloses node detail only on loopback; absent is not empty.
  if (fabric.nodes === null) return { kind: 'withheld' }
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

function boundedInteger(value, { min, max, fallback }) {
  if (value === null || value === undefined || String(value).trim() === '') return fallback
  const number = Number(value)
  if (!Number.isFinite(number)) return fallback
  return Math.min(max, Math.max(min, Math.round(number)))
}

/* The body `POST /v1/fabric/compare` receives.
 *
 * Per-side ids are sent only when the operator paired two different names. The
 * proxy treats any per-side id as an explicit override and skips its alias
 * table, so sending both on every request would ask a node for a name it may
 * only know under an alias, and be refused for no visible reason.
 *
 * `max_tokens` is always sent, so the cap the result reports is the one the
 * operator chose rather than a default they never saw. */
export function buildCompareRequest({ left, right, prompt, repetitions, maxTokens }) {
  const request = {
    left: left.node,
    right: right.node,
    model: left.model,
    prompt,
    repetitions: boundedInteger(repetitions, COMPARE_REPETITIONS),
    max_tokens: boundedInteger(maxTokens, COMPARE_MAX_TOKENS),
    temperature: 0,
  }
  if (needsIdentityAssertion(left.model, right.model)) {
    request.left_model = left.model
    request.right_model = right.model
  }
  return request
}
