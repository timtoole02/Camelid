/* Token inspection contract lane.

   The engine can report, for each generated token, the log-probability the model
   assigned it plus the top-N alternatives it outranked. This module owns every
   decision about that data: whether the contract permits asking for it, what
   request fields to send, how to normalize a response, and what may honestly be
   said about the result. It performs no I/O and touches no storage.

   Three rules shape everything here.

   1. CAPTURED, NEVER RECONSTRUCTED. The distribution is collected during the
      decode that produced the visible reply, by sending the inspection fields on
      the original request. Re-running a turn afterwards would produce a DIFFERENT
      generation whose numbers describe a reply the reader never saw — and on a
      sampled row it would not even reproduce the text. A surface that cannot make
      that false claim is stronger than one that discloses it.

   2. THE CONTRACT DECIDES, AND IT IS INCOMPLETE. `/api/capabilities` ships
      `api_conformance` rows carrying machine-readable supported/unsupported modes.
      Those cover the ROUTE axis (streaming, multi-choice) and say nothing about the
      SERVE-LANE axis: several non-dense lanes short-circuit before the logprobs
      handler and answer 200 with the key simply absent. So the conformance row is
      the pre-flight and the presence of the key in the response is the correctness
      guard. Never infer certainty from missing data.

   3. RAW MODEL DISTRIBUTION, NOT SAMPLING PROBABILITY. The values are a full-vocab
      log-softmax over the unmodified logits — no temperature, no penalties, no
      grammar mask. They describe what the model computed, not how a token was
      picked. Copy in this lane never uses the words "confidence" or "certainty";
      both would assert something about the model's epistemic state that a softmax
      over logits does not license. */

import { isSupportedCapabilityStatus, displayCapabilityCopy } from './capabilities.js'

/* Engine caps, mirrored from the API's own validation so the UI cannot compose a
   request the backend will reject. src/api/mod.rs MAX_LOGPROBS / MAX_N_CHOICES. */
export const MAX_TOP_LOGPROBS = 20
export const MAX_N_CHOICES = 8
export const DEFAULT_TOP_LOGPROBS = 5
export const TOP_LOGPROBS_CHOICES = [3, 5, 10, 20]

const LOGPROBS_ROW_ID = 'rich_logprobs'
const MULTI_CHOICE_ROW_ID = 'multi_choice_generation'

function findContractRow(rows, id) {
  return (rows || []).find((row) => row?.id === id) || null
}

/* Resolve a feature row by EXACT id across both projections the contract ships.

   `api_conformance` is preferred: it carries the machine-readable mode lists, and
   its projection emits no `notes` string, so it cannot smuggle vendor names into
   rendered copy. `api_features` is the fallback for an engine build that predates
   the conformance array — there the modes are unknown, and unknown means closed. */
export function readInspectionContract(capabilities) {
  const conformance = findContractRow(capabilities?.api_conformance, LOGPROBS_ROW_ID)
  const feature = findContractRow(capabilities?.api_features, LOGPROBS_ROW_ID)
  const row = conformance || feature
  if (!row) {
    return {
      present: false,
      supported: false,
      modesKnown: false,
      nonStreamingSupported: false,
      rowId: LOGPROBS_ROW_ID,
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
    /* Absent mode lists fail closed: an engine that does not describe its modes
       does not get assumed into supporting the one we want. */
    nonStreamingSupported: supported && modesKnown && supportedModes.includes('chat_nonstreaming'),
    rowId: LOGPROBS_ROW_ID,
    status: row.status || null,
    note: feature?.notes ? displayCapabilityCopy(feature.notes) : null,
  }
}

/* The n>1 lane, resolved the same way. It is read only to explain why the
   candidates surface is guarded — nothing in this file ever requests n>1. */
export function readCandidatesContract(capabilities) {
  const conformance = findContractRow(capabilities?.api_conformance, MULTI_CHOICE_ROW_ID)
  const feature = findContractRow(capabilities?.api_features, MULTI_CHOICE_ROW_ID)
  const row = conformance || feature
  const unsupported = Array.isArray(conformance?.unsupported_modes) ? conformance.unsupported_modes : []
  return {
    present: Boolean(row),
    supported: Boolean(row) && isSupportedCapabilityStatus(String(row.status || '')),
    excludesLogprobs: unsupported.includes('logprobs'),
    rowId: MULTI_CHOICE_ROW_ID,
    status: row?.status || null,
  }
}

export function clampTopLogprobs(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return DEFAULT_TOP_LOGPROBS
  return Math.min(MAX_TOP_LOGPROBS, Math.max(1, Math.round(n)))
}

/* Request fields for an inspected turn.

   Returns {} whenever inspection is off or the contract does not permit it, so a
   caller can spread this unconditionally and a guarded contract simply contributes
   nothing. `top_logprobs` is never emitted without `logprobs: true` — the engine
   rejects that pairing outright — and the depth is clamped to the engine's cap. */
export function inspectionRequestFields({ enabled, contract, topLogprobs = DEFAULT_TOP_LOGPROBS } = {}) {
  if (!enabled) return {}
  if (!contract?.nonStreamingSupported) return {}
  return { logprobs: true, top_logprobs: clampTopLogprobs(topLogprobs) }
}

/* Inspection forces the reply off the streaming path: the engine rejects
   logprobs with stream:true. Callers use this to keep one source of truth for
   "is this turn streaming" rather than re-deriving the condition. */
export function inspectionForcesNonStreaming({ enabled, contract } = {}) {
  return Boolean(enabled && contract?.nonStreamingSupported)
}

const clampProbability = (p) => (Number.isFinite(p) ? Math.min(1, Math.max(0, p)) : null)

function probabilityFromLogprob(logprob) {
  const value = Number(logprob)
  if (!Number.isFinite(value)) return null
  return clampProbability(Math.exp(value))
}

/* Display text for one token.

   `bytes` is the ground truth. A BPE token can be a fragment of a multi-byte
   character, whitespace, or a control marker, and the `token` string is a lossy
   rendering of those bytes. So the glyph shown is an AFFORDANCE and the byte array
   travels with it for anything that needs the truth. `substituted` tells the
   renderer when what is on screen is not literally the token text. */
export function describeToken(entry) {
  const raw = typeof entry?.token === 'string' ? entry.token : ''
  const bytes = Array.isArray(entry?.bytes) ? entry.bytes : []
  if (raw === '') return { display: '∅', substituted: true, kind: 'empty', raw, bytes }
  if (/^<\|.+\|>$/.test(raw) || /^<\/?s>$/.test(raw)) {
    return { display: raw, substituted: false, kind: 'special', raw, bytes }
  }
  if (/^\s+$/.test(raw)) {
    const display = raw.replace(/\n/g, '⏎').replace(/\t/g, '⇥').replace(/ /g, '·')
    return { display, substituted: true, kind: 'whitespace', raw, bytes }
  }
  const display = raw.replace(/^ /, '·').replace(/\n/g, '⏎').replace(/\t/g, '⇥')
  return { display, substituted: display !== raw, kind: 'text', raw, bytes }
}

/* Three bands, chosen so the SALIENCE BUDGET goes to the minority worth reading.

   A greedy reply is overwhelmingly high-probability; encoding probability directly
   would ink the boring 90% and fade the interesting 5% to nothing. These bands
   invert that: 'settled' gets no marking at all, and only contested positions
   carry weight. The thresholds are presentation, not a claim about the model. */
export function probabilityBand(probability) {
  if (probability === null) return 'unknown'
  if (probability >= 0.9) return 'settled'
  if (probability >= 0.5) return 'leading'
  return 'contested'
}

/* Near-ties must not be rendered as a strict ordering.

   At f32 precision, and with lane-to-lane numeric variance around 0.007 nats, a
   0.001-nat gap between two alternatives is not a meaningful ranking. Anything
   inside this window is marked as tied so the UI can say so rather than implying
   the order is information. */
const TIE_WINDOW_NATS = 0.01

export function alternativesFor(entry) {
  const list = Array.isArray(entry?.top_logprobs) ? entry.top_logprobs : []
  return list.map((alt, index) => {
    const probability = probabilityFromLogprob(alt?.logprob)
    const previous = index > 0 ? Number(list[index - 1]?.logprob) : null
    const tiedWithPrevious = previous !== null
      && Number.isFinite(Number(alt?.logprob))
      && Math.abs(Number(alt.logprob) - previous) <= TIE_WINDOW_NATS
    return {
      ...describeToken(alt),
      logprob: Number.isFinite(Number(alt?.logprob)) ? Number(alt.logprob) : null,
      probability,
      rank: index + 1,
      tiedWithPrevious,
    }
  })
}

/* Normalize one `choices[].logprobs` object into the shape the panel renders.

   Guards the two structural cases the wire permits but naive code assumes away:
   an empty/malformed content array, and a chosen token that does not appear in its
   own top-N list (possible whenever the emitted token was not among the k returned
   alternatives). The second is reported rather than hidden — a chosen token outside
   the top-N is real signal about the decode, not a rendering nuisance. */
export function normalizeInspection(logprobs) {
  const content = Array.isArray(logprobs?.content) ? logprobs.content : []
  if (!content.length) return null

  const tokens = content.map((entry, index) => {
    const logprob = Number.isFinite(Number(entry?.logprob)) ? Number(entry.logprob) : null
    const probability = probabilityFromLogprob(logprob)
    const alternatives = alternativesFor(entry)
    const top = alternatives[0] || null
    /* Identity by rank position and value, not by token string: two distinct
       token ids can decode to the same text. */
    const chosenIsTop = top !== null && logprob !== null && Math.abs(top.logprob - logprob) <= TIE_WINDOW_NATS
    const chosenInAlternatives = alternatives.some((alt) => (
      alt.logprob !== null && logprob !== null && Math.abs(alt.logprob - logprob) <= TIE_WINDOW_NATS && alt.raw === (entry?.token ?? '')
    ))
    /* Residual mass: what the shown alternatives do NOT account for. Without it,
       k rows read as a closed set — the single most likely misreading of this
       surface. Clamped at zero because a truncated set can round above 1.0. */
    const shownMass = alternatives.reduce((sum, alt) => sum + (alt.probability ?? 0), 0)
    return {
      index,
      ...describeToken(entry),
      logprob,
      probability,
      band: probabilityBand(probability),
      alternatives,
      chosenIsTop,
      chosenInAlternatives,
      shownMass: clampProbability(shownMass),
      residualMass: clampProbability(1 - shownMass),
    }
  })

  const scored = tokens.filter((token) => token.logprob !== null)
  const sumLogprob = scored.reduce((sum, token) => sum + token.logprob, 0)
  const meanLogprob = scored.length ? sumLogprob / scored.length : null
  const contested = tokens.filter((token) => token.band === 'contested')
  const offTop = tokens.filter((token) => token.logprob !== null && !token.chosenIsTop)
  const lowest = scored.length
    ? scored.reduce((carry, token) => (token.logprob < carry.logprob ? token : carry), scored[0])
    : null

  return {
    tokens,
    stats: {
      tokenCount: tokens.length,
      scoredCount: scored.length,
      depth: tokens[0]?.alternatives?.length ?? 0,
      sumLogprob: scored.length ? sumLogprob : null,
      meanLogprob,
      /* Perplexity over the generated tokens only — not a model-quality score and
         not comparable across prompts. Reported because it is the one number that
         summarizes a whole reply, and it is derived from data already in hand. */
      perplexity: meanLogprob === null ? null : Math.exp(-meanLogprob),
      contestedCount: contested.length,
      offTopCount: offTop.length,
      lowestProbabilityIndex: lowest ? lowest.index : null,
    },
    contested: contested.slice(0, 5).map((token) => token.index),
  }
}

/* Why a reply carries no inspection data.

   The critical distinction is between "not asked for" and "asked for, answered
   200, and the key was absent" — the second is a serve lane that short-circuits
   before the logprobs handler. Reporting that as an empty panel, or worse as
   high probability, would attribute a lane's silence to the model. */
export function inspectionAbsenceReason({ requested, responded, hasLogprobs, streamed }) {
  if (!requested) return null
  if (!responded) return null
  if (hasLogprobs) return null
  if (streamed) {
    return {
      code: 'streamed',
      title: 'This reply arrived as a stream',
      detail: 'Probabilities are only reported on a non-streaming reply. Something between this page and the engine converted the response to a stream, so the engine had nothing to attach them to.',
    }
  }
  return {
    code: 'lane_absent',
    title: 'This execution lane did not report probabilities',
    detail: 'The reply completed normally and the engine returned no probability record for it. Several execution lanes answer chat requests before the probability step runs. This says nothing about how the model ranked these tokens — the measurement is missing, not flat.',
  }
}

/* Formatters. Precision is deliberately coarse: lane-to-lane numeric variance on
   the same model and prompt runs to a few thousandths of a nat, so printing four
   decimals would advertise precision the value does not carry. */
export function formatProbability(probability) {
  if (probability === null || probability === undefined) return '—'
  if (probability >= 0.9995) return '>99.9%'
  if (probability < 0.001) return '<0.1%'
  return `${(probability * 100).toFixed(1)}%`
}

export function formatLogprob(logprob) {
  if (logprob === null || logprob === undefined) return '—'
  if (logprob > -0.005 && logprob <= 0) return '−0.00'
  return `−${Math.abs(logprob).toFixed(2)}`
}

export function formatPerplexity(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value >= 100 ? value.toFixed(0) : value.toFixed(2)
}
