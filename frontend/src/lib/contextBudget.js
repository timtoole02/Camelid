/* Context-window composition for the chat composer's meter.
 *
 * A user cannot see how much of the model's context a conversation has spent,
 * so a long chat looks like it "ran out" for no reason, and the verified-bound
 * caution ("beyond the verified row's tested N-token context — allowed,
 * untested") reads as a hard ceiling rather than the permission it is.
 *
 * This module turns the numbers the composer already computes into one
 * composition, so the meter, the tooltip, and `validateSendBudget` cannot
 * disagree about how full the window is.
 *
 * Two rules mirror the backend exactly (`src/api/mod.rs`, prompt-admission):
 *   1. A response limit is an UPPER BOUND. The backend clamps it to the room
 *      left in the window, so a reservation larger than the remaining room is
 *      shown clamped, never as an overflow.
 *   2. The only hard failure is a prompt that already fills the whole window,
 *      leaving no room to generate — the backend's `context_length_exceeded`.
 *
 * Prompt sizes here are client-side ESTIMATES. Callers must label them as such;
 * exact counts require the loaded model's tokenizer via `/tokenize`.
 */

/* Segment order is render order: the bar reads left-to-right in the same order
   the model receives the prompt, then the reservation, then free space. */
export const CONTEXT_SEGMENT_KEYS = ['system', 'messages', 'images', 'reserved', 'free']

const SEGMENT_LABELS = {
  system: 'system prompt',
  messages: 'this conversation',
  images: 'images',
  reserved: 'held for the reply',
  free: 'still free',
}

/* When the reservation has been clamped it is no longer a chosen size, it is
   simply whatever the conversation has not taken, so calling it "held" implies a
   commitment the user never made. */
const CLAMPED_RESERVE_LABEL = 'left for the reply'

function nonNegativeInteger(value) {
  const n = Number(value)
  if (!Number.isFinite(n) || n <= 0) return 0
  return Math.round(n)
}

function positiveIntegerOrNull(value) {
  const n = Number(value)
  if (!Number.isFinite(n) || n <= 0) return null
  return Math.round(n)
}

/* Percentages are rendered as bar widths, so they must sum to <= 100 and never
   round a non-empty segment down to an invisible zero. */
function percentOf(tokens, total) {
  if (!total || tokens <= 0) return 0
  return (tokens / total) * 100
}

/**
 * Compose the context window into renderable segments.
 *
 * @param {object} input
 * @param {number|null} input.contextLength Model's native window, or null when unknown.
 * @param {number} input.promptTokens Estimated tokens the prompt will occupy, system and images included.
 * @param {number} input.systemTokens Estimated share of `promptTokens` spent on system instructions.
 * @param {number} input.imageTokens Estimated share of `promptTokens` spent on image content.
 * @param {number} input.reservedTokens Configured response limit, before clamping.
 * @param {number|null} input.verifiedBound Largest context size with committed evidence, or null.
 * @param {number|null} input.warnAtPercent Fill level at which the window counts as nearly full.
 * @returns {object|null} Composition, or null when the window size is unknown.
 */
export function composeContextBudget({
  contextLength,
  promptTokens = 0,
  systemTokens = 0,
  imageTokens = 0,
  reservedTokens = 0,
  verifiedBound = null,
  warnAtPercent = null,
} = {}) {
  const total = positiveIntegerOrNull(contextLength)
  if (total === null) return null

  const used = Math.min(nonNegativeInteger(promptTokens), total)
  const room = Math.max(total - used, 0)

  /* Rule 1: the backend clamps the reservation to the remaining room. */
  const requestedReserve = nonNegativeInteger(reservedTokens)
  const reserved = Math.min(requestedReserve, room)
  const reserveClamped = requestedReserve > reserved
  const free = Math.max(room - reserved, 0)

  /* System and image estimates are subsets of the prompt estimate; whatever is
     left over is conversation text. Clamp so a bad caller cannot invent tokens. */
  const system = Math.min(nonNegativeInteger(systemTokens), used)
  const images = Math.min(nonNegativeInteger(imageTokens), used - system)
  const messages = Math.max(used - system - images, 0)

  const tokensByKey = { system, messages, images, reserved, free }
  const segments = CONTEXT_SEGMENT_KEYS
    .map((key) => ({
      key,
      label: key === 'reserved' && reserveClamped ? CLAMPED_RESERVE_LABEL : SEGMENT_LABELS[key],
      tokens: tokensByKey[key],
      percent: percentOf(tokensByKey[key], total),
    }))
    .filter((segment) => segment.tokens > 0)

  const bound = positiveIntegerOrNull(verifiedBound)
  /* A marker at or past the end of the bar tells the user nothing, and a model
     whose whole window is verified should not be decorated as if it were not. */
  const showVerifiedMarker = bound !== null && bound < total

  /* Rule 2: only a prompt with no room left to generate is a hard failure. */
  let level = 'ok'
  if (used >= total) level = 'error'
  else if (reserveClamped) level = 'notice'

  /* The share unavailable to the next message, which is what both the chip and
     the automatic-compaction threshold read.

     A clamped reservation is deliberately NOT counted. Once clamped it equals
     the remaining room by definition, so `used + reserved` is exactly the whole
     window no matter how short the conversation is: every model whose context is
     smaller than the configured reply limit would sit at 100% from its first
     empty message, and the automatic trim would fire on every send. Rule 2 is
     the honest measure of pressure here -- what threatens the conversation is
     the prompt filling the window, not a reply that simply gets shorter. */
  const committed = reserveClamped ? used : used + reserved
  const filledPercent = percentOf(committed, total)
  const warnAt = Number(warnAtPercent)
  const nearLimit = Number.isFinite(warnAt) && filledPercent >= warnAt

  return {
    contextLength: total,
    usedTokens: used,
    reservedTokens: reserved,
    requestedReserveTokens: requestedReserve,
    freeTokens: free,
    reserveClamped,
    committedTokens: committed,
    usedPercent: percentOf(used, total),
    reservedPercent: percentOf(reserved, total),
    freePercent: percentOf(free, total),
    filledPercent,
    nearLimit,
    verifiedBound: bound,
    showVerifiedMarker,
    verifiedPercent: showVerifiedMarker ? percentOf(bound, total) : null,
    beyondVerified: bound !== null && used > bound,
    segments,
    level,
  }
}

/* Compact, locale-aware token counts. Long conversations reach six figures, and
   `490,432` in a 90px chip truncates where `490.4K` does not. */
export function formatTokenCount(value) {
  const tokens = nonNegativeInteger(value)
  if (tokens < 1000) return String(tokens)
  if (tokens < 1_000_000) {
    const thousands = tokens / 1000
    return `${thousands >= 100 ? Math.round(thousands) : thousands.toFixed(1)}K`
  }
  const millions = tokens / 1_000_000
  return `${millions >= 100 ? Math.round(millions) : millions.toFixed(1)}M`
}

export function formatPercent(value) {
  const n = Number(value)
  if (!Number.isFinite(n) || n <= 0) return '0%'
  if (n < 1) return '<1%'
  return `${Math.round(n)}%`
}
