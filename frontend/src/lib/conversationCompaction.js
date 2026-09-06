/* Send-time conversation compaction.
 *
 * A long chat eventually fills the model's context window, and the backend
 * answers `context_length_exceeded` -- a hard 400 with no reply. Compaction
 * keeps the conversation sendable by trimming what leaves the browser.
 *
 * Two decisions here differ from how hosted assistants do this, on purpose.
 *
 * 1. NOTHING IS SUMMARISED, AND NO MODEL IS CALLED.
 *    This mirrors the agent loop's own contract (`src/chat/agent.rs::compact`,
 *    D-DROVER-1 "the safety spine"): fold the middle by ELISION, never by
 *    generating prose about it. A generated summary can be wrong about a
 *    conversation the user can still scroll up and read, and on a single-decode
 *    -thread engine it would also block the very request it is meant to enable.
 *    Elision is instant, deterministic, and cannot hallucinate.
 *
 * 2. NOTHING IS DESTROYED.
 *    Only the request payload is trimmed. The stored transcript is untouched,
 *    so "undo" is a toggle rather than a recovery path, and automatic
 *    compaction cannot lose a user's work even if the heuristic is wrong.
 *
 * What is retained, always -- the same spine the agent loop protects:
 *   - every `system` message, wherever it sits;
 *   - every `user` message, because in a multi-turn chat the CURRENT question
 *     is the last one, and dropping earlier questions while their answers
 *     survive inverts the conversation's meaning;
 *   - the last `keepRecent` messages, so immediate state is intact.
 *
 * Everything between is dropped from the payload. Deliberately, NO marker
 * message is injected in its place: a synthetic mid-conversation system turn
 * would change how the row's chat template renders, and exact template
 * fidelity is the one thing this project will not trade for convenience. The
 * elision is reported to the caller so the UI can show it to the human, which
 * is where that information belongs.
 */

import { appStorage } from './appStorage.js'

/* Matches KEEP_RECENT in src/chat/agent.rs. Six covers three exchanges, which
   is where follow-up pronouns ("that one", "the second option") still resolve. */
export const KEEP_RECENT_MESSAGES = 6

/* Compaction starts here rather than at the wall, because a prompt that only
   just fits still leaves no useful room for a reply. */
export const AUTO_COMPACT_THRESHOLD_PERCENT = 80

export const AUTO_COMPACT_STORAGE_KEY = 'camelid.autoCompact'

function roleOf(message) {
  return String(message?.role || '').toLowerCase()
}

/* A turn is protected when the spine covers it, independent of position. */
function isProtectedRole(message) {
  const role = roleOf(message)
  return role === 'system' || role === 'user'
}

/**
 * Trim a conversation for sending.
 *
 * @param {Array} messages Ordered chat messages destined for the request.
 * @param {object} [options]
 * @param {number} [options.keepRecent] Trailing messages retained regardless of role.
 * @returns {{messages: Array, elidedCount: number, elidedFrom: number}|null}
 *   `null` when there is nothing to elide, mirroring `agent.rs::compact`
 *   returning `None` so callers can leave the payload byte-identical.
 */
export function compactForSend(messages, { keepRecent = KEEP_RECENT_MESSAGES } = {}) {
  if (!Array.isArray(messages) || messages.length === 0) return null
  const recentFrom = Math.max(messages.length - Math.max(keepRecent, 0), 0)

  const kept = []
  let elidedCount = 0
  let elidedFrom = -1
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index]
    if (index >= recentFrom || isProtectedRole(message)) {
      kept.push(message)
      continue
    }
    if (elidedFrom < 0) elidedFrom = index
    elidedCount += 1
  }

  if (elidedCount === 0) return null
  return { messages: kept, elidedCount, elidedFrom }
}

/**
 * Whether automatic compaction should run for the current fill level.
 *
 * Kept separate from `compactForSend` so the decision is testable without a
 * conversation, and so a caller can apply the same threshold to a warning
 * badge without trimming anything.
 */
export function shouldAutoCompact({
  enabled,
  filledPercent,
  threshold = AUTO_COMPACT_THRESHOLD_PERCENT,
} = {}) {
  if (!enabled) return false
  const percent = Number(filledPercent)
  if (!Number.isFinite(percent)) return false
  return percent >= threshold
}

/**
 * Apply compaction only when it is both wanted and useful.
 *
 * Returns the original array when compaction is off, not yet triggered, or
 * would elide nothing, so the common case sends an unmodified payload.
 */
export function applySendCompaction(messages, {
  enabled = false,
  forced = false,
  filledPercent = 0,
  threshold = AUTO_COMPACT_THRESHOLD_PERCENT,
  keepRecent = KEEP_RECENT_MESSAGES,
} = {}) {
  const wanted = forced || shouldAutoCompact({ enabled, filledPercent, threshold })
  if (!wanted) return { messages, compacted: false, elidedCount: 0 }
  const result = compactForSend(messages, { keepRecent })
  if (!result) return { messages, compacted: false, elidedCount: 0 }
  return { messages: result.messages, compacted: true, elidedCount: result.elidedCount }
}

/* Preference storage.
 *
 * The meter and the send path each read these directly rather than passing
 * state between them (the same shape `responseLimits.js` uses for the response
 * limit). One source means the panel cannot advertise a trim the request does
 * not perform. */

const OVERRIDE_KEY_PREFIX = 'camelid.compactOverride'

export function getAutoCompactEnabled() {
  if (typeof window === 'undefined') return true
  /* Default ON: the failure it prevents is a hard `context_length_exceeded`
     with no reply, and because only the payload is trimmed, being wrong costs
     the user nothing they cannot undo with one click. */
  return appStorage.getItem(AUTO_COMPACT_STORAGE_KEY) !== '0'
}

export function setAutoCompactEnabled(value) {
  if (typeof window === 'undefined') return
  appStorage.setItem(AUTO_COMPACT_STORAGE_KEY, value ? '1' : '0')
}

/**
 * Per-conversation override: `'force'`, `'off'`, or `null` to follow the
 * automatic threshold. Scoped per conversation because "send everything" is a
 * decision about one chat, not a change of preference.
 */
export function getCompactionOverride(conversationId) {
  if (typeof window === 'undefined' || !conversationId) return null
  const value = appStorage.getItem(`${OVERRIDE_KEY_PREFIX}.${conversationId}`)
  return value === 'force' || value === 'off' ? value : null
}

export function setCompactionOverride(conversationId, value) {
  if (typeof window === 'undefined' || !conversationId) return
  const key = `${OVERRIDE_KEY_PREFIX}.${conversationId}`
  if (value === 'force' || value === 'off') appStorage.setItem(key, value)
  else appStorage.removeItem(key)
}

/** Resolve the stored preference and override into `applySendCompaction` inputs. */
export function resolveCompactionIntent(conversationId) {
  const override = getCompactionOverride(conversationId)
  if (override === 'off') return { enabled: false, forced: false, override }
  if (override === 'force') return { enabled: true, forced: true, override }
  return { enabled: getAutoCompactEnabled(), forced: false, override }
}
