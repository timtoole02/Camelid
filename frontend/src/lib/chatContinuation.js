/* Continue a length-truncated reply.

   A reply that ends with finish_reason="length" ran out of response budget, it
   did not finish. Continuing it is a different operation from regenerating:
   the existing text is kept and the model is asked to resume from it, and the
   new text is APPENDED to the same assistant message so the transcript reads
   as one reply rather than two.

   The continuation instruction is REQUEST-ONLY. It is never written into the
   stored transcript, so the next ordinary turn sends one clean assistant
   message instead of a conversation littered with "continue" turns. What was
   actually sent stays disclosed on the message itself (continuation_count) and
   in the developer diagnostics block — the instruction is hidden from the
   transcript, never from the reader. */

export const CONTINUATION_INSTRUCTION = [
  'Continue your previous reply from exactly where it stopped.',
  'Do not repeat any text you already wrote, do not restate the question, and do not add a preamble.',
  'Resume mid-sentence if that is where it ended.',
].join(' ')

/* Only the tail of the prefix is compared, and only a substantial overlap
   counts. A model that resumes by restating its last sentence is common; a
   model that legitimately opens with the same three words is also common, so
   a short match must not be treated as a repeat. */
const OVERLAP_SCAN_CHARS = 400
const MIN_OVERLAP_CHARS = 24

export function canContinueMessage(message) {
  if (!message || message.role !== 'assistant' || message.streaming) return false
  if (message.finish_reason !== 'length') return false
  return Boolean(String(message.content || '').trim())
}

export function continuationPrefixOf(message) {
  return canContinueMessage(message) ? String(message.content || '') : ''
}

/* Drop the longest prefix of `addition` that repeats the tail of `prefix`.
   Longest-first so a model that repeats two sentences is trimmed once, not
   partially. */
function stripRepeatedOverlap(prefix, addition) {
  if (!prefix || !addition) return addition
  const tail = prefix.slice(-OVERLAP_SCAN_CHARS)
  const limit = Math.min(tail.length, addition.length)
  for (let length = limit; length >= MIN_OVERLAP_CHARS; length -= 1) {
    if (addition.startsWith(tail.slice(tail.length - length))) return addition.slice(length)
  }
  return addition
}

/* Join a continuation onto the text it resumes.

   Whitespace is the whole problem here: the truncated text can stop
   mid-word ("the imple"), at a word boundary, or after a newline, and the
   model's continuation may or may not lead with a space. Inserting a space
   would corrupt a mid-word resume, so the prefix decides — when it already
   ends in whitespace the continuation's own leading whitespace is dropped,
   and otherwise both sides are left exactly as they are. */
export function joinContinuation(prefix, addition) {
  const base = String(prefix || '')
  const next = stripRepeatedOverlap(base, String(addition || ''))
  if (!base) return next
  if (!next) return base
  if (/\s$/.test(base)) return base + next.replace(/^[ \t]+/, '')
  return base + next
}

/* Cumulative usage for a continued reply.

   completion_tokens is the whole reply's output, so it accumulates across
   continuations — the footer would otherwise report only the last segment and
   understate a reply the reader can see is longer. prompt_tokens is the LAST
   request's prompt (each continuation re-sends a longer prompt); carrying a
   sum there would describe no request that was ever made. */
export function mergeContinuedUsage(previousUsage, nextUsage) {
  const previousOut = Math.max(0, Number(previousUsage?.completion_tokens) || 0)
  const nextOut = Math.max(0, Number(nextUsage?.completion_tokens) || 0)
  const promptTokens = Number(nextUsage?.prompt_tokens)
  const completionTokens = previousOut + nextOut
  const resolvedPrompt = Number.isFinite(promptTokens) ? promptTokens : 0
  return {
    prompt_tokens: resolvedPrompt,
    completion_tokens: completionTokens,
    total_tokens: resolvedPrompt + completionTokens,
  }
}

export function continuationCountOf(message) {
  return Math.max(0, Number(message?.continuation_count) || 0)
}
