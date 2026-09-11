/* TeX delimiter scanning for assistant prose.

   Splitting model output into "prose" and "math" is mostly an exercise in NOT
   matching. Currency is the reason: "the 8B costs $40 and the 27B costs $90"
   contains two dollar signs and is not math, and a renderer that treats it as
   math turns a sentence into a garbled formula. Models also emit at least four
   delimiter styles depending on family, so ignoring \( \) and \[ \] misses
   most of what Llama and Qwen actually produce.

   The rules for a bare $...$ span, which are the ones that matter:
     - the opening $ is not followed by whitespace
     - the closing $ is not preceded by whitespace
     - the closing $ is not followed by a digit
     - the span does not cross a blank line
   Under those rules "$40 and the 27B costs $90" fails at the second rule (the
   candidate closing $ has a space before it) and stays prose, while "$x^2$"
   passes all four. $$...$$ and \[...\] are unambiguous about intent and need
   only the blank-line guard below, which stops an unterminated delimiter from
   pairing with one several paragraphs away and swallowing the text between.

   Written as a scanner rather than one regex on purpose: the guards need
   lookbehind, and Safari did not ship lookbehind until 16.4. */

const BLOCK_DELIMITERS = [
  { open: '$$', close: '$$' },
  { open: '\\[', close: '\\]' },
]
const INLINE_ESCAPED = { open: '\\(', close: '\\)' }

function isWhitespace(char) {
  return char === undefined || /\s/.test(char)
}

/* A math span must not swallow a paragraph break; an unterminated delimiter is
   far more likely than a formula containing one. */
function crossesBlankLine(value) {
  return /\n[ \t]*\n/.test(value)
}

function pushText(segments, value) {
  if (!value) return
  const last = segments[segments.length - 1]
  if (last && last.type === 'text') last.value += value
  else segments.push({ type: 'text', value })
}

/* Returns [{ type: 'text', value } | { type: 'math', value, display }].
   `value` on a math segment is the TeX source with delimiters stripped. */
export function splitMathSegments(input) {
  const text = String(input || '')
  const segments = []
  let cursor = 0
  let plainStart = 0

  const commit = (end) => pushText(segments, text.slice(plainStart, end))

  while (cursor < text.length) {
    const char = text[cursor]

    // An escaped dollar is a literal dollar, never a delimiter.
    if (char === '\\' && text[cursor + 1] === '$') {
      cursor += 2
      continue
    }

    let matched = false
    for (const { open, close } of BLOCK_DELIMITERS) {
      if (!text.startsWith(open, cursor)) continue
      const bodyStart = cursor + open.length
      const closeAt = text.indexOf(close, bodyStart)
      if (closeAt === -1) continue
      const body = text.slice(bodyStart, closeAt)
      /* An unterminated delimiter that finds its partner paragraphs later
         would swallow everything between. Failing the match degrades to
         visible TeX; matching it destroys the rest of the message. */
      if (!body.trim() || crossesBlankLine(body)) continue
      commit(cursor)
      segments.push({ type: 'math', value: body.trim(), display: true })
      cursor = closeAt + close.length
      plainStart = cursor
      matched = true
      break
    }
    if (matched) continue

    if (text.startsWith(INLINE_ESCAPED.open, cursor)) {
      const bodyStart = cursor + INLINE_ESCAPED.open.length
      const closeAt = text.indexOf(INLINE_ESCAPED.close, bodyStart)
      const body = closeAt === -1 ? null : text.slice(bodyStart, closeAt)
      if (body !== null && body.trim() && !crossesBlankLine(body)) {
        commit(cursor)
        segments.push({ type: 'math', value: body.trim(), display: false })
        cursor = closeAt + INLINE_ESCAPED.close.length
        plainStart = cursor
        continue
      }
    }

    if (char === '$') {
      const bodyStart = cursor + 1
      if (!isWhitespace(text[bodyStart])) {
        let scan = bodyStart
        let closeAt = -1
        while (scan < text.length) {
          if (text[scan] === '\\') { scan += 2; continue }
          if (text[scan] === '$') {
            const precededBySpace = isWhitespace(text[scan - 1])
            const followedByDigit = /\d/.test(text[scan + 1] || '')
            if (!precededBySpace && !followedByDigit) { closeAt = scan; break }
            // A candidate that fails the guards is currency, not a delimiter:
            // stop rather than reaching further for a later $.
            break
          }
          scan += 1
        }
        if (closeAt > bodyStart) {
          const body = text.slice(bodyStart, closeAt)
          if (body.trim() && !crossesBlankLine(body)) {
            commit(cursor)
            segments.push({ type: 'math', value: body.trim(), display: false })
            cursor = closeAt + 1
            plainStart = cursor
            continue
          }
        }
      }
    }

    cursor += 1
  }

  commit(text.length)
  return segments
}

export function hasMath(input) {
  return splitMathSegments(input).some((segment) => segment.type === 'math')
}

/* A whole line that is nothing but one display formula renders as its own
   block instead of a paragraph containing a formula. */
export function displayMathOnlyLine(line) {
  const segments = splitMathSegments(String(line || '').trim())
  if (segments.length !== 1) return null
  const [only] = segments
  return only.type === 'math' && only.display ? only.value : null
}
