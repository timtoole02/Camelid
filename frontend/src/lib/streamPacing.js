/* Display pacing (Phase 8B): smooths bursty token arrival into a steady visual
   cadence under hard honesty bounds — the displayed text may never lag the
   truly received stream by more than MAX_LAG_MS, drains instantly on stream
   end/abort, and the final rendered text is byte-identical to the received
   text. Metrics (TTFT/tok-s tiles, telemetry, Flow Bench) always use real
   arrival data, never the paced view (I4). Pure functions; smoke-tested. */

export const MAX_LAG_MS = 150
export const FIRST_VISIBLE_PREFIX_CHARS = 32

/* Exponential catch-up time constant. The advance is computed from ELAPSED
   TIME, not from how often paceStep happens to be called, so the text flows at
   the same speed whether the display refreshes at 60Hz or 120Hz — a 120Hz
   panel simply gets twice as many, half-as-large steps, which is what makes
   the motion read as smooth instead of stepped. */
const CATCH_UP_TAU_MS = 25
/* Floor so a short tail still closes visibly rather than crawling. */
const MIN_CHARS_PER_MS = 0.05
/* A frame longer than this (tab was backgrounded, GC pause) must not teleport
   the text; the lag bound below still guarantees honesty. */
const MAX_FRAME_MS = 100

export function createPacerState() {
  return { shownChars: 0, arrivals: [], lastStepAt: null }
}

/* Record what has truly arrived; returns the text that may be shown at `now`:
   at least everything that arrived MAX_LAG_MS ago (the lag bound), advanced
   smoothly toward the freshest text. */
export function paceStep(state, receivedText, nowMs) {
  const received = receivedText.length
  const last = state.arrivals[state.arrivals.length - 1]
  if (!last || last.chars < received) state.arrivals.push({ chars: received, at: nowMs })
  while (state.arrivals.length > 2 && state.arrivals[1].at <= nowMs - MAX_LAG_MS) state.arrivals.shift()
  // lag bound: everything that arrived at or before now - MAX_LAG_MS must show
  let mustShow = 0
  for (const arrival of state.arrivals) {
    if (arrival.at <= nowMs - MAX_LAG_MS) mustShow = arrival.chars
  }
  // Smooth, time-based advance: close the remaining gap on an exponential
  // curve so bursty arrivals become steady motion. Frame-rate independent, and
  // fast enough that the tail converges well inside the lag bound on its own.
  const elapsed = state.lastStepAt == null ? 0 : Math.min(Math.max(nowMs - state.lastStepAt, 0), MAX_FRAME_MS)
  state.lastStepAt = nowMs
  const gap = received - state.shownChars
  const eased = gap * (1 - Math.exp(-elapsed / CATCH_UP_TAU_MS))
  const advance = Math.min(gap, Math.max(eased, elapsed * MIN_CHARS_PER_MS))
  let shown = state.shownChars + advance
  // Snap the last fraction so a quiet stream lands exactly on the received text.
  if (received - shown < 1) shown = received
  state.shownChars = Math.min(received, Math.max(mustShow, shown, state.shownChars))
  return receivedText.slice(0, Math.floor(state.shownChars))
}

/* Reveal a small first prefix immediately so a backgrounded tab never looks
   stuck at zero output. Record the full arrival with the ordinary pacer, but
   keep the remainder queued for paced display instead of synchronously
   rendering an arbitrarily large first network chunk. */
export function paceFirstVisiblePrefix(state, receivedText, nowMs, maxChars = FIRST_VISIBLE_PREFIX_CHARS) {
  const text = String(receivedText || '')
  if (!text) return ''
  paceStep(state, text, nowMs)
  const parsedLimit = Math.floor(Number(maxChars))
  const limit = Number.isFinite(parsedLimit) && parsedLimit > 0
    ? parsedLimit
    : FIRST_VISIBLE_PREFIX_CHARS
  const prefix = [...text].slice(0, limit).join('')
  state.shownChars = Math.max(state.shownChars, prefix.length)
  return text.slice(0, Math.floor(state.shownChars))
}

export function paceHasPendingText(displayedText, receivedText) {
  return String(displayedText || '').length < String(receivedText || '').length
}

/* Stream ended or aborted: drain instantly, byte-identical. */
export function paceDrain(state, receivedText) {
  state.shownChars = receivedText.length
  state.arrivals = []
  state.lastStepAt = null
  return receivedText
}
