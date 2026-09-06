import { useEffect, useRef, useState } from 'react'
import { IconMemory, IconCheckCircle, IconRefresh } from '../ui/icons.jsx'
import {
  composeContextBudget,
  formatTokenCount,
  formatPercent,
} from '../../lib/contextBudget.js'
import { AUTO_COMPACT_THRESHOLD_PERCENT } from '../../lib/conversationCompaction.js'

/* How full the model's context window is, and what is filling it.
 *
 * Collapsed it is a chip in the composer status line; expanded it breaks the
 * window into the segments the prompt actually occupies. Two things here are
 * deliberate and should survive edits:
 *
 *   - The reservation is drawn as its own hatched segment rather than folded
 *     into "used", because it is room set aside for the reply, not spent yet.
 *   - The verified bound is drawn as a marker ON the bar with context beyond it
 *     still rendered as usable. The tested envelope is evidence, not a ceiling;
 *     showing it as a wall is what makes a 40k-token model look like an 8k one.
 *
 * Prompt size is a client estimate and is labelled as one everywhere it shows.
 */
export function ContextMeter({
  contextLength,
  promptTokens,
  systemTokens = 0,
  imageTokens = 0,
  reservedTokens,
  verifiedBound = null,
  executionLane = '',
  autoCompact = false,
  onToggleAutoCompact = null,
  onCompactNow = null,
  canCompact = false,
  compaction = null,
  onSendEverything = null,
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    function onPointerDown(event) {
      if (!rootRef.current?.contains(event.target)) setOpen(false)
    }
    function onKeyDown(event) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  const budget = composeContextBudget({
    contextLength,
    promptTokens,
    systemTokens,
    imageTokens,
    reservedTokens,
    verifiedBound,
    warnAtPercent: AUTO_COMPACT_THRESHOLD_PERCENT,
  })

  /* A model whose window we cannot read gets no meter at all rather than a
     guessed one — an invented denominator is worse than no denominator. */
  if (!budget) return null

  /* Same total the percentage is computed from, so the chip cannot read "0%"
     next to a token count that has already reached the window size. */
  const summary = `${formatTokenCount(budget.committedTokens)} / ${formatTokenCount(budget.contextLength)} tokens`
  const tone = budget.level === 'ok' && budget.nearLimit ? 'near' : budget.level

  return (
    <div className="ctxmeter" ref={rootRef}>
      <button
        type="button"
        className={`ctxmeter__chip is-${tone}`}
        aria-expanded={open}
        aria-label={`Context window ${formatPercent(budget.filledPercent)} used. ${summary}. Show breakdown.`}
        onClick={() => setOpen((value) => !value)}
      >
        <IconMemory size={13} />
        <span className="ctxmeter__track" aria-hidden="true">
          <span className="ctxmeter__fill" style={{ width: `${Math.min(budget.usedPercent, 100)}%` }} />
          <span
            className="ctxmeter__fill ctxmeter__fill--reserved"
            style={{ left: `${Math.min(budget.usedPercent, 100)}%`, width: `${budget.reservedPercent}%` }}
          />
        </span>
        <span className="ctxmeter__chip-value">{formatPercent(budget.filledPercent)}</span>
      </button>

      {open && (
        <div className="ctxmeter__panel" role="group" aria-label="Context window breakdown">
          <p className="ctxmeter__headline">
            <strong>{formatPercent(budget.filledPercent)}</strong> of this model&rsquo;s context is spoken for
          </p>

          <div className="ctxmeter__bar">
            <span className="ctxmeter__bar-fill" style={{ width: `${Math.min(budget.usedPercent, 100)}%` }} />
            <span
              className="ctxmeter__bar-fill ctxmeter__bar-fill--reserved"
              style={{ left: `${Math.min(budget.usedPercent, 100)}%`, width: `${budget.reservedPercent}%` }}
            />
            {budget.showVerifiedMarker && (
              <span
                className="ctxmeter__marker"
                style={{ left: `${budget.verifiedPercent}%` }}
                aria-hidden="true"
              />
            )}
          </div>

          {/* Reads as a receipt rather than a formula: the numbers stack and
              sum, which is the shape this project already asks people to trust. */}
          <ul className="ctxmeter__receipt">
            {budget.segments.map((segment) => (
              <li key={segment.key} className={`ctxmeter__receipt-row is-${segment.key}`}>
                <span className="ctxmeter__swatch" aria-hidden="true" />
                <span className="ctxmeter__receipt-num">{segment.tokens.toLocaleString()}</span>
                <span className="ctxmeter__receipt-label">{segment.label}</span>
              </li>
            ))}
            <li className="ctxmeter__receipt-row ctxmeter__receipt-row--total">
              <span className="ctxmeter__swatch ctxmeter__swatch--blank" aria-hidden="true" />
              <span className="ctxmeter__receipt-num">{budget.contextLength.toLocaleString()}</span>
              <span className="ctxmeter__receipt-label">tokens in total</span>
            </li>
          </ul>

          {budget.showVerifiedMarker && (
            <p className="ctxmeter__verified">
              <IconCheckCircle size={12} />
              <span>
                {budget.beyondVerified
                  ? <>Past the {budget.verifiedBound.toLocaleString()}-token tested mark — still fine, just no receipt.</>
                  : <>Tested to {budget.verifiedBound.toLocaleString()} tokens — it works beyond that, just without a receipt.</>}
              </span>
            </p>
          )}

          {(onCompactNow || onToggleAutoCompact) && (
            <div className="ctxmeter__compact">
              {onCompactNow && (
                <button
                  type="button"
                  className="ctxmeter__compact-action"
                  onClick={onCompactNow}
                  disabled={!canCompact}
                  /* A button that silently does nothing is worse than one that
                     refuses and says why. */
                  title={canCompact ? undefined : 'Everything here is either yours or recent, so there is nothing to leave out.'}
                >
                  <IconRefresh size={12} /> Trim what gets sent
                </button>
              )}
              {onToggleAutoCompact && (
                <label className="ctxmeter__compact-auto">
                  <input
                    type="checkbox"
                    checked={autoCompact}
                    onChange={(event) => onToggleAutoCompact(event.target.checked)}
                  />
                  <span>Trim automatically at {AUTO_COMPACT_THRESHOLD_PERCENT}%</span>
                </label>
              )}
            </div>
          )}

          {onCompactNow && !canCompact && (
            <p className="ctxmeter__compact-hint">
              Nothing to trim yet — everything here is recent or yours.
            </p>
          )}

          {compaction?.active && (
            <p className="ctxmeter__compacted">
              <span>
                {compaction.elidedCount.toLocaleString()} older{' '}
                {compaction.elidedCount === 1 ? 'reply is' : 'replies are'} being left out of what gets sent
                {compaction.freedTokens > 0 && <>, freeing about {compaction.freedTokens.toLocaleString()} tokens</>}.
                {' '}Nothing was deleted — your transcript is untouched.
              </span>
              {onSendEverything && (
                <button type="button" className="ctxmeter__compact-undo" onClick={onSendEverything}>
                  Send it all
                </button>
              )}
            </p>
          )}

          <p className="ctxmeter__foot">
            Your hardware
            {executionLane && <> &middot; <code>{executionLane}</code></>}
            {' '}&middot; sizes estimated until sent
          </p>
        </div>
      )}
    </div>
  )
}

export default ContextMeter
