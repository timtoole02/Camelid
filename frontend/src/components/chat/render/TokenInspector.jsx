import { useMemo, useRef, useState } from 'react'
import {
  alternativesFor,
  formatLogprob,
  formatPerplexity,
  formatProbability,
  normalizeInspection,
} from '../../../lib/tokenInspection'

/* Token Inspector card — the per-token probability record for one reply.

   Self-gating like the parity receipt beside it: MessageTurn renders this
   unconditionally and the component decides whether it has anything to say.

   COPY RULE. Everything here describes ONE captured generation. The numbers are a
   log-softmax over the model's raw logits for this decode — not sampling
   probabilities, not a quality score, and not a claim about the lane. The words
   "confidence" and "certainty" are deliberately absent: a softmax over logits does
   not license a statement about what the model knew, only about what it computed.

   The salience budget is inverted on purpose. A greedy reply is overwhelmingly
   high-probability, so marking probability directly would ink the unremarkable
   majority. Only contested positions carry weight; settled ones render as plain
   text. */

const RENDER_CAP = 512

const BAND_LABEL = {
  settled: 'ranked far above the alternatives',
  leading: 'ranked ahead of the alternatives',
  contested: 'closely contested',
  unknown: 'not reported',
}

const downloadJson = (filename, value) => {
  try {
    const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)
  } catch {
    // Download is best-effort outside full browser contexts.
  }
}

function AlternativeRow({ alternative, isChosen }) {
  const width = Math.max(1, Math.round((alternative.probability ?? 0) * 100))
  return (
    <li className={`tokinsp__alt ${isChosen ? 'is-chosen' : ''}`}>
      <span className="tokinsp__alt-rank" aria-hidden="true">{alternative.rank}</span>
      <span className="tokinsp__alt-token" title={alternative.substituted ? `bytes: [${alternative.bytes.join(', ')}]` : undefined}>
        {alternative.display}
      </span>
      {/* Bar width is absolute against 1.0, never normalized to the leader — a
          position whose top candidate sits at 31% must not look like one at 99%. */}
      <span className="tokinsp__alt-bar" aria-hidden="true">
        <span className="tokinsp__alt-bar-fill" style={{ width: `${width}%` }} />
      </span>
      <span className="tokinsp__alt-value">{formatProbability(alternative.probability)}</span>
      {alternative.tiedWithPrevious && (
        <span className="tokinsp__alt-tie" title="Within measurement noise of the entry above — the ordering between them is not meaningful.">tie</span>
      )}
      {isChosen && <span className="sr-only">this is the token that was emitted</span>}
    </li>
  )
}

function AbsenceNotice({ reason }) {
  return (
    <div className="tokinsp tokinsp--absent">
      <p className="tokinsp__absent-title">{reason.title}</p>
      <p className="tokinsp__absent-detail">{reason.detail}</p>
    </div>
  )
}

/* `defaultOpen` exists so the expanded body can be rendered without a browser.
   Without it the panel's entire content sits behind internal state, and a static
   render would assert only against the collapsed trigger — passing while the body
   was broken. */
export function TokenInspectorCard({ inspection, absence, candidatesContract = null, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const [selected, setSelected] = useState(null)
  const stripRef = useRef(null)

  const model = useMemo(() => (inspection ? normalizeInspection(inspection) : null), [inspection])

  if (absence) return <AbsenceNotice reason={absence} />
  if (!model) return null

  const { tokens, stats } = model
  const shown = tokens.slice(0, RENDER_CAP)
  const truncated = tokens.length - shown.length
  /* Opening on token 0 would make the panel look like it has nothing to say. The
     lowest-probability position is the one a reader came here for. */
  const activeIndex = selected ?? stats.lowestProbabilityIndex ?? 0
  const active = tokens[activeIndex] || tokens[0]

  const summary = stats.contestedCount > 0
    ? `${stats.tokenCount} tokens · ${stats.contestedCount} closely contested`
    : `${stats.tokenCount} tokens · none closely contested`

  const onStripKeyDown = (event) => {
    const last = shown.length - 1
    let next = null
    if (event.key === 'ArrowRight') next = Math.min(last, activeIndex + 1)
    else if (event.key === 'ArrowLeft') next = Math.max(0, activeIndex - 1)
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = last
    if (next === null) return
    event.preventDefault()
    setSelected(next)
    stripRef.current?.querySelector(`[data-token-index="${next}"]`)?.focus()
  }

  return (
    <div className="tokinsp">
      <button
        type="button"
        className="tokinsp__trigger"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="tokinsp__trigger-label">Token probabilities</span>
        <span className="tokinsp__trigger-summary">{summary}</span>
      </button>

      {open && (
        <div className="tokinsp__body">
          <p className="tokinsp__provenance">
            Recorded during the decode that produced this reply. These are the model’s raw
            per-token scores, before any sampling settings were applied.
          </p>

          <div
            className="tokinsp__strip"
            role="list"
            ref={stripRef}
            onKeyDown={onStripKeyDown}
            aria-label="Generated tokens in order"
          >
            {shown.map((token) => (
              <button
                key={token.index}
                type="button"
                role="listitem"
                data-token-index={token.index}
                tabIndex={token.index === activeIndex ? 0 : -1}
                className={`tokinsp__tok tokinsp__tok--${token.band} ${token.index === activeIndex ? 'is-active' : ''}`}
                onClick={() => setSelected(token.index)}
                onFocus={() => setSelected(token.index)}
                title={token.substituted ? `bytes: [${token.bytes.join(', ')}]` : undefined}
                aria-label={`Token ${token.index + 1}, ${token.raw || 'empty'}, ${formatProbability(token.probability)}, ${BAND_LABEL[token.band]}`}
              >
                {token.display}
                {/* A chosen token that was not the top-ranked one is real signal
                    about the decode, so it is marked in a non-color channel. */}
                {!token.chosenIsTop && token.logprob !== null && (
                  <span className="tokinsp__tok-rank" aria-hidden="true">▾</span>
                )}
              </button>
            ))}
          </div>
          {truncated > 0 && (
            <p className="tokinsp__truncation">
              Showing the first {RENDER_CAP} of {stats.tokenCount} tokens. The full record is in the downloaded JSON.
            </p>
          )}

          {active && (
            <div className="tokinsp__detail">
              <p className="tokinsp__detail-head">
                Position {active.index + 1} — emitted{' '}
                <code className="tokinsp__detail-token">{active.display}</code>{' '}
                at {formatProbability(active.probability)} ({formatLogprob(active.logprob)} log)
              </p>
              {!active.chosenInAlternatives && active.logprob !== null && (
                <p className="tokinsp__detail-note">
                  The emitted token is not among the {active.alternatives.length} alternatives returned for this
                  position, so it ranked below them.
                </p>
              )}
              <ul className="tokinsp__alts">
                {active.alternatives.map((alternative) => (
                  <AlternativeRow
                    key={`${alternative.rank}-${alternative.raw}`}
                    alternative={alternative}
                    isChosen={alternative.raw === active.raw && alternative.rank === 1 && active.chosenIsTop}
                  />
                ))}
                {/* The shown alternatives are a truncation of the whole vocabulary.
                    Without this row, k entries read as a closed set. */}
                <li className="tokinsp__alt tokinsp__alt--residual">
                  <span className="tokinsp__alt-rank" aria-hidden="true">·</span>
                  <span className="tokinsp__alt-token">rest of the vocabulary</span>
                  <span className="tokinsp__alt-bar" aria-hidden="true">
                    <span className="tokinsp__alt-bar-fill" style={{ width: `${Math.max(1, Math.round((active.residualMass ?? 0) * 100))}%` }} />
                  </span>
                  <span className="tokinsp__alt-value">{formatProbability(active.residualMass)}</span>
                </li>
              </ul>
            </div>
          )}

          <dl className="tokinsp__stats">
            <div><dt>Tokens</dt><dd>{stats.tokenCount}</dd></div>
            <div><dt>Alternatives per position</dt><dd>{stats.depth}</dd></div>
            <div><dt>Mean log-probability</dt><dd>{formatLogprob(stats.meanLogprob)}</dd></div>
            <div>
              <dt>Perplexity</dt>
              <dd title="Derived from this reply's own tokens. Not comparable across different prompts.">
                {formatPerplexity(stats.perplexity)}
              </dd>
            </div>
          </dl>

          {stats.offTopCount > 0 && (
            <p className="tokinsp__finding">
              At {stats.offTopCount} position{stats.offTopCount === 1 ? '' : 's'} the emitted token was not the
              highest-scoring one.
            </p>
          )}

          {/* n>1 is guarded, not offered. The engine refuses logprobs together with
              multiple choices, and on a parity-locked row every choice would decode
              identically anyway. Stating both keeps the reason honest. */}
          <div className="tokinsp__guarded">
            <p className="tokinsp__guarded-title">Side-by-side candidates — not available</p>
            <p className="tokinsp__guarded-detail">
              The engine reports probabilities for a single reply only, and refuses them alongside
              multiple candidates. On a parity-locked model row the reply is decoded greedily, so
              repeated candidates would be identical.
              {candidatesContract?.rowId ? ` Contract row: ${candidatesContract.rowId}.` : ''}
            </p>
          </div>

          <div className="tokinsp__actions">
            <button
              type="button"
              className="tokinsp__action"
              onClick={() => downloadJson(`camelid-token-probabilities-${stats.tokenCount}.json`, inspection)}
            >
              Download JSON
            </button>
            <span className="tokinsp__retention">
              Held for this session only — not saved with the conversation.
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default TokenInspectorCard

/* Re-exported so a caller can build a preview without normalizing a whole reply. */
export { alternativesFor }
