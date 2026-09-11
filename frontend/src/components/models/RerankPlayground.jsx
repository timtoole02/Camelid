import { useEffect, useMemo, useState } from 'react'
import { EvidenceChip } from '../ui/EvidenceChip'
import {
  MAX_RERANK_DOCUMENTS,
  formatMovement,
  formatSimilarity,
  normalizeRerank,
  readRerankContract,
  rerankReadiness,
  rerankRequestBody,
  resolveEncoder,
  splitDocuments,
} from '../../lib/embeddingRerank'

/* Retrieval playground — rank documents against a query by meaning.

   Drives POST /v1/rerank (capability row `embedding_similarity_reranking`) on a
   resident encoder. This proves nothing about generation support, and the encoder
   is a different model from the chat one — the chip and the copy both say so.

   COPY RULE. `relevance_score` is a cosine similarity, not a probability. An
   unrelated document still scores well above zero, so a percentage rendering
   would badly mislead. Only the ORDER, the GAPS, and how far each document MOVED
   from its submitted position carry meaning here, and those are what the table
   shows. */

const SAMPLE_QUERY = 'How do I keep my laptop from overheating?'
const SAMPLE_DOCUMENTS = [
  'Bake the sourdough at 230C for 35 minutes on a stone.',
  'Thermal throttling happens when the CPU exceeds its junction temperature; clean the fans and repaste.',
  'The Treaty of Westphalia was signed in 1648.',
  'Elevate the chassis for airflow and avoid using it on a duvet, which blocks the intake vents.',
].join('\n')

/* `initialResult` exists so the results state can be rendered without a browser.
   Without it the ranked table and the score-interpretation note sit behind a live
   fetch, and a static render would assert only against the empty form — passing
   while the part that carries the claims was never exercised. */
export function RerankPlayground({ apiBase, capabilities, initialResult = null }) {
  const [query, setQuery] = useState(SAMPLE_QUERY)
  /* Self-contained like the tokenizer playground beside it: it resolves its own
     encoder from /v1/models rather than taking one through props, so mounting it
     costs one line at the call site. */
  const [served, setServed] = useState([])
  const [documentsText, setDocumentsText] = useState(SAMPLE_DOCUMENTS)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(initialResult)
  const [error, setError] = useState(null)

  const base = (apiBase || '').replace(/\/$/, '')
  const contract = useMemo(() => readRerankContract(capabilities), [capabilities])
  const encoder = useMemo(() => resolveEncoder(served, []), [served])
  const documents = useMemo(() => splitDocuments(documentsText), [documentsText])
  const readiness = useMemo(
    () => rerankReadiness({ contract, encoder, query, documents }),
    [contract, encoder, query, documents],
  )

  useEffect(() => {
    let cancelled = false
    fetch(`${base}/v1/models`)
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => { if (!cancelled && payload) setServed(payload.data || []) })
      .catch(() => { /* an unreachable backend is reported by the readiness line */ })
    return () => { cancelled = true }
  }, [base])

  const run = async () => {
    if (!readiness.ready || busy) return
    setBusy(true)
    setError(null)
    try {
      const response = await fetch(`${base}/v1/rerank`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rerankRequestBody({ encoderId: encoder.encoderId, query, documents })),
      })
      const payload = await response.json()
      if (!response.ok) {
        setResult(null)
        setError(payload?.error?.message || `rerank failed (${response.status})`)
        return
      }
      const normalized = normalizeRerank(payload, documents)
      if (!normalized) {
        setResult(null)
        setError('The engine returned no ranked results for these documents.')
        return
      }
      setResult(normalized)
    } catch (requestError) {
      setResult(null)
      setError(requestError?.message || 'The rerank request could not be sent.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rerank">
      <div className="rerank__head">
        <p className="rerank__intro">
          Rank documents against a query by meaning rather than by shared words. This runs on a
          loaded encoder, which is a different model from the chat one and loads alongside it —
          using this proves nothing about generation support.
        </p>
        {contract.present && (
          <EvidenceChip status={contract.status} source={{ rowId: contract.rowId }} />
        )}
      </div>

      {!readiness.ready && readiness.reason && (
        <p className="rerank__guard">{readiness.reason}</p>
      )}

      <label className="rerank__field">
        <span className="rerank__label">Query</span>
        <input
          type="text"
          className="rerank__input"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>

      <label className="rerank__field">
        <span className="rerank__label">Documents — one per line (max {MAX_RERANK_DOCUMENTS})</span>
        <textarea
          className="rerank__textarea"
          rows={6}
          spellCheck={false}
          value={documentsText}
          onChange={(event) => setDocumentsText(event.target.value)}
        />
      </label>

      <div className="rerank__actions">
        <button type="button" className="rerank__run" onClick={run} disabled={!readiness.ready || busy}>
          {busy ? 'Ranking…' : 'Rank by meaning'}
        </button>
        <span className="rerank__count">{documents.length} document{documents.length === 1 ? '' : 's'}</span>
      </div>

      {error && <p className="rerank__error">{error}</p>}

      {result && (
        <div className="rerank__results">
          <table className="cxv-table rerank__table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Similarity</th>
                <th scope="col">Moved</th>
                <th scope="col">Document</th>
              </tr>
            </thead>
            <tbody>
              {result.ranked.map((entry) => (
                <tr key={entry.index}>
                  <td>{entry.rank}</td>
                  <td className="rerank__score">{formatSimilarity(entry.score)}</td>
                  <td className={`rerank__moved ${entry.movedBy !== 0 ? 'is-moved' : ''}`}>
                    {formatMovement(entry.movedBy)}
                  </td>
                  <td className="rerank__doc">{entry.text}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="rerank__note">
            Scores are cosine similarities between the query and each document, not probabilities
            and not percentages — an unrelated document still scores well above zero. What carries
            meaning is the order, the gap between neighbours (spread here is{' '}
            {formatSimilarity(result.stats.spread)}), and how far each document moved from the order
            you entered it in.
            {result.stats.moved === 0
              ? ' Nothing moved: the encoder agrees with your input order.'
              : ` ${result.stats.moved} of ${result.stats.count} moved.`}
          </p>
        </div>
      )}
    </div>
  )
}

export default RerankPlayground
