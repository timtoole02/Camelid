import { useMemo, useState } from 'react'
import { EVIDENCE_COPY, assessStructuredReply } from '../../../lib/structuredOutput'

/* Structured-output card — what a constrained reply actually demonstrates.

   Self-gating like the parity receipt and the token inspector beside it.

   COPY RULE. The engine returns no field saying a constraint was applied, so this
   card must never render "constrained" on the strength of a 200. It reports the
   strongest thing it can actually see, in this order:

     Mask observed   an emitted token was not the highest-scoring one on a greedy
                     turn — the returned scores are unmasked, so the constraint
                     demonstrably moved the decode
     Matches schema  parsed and satisfied every keyword this page checks
     Parsed          valid JSON, nothing shown about the constraint
     Accepted        the request was accepted; that is all

   The first is evidence. The rest are consistent-with, and say so. */

function Row({ label, children }) {
  return (
    <div className="structout__row">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  )
}

export function StructuredOutputCard({ record }) {
  const [showRaw, setShowRaw] = useState(false)
  const assessment = useMemo(() => (record ? assessStructuredReply(record) : null), [record])
  if (!assessment) return null

  const copy = EVIDENCE_COPY[assessment.evidence] || EVIDENCE_COPY.accepted
  const isJsonMode = assessment.mode === 'json_object' || assessment.mode === 'json_schema'

  return (
    <div className="structout">
      <div className="structout__head">
        <span className="structout__label">Structured output</span>
        <span className={`structout__verdict structout__verdict--${assessment.evidence}`}>{copy.label}</span>
      </div>
      <p className="structout__detail">{copy.detail}</p>

      <dl className="structout__rows">
        <Row label="Constraint">
          {assessment.mode === 'grammar' ? 'Lark grammar' : assessment.mode === 'json_schema' ? 'JSON schema' : 'JSON object'}
        </Row>
        {isJsonMode && (
          <Row label="Parses">
            {assessment.parses ? 'yes' : `no — ${assessment.parseError}`}
          </Row>
        )}
        {assessment.schemaChecked && (
          <Row label="Schema">
            {assessment.problems.length === 0
              ? 'every checked keyword satisfied'
              : `${assessment.problems.length} problem${assessment.problems.length === 1 ? '' : 's'}`}
          </Row>
        )}
        {assessment.diverted !== null && (
          <Row label="Diverted positions">{assessment.diverted}</Row>
        )}
      </dl>

      {assessment.problems.length > 0 && (
        <ul className="structout__problems">
          {assessment.problems.slice(0, 8).map((problem) => (
            <li key={problem}>{problem}</li>
          ))}
        </ul>
      )}

      {/* Naming what was NOT examined is the point: a validator that quietly
          skipped the keywords it does not implement would report "valid" for a
          document it never checked. */}
      {assessment.unchecked.length > 0 && (
        <p className="structout__unchecked">
          Not checked here: {assessment.unchecked.slice(0, 6).join(', ')}
          {assessment.unchecked.length > 6 ? ` and ${assessment.unchecked.length - 6} more` : ''} — this page checks
          types, required keys and enums only, not the full schema dialect.
        </p>
      )}

      {assessment.parses && (
        <>
          <button type="button" className="structout__toggle" onClick={() => setShowRaw((v) => !v)} aria-expanded={showRaw}>
            {showRaw ? 'Hide parsed value' : 'Show parsed value'}
          </button>
          {showRaw && (
            <pre className="structout__json">{JSON.stringify(assessment.parsed, null, 2)}</pre>
          )}
        </>
      )}
    </div>
  )
}

export default StructuredOutputCard
