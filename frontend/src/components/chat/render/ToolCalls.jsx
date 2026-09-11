import { useState } from 'react'
import { formatArguments, normalizeToolCalls, replyCarriesRawEnvelope } from '../../../lib/toolCalling'

/* Tool calls — what the model ASKED to call.

   Self-gating like the parity receipt beside it.

   COPY RULE, and it is the whole point of this component: nothing here ran.
   The model produced a request, the turn ended with finish_reason "tool_calls",
   and that is the entire event. This card never says a tool executed, never
   implies a result, and offers no way to run one — a browser executing
   model-chosen calls against the user's machine is a different and much larger
   proposition than showing what was asked for.

   The arguments are a JSON string the MODEL wrote. They can be malformed and can
   disagree with the schema that was offered. A parse failure is shown, not
   swallowed: a model emitting broken arguments is a real thing to know, and
   presenting it as a clean call would hide it. */

export function ToolCallsCard({ toolCalls, repeated = null, replyContent = '' }) {
  const [expanded, setExpanded] = useState(true)
  const calls = normalizeToolCalls(toolCalls)
  if (!calls) return null

  return (
    <div className="toolcalls">
      <button
        type="button"
        className="toolcalls__trigger"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        <span className="toolcalls__label">Tool call requested</span>
        <span className="toolcalls__summary">
          {calls.length === 1 ? calls[0].name || 'unnamed' : `${calls.length} calls`}
        </span>
      </button>

      {expanded && (
        <div className="toolcalls__body">
          <p className="toolcalls__note">
            The model ended this turn by asking to call {calls.length === 1 ? 'this' : 'these'}. Nothing
            has run — Camelid does not execute tool calls from the browser. Supply a result below to
            continue the conversation.
          </p>

          {replyCarriesRawEnvelope(replyContent) && (
            <p className="toolcalls__note">
              This lane also left the model&rsquo;s raw call markup in the reply text above. The calls
              below were lifted from it — the two describe the same request, not two requests.
            </p>
          )}

          {repeated?.repeated && (
            <p className="toolcalls__repeat">
              This is the same call as before{repeated.name ? ` (${repeated.name})` : ''}, with the same
              arguments. The model is not using the result it was given, so continuing again will
              probably repeat it.
            </p>
          )}

          <ul className="toolcalls__list">
            {calls.map((call) => (
              <li key={call.id || call.index} className="toolcalls__call">
                <div className="toolcalls__call-head">
                  {/* Rendered as text, never as markup: a tool name is model output. */}
                  <code className="toolcalls__name">{call.name || '(no name)'}</code>
                  {call.id && <span className="toolcalls__id">{call.id}</span>}
                </div>
                {call.parseError ? (
                  <p className="toolcalls__parse-error">
                    The model&rsquo;s arguments are not valid JSON ({call.parseError}). Shown exactly as
                    they arrived:
                  </p>
                ) : null}
                <pre className="toolcalls__args">{formatArguments(call)}</pre>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default ToolCallsCard
