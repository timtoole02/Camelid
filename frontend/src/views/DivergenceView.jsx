import { useState } from 'react'
import { CorsHint } from '../components/fabric/CorsHint.jsx'
import { Unknown } from '../components/fabric/Unknown.jsx'
import { useDivergence } from '../hooks/useDivergence.js'
import { useFabric } from '../hooks/useFabric.js'
import {
  COMPARE_MAX_TOKENS,
  advertisedTemplateNote,
  buildCompareRequest,
  comparisonCaveats,
  identityStatement,
  isAttributable,
  modelChoices,
  modelLooksAbsent,
  needsIdentityAssertion,
  renderedPromptComparison,
  reportedModelMismatch,
  templateDivergence,
  tokenCapStatement,
  uncontrolledStatements,
  verdictHeadline,
} from '../lib/divergenceModel.js'
import { DETAIL_WITHHELD_REASON, crossOriginDiagnosis, fabricProblemMessage } from '../lib/fabricModel.js'
import { endpointLabel, normalizeEndpoint } from '../lib/fabricClient.js'
import '../styles/divergence.css'

const VERDICT_TONE = {
  identical: 'same',
  divergent: 'differ',
  different_models: 'refused',
  not_attributable: 'refused',
}

/* The text of a line never carries its terminator, so a difference that is
   only CRLF against LF, or a missing final newline, needs a visible mark. */
const EOL_MARKER = {
  crlf: { text: '␍␊ CRLF', why: 'This line ends in a carriage return and a line feed.' },
  none: { text: 'no newline at end', why: 'The answer ends here, with no line ending.' },
}

function currentPageOrigin() {
  return typeof window === 'undefined' ? null : window.location.origin
}

function waitLabel(ms) {
  return ms >= 60000 ? `${Math.round(ms / 60000)} minutes` : `${Math.round(ms / 1000)} seconds`
}

function problemText(problem) {
  switch (problem.code) {
    case 'refused':
    case 'malformed':
      return problem.detail
    case 'unreachable':
      return `No fabric proxy answered the comparison request. ${problem.detail || ''}`.trim()
    case 'bad_endpoint':
      return 'That proxy address cannot be used.'
    case 'key_required':
      return 'This proxy requires a client key for comparisons. Enter one of its client keys above and compare again.'
    case 'key_refused':
      return 'The proxy did not accept that client key.'
    case 'timeout':
      return `No answer within ${waitLabel(problem.timeoutMs)}, so this page stopped waiting. The proxy may still `
        + 'be generating; its own forward timeout decides when a node has failed. Fewer runs or a lower token cap finish sooner.'
    default:
      return null
  }
}

/* Why there is no node list to pick from, when there is not. Each of these is
   a different fact, and "no nodes" is only one of them. */
function FabricState({ fabric, endpoint, pageOrigin }) {
  const origin = normalizeEndpoint(endpoint)
  const shown = endpointLabel(origin) || endpoint

  if (!fabric) {
    return (
      <p className="divergence__note-inline" data-testid="divergence-fabric-state" data-state="reading">
        Reading the fabric proxy at {shown}…
      </p>
    )
  }

  if (fabric.problem) {
    const diagnosis = crossOriginDiagnosis(fabric.problem, pageOrigin, origin)
    return (
      <div
        className="fabric-panel fabric-panel--problem divergence__fabric-state"
        data-testid="divergence-fabric-state"
        data-state="problem"
        data-code={fabric.problem.code}
        role="status"
      >
        <p>{fabricProblemMessage(fabric.problem, shown)}</p>
        <p className="fabric-note">
          Its nodes cannot be listed, so type the labels the proxy knows them by. The proxy address is set on the Cluster page.
        </p>
        <CorsHint pageOrigin={pageOrigin} diagnosis={diagnosis} />
      </div>
    )
  }

  if (fabric.detail === 'withheld') {
    return (
      <div
        className="fabric-panel fabric-panel--withheld divergence__fabric-state"
        data-testid="divergence-fabric-state"
        data-state="withheld"
        role="status"
      >
        <p>{DETAIL_WITHHELD_REASON}</p>
        <p className="fabric-note">
          You can still compare: type each node's label as the proxy knows it. The proxy resolves the label, and
          refuses by name if it has no such node.
        </p>
      </div>
    )
  }

  if (fabric.nodes && fabric.nodes.length === 0) {
    return (
      <p className="divergence__note-inline" data-testid="divergence-no-nodes" data-state="empty">
        This proxy answered, and it has no nodes to compare. Give it two
        with <code>camelid fabric serve --node LABEL=HOST:PORT</code>.
      </p>
    )
  }

  return null
}

/* One side's node and model. The node is a picker over what the proxy
   disclosed, or free text when it disclosed nothing: the proxy resolves labels
   itself, so a typed one is as good as a picked one. The model is a picker over
   what that node reported holding, with free text as well, because the list can
   be withheld and the proxy remains the authority on what is really there. */
function SidePicker({ side, label, nodes, fabric, value, onChange }) {
  const choices = modelChoices(fabric, value.node)
  const absent = modelLooksAbsent(choices, value.model)

  return (
    <fieldset className="divergence-pick" data-side={side}>
      <legend>{label}</legend>

      <label>
        Node
        {nodes ? (
          <select
            value={value.node}
            onChange={(event) => onChange({ node: event.target.value, model: '' })}
            required
          >
            <option value="">choose a node…</option>
            {nodes.map((node) => (
              <option key={node.label} value={node.label}>
                {node.label} · {node.engine || 'engine unknown'}
              </option>
            ))}
          </select>
        ) : (
          <input
            value={value.node}
            onChange={(event) => onChange({ ...value, node: event.target.value })}
            placeholder="node label"
            spellCheck="false"
            autoComplete="off"
            required
            data-testid={`node-input-${side}`}
          />
        )}
      </label>

      <label>
        Model
        {choices.kind === 'listed' ? (
          <select
            value={choices.models.includes(value.model) ? value.model : ''}
            onChange={(event) => onChange({ ...value, model: event.target.value })}
            required
          >
            <option value="">choose a model…</option>
            {choices.models.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        ) : (
          <input
            value={value.model}
            onChange={(event) => onChange({ ...value, model: event.target.value })}
            placeholder="model id"
            spellCheck="false"
            required
          />
        )}
      </label>

      <p className="divergence-pick__note" data-choices={choices.kind}>
        {choices.kind === 'listed' && `${choices.models.length} model(s) reported by this node`}
        {choices.kind === 'withheld' && 'This proxy does not disclose its node detail, so its models cannot be listed. Type the id.'}
        {choices.kind === 'unread' && 'The fabric could not be read, so this node\'s models cannot be listed. Type the id.'}
        {choices.kind === 'not_ready' && `This node is not serving${choices.reason ? `: ${choices.reason}` : ''}, so it reports no models.`}
        {choices.kind === 'unknown' && <Unknown why="This node reported no model list." />}
        {choices.kind === 'no_node' && (nodes ? 'Choose a node to see what it holds.' : 'Type the node label first.')}
      </p>

      {absent && (
        <p className="divergence-pick__warn" data-testid={`model-warning-${side}`}>
          This node did not report holding <code>{value.model}</code>. You can still try — the proxy
          decides, and names what the node actually holds if it refuses.
        </p>
      )}
    </fieldset>
  )
}

/* Which weights this side serves, where its engine said, and whether its
   engine checked them against the other side's. */
function Weights({ side }) {
  const { weightsDigest: digest, weightsCheck: check } = side
  const testid = `weights-${side.label || 'unlabelled'}`
  return (
    <>
      <p className="divergence-side__runtime" data-testid={testid} data-weights={digest.kind || (digest.reported ? 'unknown' : 'not_reported')}>
        {digest.kind === 'published' && <>GGUF file sha256 <code>{digest.digest}</code> <span className="divergence-side__runtime-note">via {digest.source}</span></>}
        {digest.kind === 'unavailable' && <>GGUF file digest not published: {digest.reason}</>}
        {!digest.kind && digest.reported && <Unknown why="The proxy sent a weights-digest kind this build does not recognise." />}
        {!digest.reported && <Unknown why="This proxy predates weights digests, so it read none.">weights digest not reported</Unknown>}
      </p>
      {check?.kind === 'enforced' && (
        <p className="divergence-side__runtime" data-testid={`${testid}-check`} data-check="enforced">
          Every run was bound to sha256 <code>{check.expected}</code>, and the engine served them.
        </p>
      )}
      {check?.kind === 'refused' && (
        <p className="divergence-side__reported" data-testid={`${testid}-check`} data-check="refused">
          This node refused runs bound to sha256 <code>{check.expected}</code>: its loaded GGUF file is other bytes.
        </p>
      )}
    </>
  )
}

function Side({ side }) {
  if (!side) return null
  const stability = side.stability.kind
  return (
    <section className="divergence-side" data-node-label={side.label || ''} data-engine={side.engine || ''}>
      <h3>
        {side.label || <Unknown why="This side arrived without a label." />}
        <span className="divergence-side__engine">
          {side.engine || <Unknown why="This side arrived without an engine." />}
          {' '}
          {side.engineVersion || <Unknown why="This engine publishes no version this build can read." />}
        </span>
      </h3>

      <p className="divergence-side__model">
        asked for{' '}
        {side.model ? <code>{side.model}</code> : <Unknown why="This side arrived without the model it was asked for." />}
      </p>

      {/* A node answering under another name may be serving other weights, and
          nothing else on this page would show it. */}
      {reportedModelMismatch(side) && (
        <p className="divergence-side__reported" data-testid={`reported-model-${side.label || 'unlabelled'}`}>
          This node's response named <code>{side.reported.model}</code>, not the requested <code>{side.model}</code>.
          The fabric cannot tell whether those are the same weights.
        </p>
      )}
      {side.reported.state === 'unnamed' && (
        <p className="divergence-side__runtime" data-testid={`reported-model-${side.label || 'unlabelled'}`}>
          Its response did not name a model.
        </p>
      )}

      <Weights side={side} />

      {side.runtime && (
        <p className="divergence-side__runtime">
          runtime: {side.runtime}
          <span className="divergence-side__runtime-note">
            {' '}— the inference runtime, not the engine's own version
          </span>
        </p>
      )}

      <p className="divergence-side__stability" data-stability={stability || 'unknown'}>
        {stability === 'stable' && (
          <>agreed with itself across {side.samples.length} runs · <code>{side.settledDigest}</code></>
        )}
        {stability === 'unstable' && (
          <>did <strong>not</strong> agree with itself: {side.stability.distinctAnswers} distinct answers across {side.samples.length} runs</>
        )}
        {stability === 'unmeasured' && <>run once, so it was never shown to repeat itself</>}
        {!stability && <Unknown why="The proxy sent a stability this build does not recognise." />}
      </p>

      <ol className="divergence-side__samples">
        {side.samples.map((sample, index) => (
          <li key={`${sample.sha256}-${index}`}>
            <span className="divergence-side__timing">
              {sample.elapsedMs === null ? <Unknown why="No timing was recorded." /> : `${sample.elapsedMs} ms`}
            </span>
            <pre>{sample.text}</pre>
          </li>
        ))}
      </ol>
    </section>
  )
}

function Template({ side }) {
  if (!side) return null
  const { kind, source, template, detail } = side.advertisedTemplate
  return (
    <section className="divergence-template" data-template-kind={kind || 'unknown'}>
      <h4>{side.label} · {kind === 'captured' ? `advertised via ${source}` : 'no advertised template'}</h4>
      {kind === 'captured' && <pre>{template}</pre>}
      {kind === 'not_exposed' && <p className="divergence-note">{detail}</p>}
      {kind === 'unavailable' && <p className="divergence-note">Could not be read: {detail}</p>}
      {!kind && <p className="divergence-note"><Unknown why="The proxy sent a template kind this build does not recognise." /></p>}
    </section>
  )
}

/* The prompt an engine built from the messages this comparison sent, which is
   the only thing on this page that shows what an engine applied. */
function Rendered({ side }) {
  if (!side) return null
  const { kind, reported, source, text, reason } = side.renderedPrompt
  return (
    <section
      className="divergence-template"
      data-testid={`rendered-${side.label || 'unlabelled'}`}
      data-rendered-kind={kind || (reported ? 'unknown' : 'not_reported')}
    >
      <h4>{side.label} · {kind === 'captured' ? `rendered via ${source}` : 'not captured'}</h4>
      {kind === 'captured' && <pre>{text}</pre>}
      {kind === 'unavailable' && <p className="divergence-note">{reason}</p>}
      {!kind && reported && <p className="divergence-note"><Unknown why="The proxy sent a rendered-prompt kind this build does not recognise." /></p>}
      {!reported && (
        <p className="divergence-note">
          <Unknown why="This proxy predates rendered prompts, so it captured none.">not reported</Unknown>
        </p>
      )}
    </section>
  )
}

function DiffLine({ line }) {
  const marker = line.eol ? EOL_MARKER[line.eol] : null
  return (
    <span className="divergence-diff__line" data-op={line.op || 'unknown'} data-eol={line.eol || undefined}>
      {line.op === 'removed' ? '-' : line.op === 'added' ? '+' : ' '} {line.text}
      {marker && (
        <span className="divergence-diff__eol" data-eol-marker={line.eol} title={marker.why}>{marker.text}</span>
      )}
      {line.eolUnrecognised && (
        <span className="divergence-diff__eol">
          <Unknown why="The proxy sent a line ending this build does not recognise.">line ending unknown</Unknown>
        </span>
      )}
      {'\n'}
    </span>
  )
}

export default function DivergenceView() {
  const { endpoint, fabric } = useFabric()
  const [left, setLeft] = useState({ node: '', model: '' })
  const [right, setRight] = useState({ node: '', model: '' })
  const [prompt, setPrompt] = useState('What is 7 plus 5?')
  const [repetitions, setRepetitions] = useState('2')
  const [maxTokens, setMaxTokens] = useState(String(COMPARE_MAX_TOKENS.fallback))
  // A secret, so it lives in this component's memory and nowhere else: not
  // browser storage, not the URL. Leaving the page forgets it.
  const [clientKey, setClientKey] = useState('')
  const [asserted, setAsserted] = useState(false)
  const { phase, comparison, problem, requested, run } = useDivergence(endpoint)

  const pageOrigin = currentPageOrigin()
  // Only a disclosed list can be offered as a choice. Anything else — not read
  // yet, unreadable, withheld — is typed, and FabricState says which it is.
  const nodes = fabric?.detail === 'disclosed' ? fabric.nodes : null
  const mustAssert = needsIdentityAssertion(left.model, right.model)
  const blocked = mustAssert && !asserted

  const caveats = comparison ? comparisonCaveats(comparison) : []
  const uncontrolled = comparison ? uncontrolledStatements(comparison) : []
  const templates = comparison ? templateDivergence(comparison) : null
  const templateNote = comparison ? advertisedTemplateNote(comparison) : null
  const rendered = comparison ? renderedPromptComparison(comparison) : null
  const identity = comparison ? identityStatement(comparison) : null
  const cap = comparison ? tokenCapStatement(comparison, requested?.max_tokens ?? null) : null
  const problemMessage = problem ? problemText(problem) : null
  const requestDiagnosis = problem
    ? crossOriginDiagnosis(problem, pageOrigin, normalizeEndpoint(endpoint))
    : null

  return (
    <div className="camelid-page divergence">
      <header className="divergence__head">
        <h1>Compare two nodes</h1>
        <p className="divergence__lede">
          The same prompt on two machines. This never says which answer is correct — it reports
          what differed, how both sides were sampled, the chat template each engine advertises and,
          where an engine can render one without generating, the exact prompt it built.
        </p>
      </header>

      <form
        className="divergence__form"
        onSubmit={(event) => {
          event.preventDefault()
          if (blocked) return
          run(buildCompareRequest({ left, right, prompt, repetitions, maxTokens }), { clientKey })
        }}
      >
        <div className="divergence__picks">
          <SidePicker side="left" label="Left" nodes={nodes} fabric={fabric} value={left} onChange={setLeft} />
          <SidePicker side="right" label="Right" nodes={nodes} fabric={fabric} value={right} onChange={setRight} />
        </div>

        <div className="divergence__fabric-slot">
          <FabricState fabric={fabric} endpoint={endpoint} pageOrigin={pageOrigin} />
        </div>

        {/* Two names are not evidence of the same weights, so the pairing is
            offered and the claim stays the operator's to make. */}
        {mustAssert && (
          <label className="divergence__assert" data-testid="divergence-assert">
            <input type="checkbox" checked={asserted} onChange={(event) => setAsserted(event.target.checked)} />
            <span>
              These are two different ids — <code>{left.model}</code> and <code>{right.model}</code>.
              Engines name weights differently. Where both engines publish a digest of the weights
              they serve, the result checks it; otherwise nothing here can verify they match. Tick to
              declare they are the same weights; the result records that you asserted it.
            </span>
          </label>
        )}

        <label className="divergence__prompt">
          Prompt
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} required rows={2} />
        </label>

        <label>
          Runs each
          <input type="number" min="1" max="5" value={repetitions} onChange={(event) => setRepetitions(event.target.value)} />
          <span className="divergence__hint">Two is the fewest that can show a node repeats itself.</span>
        </label>

        <label>
          Token cap
          <input
            type="number"
            min={COMPARE_MAX_TOKENS.min}
            max={COMPARE_MAX_TOKENS.max}
            value={maxTokens}
            onChange={(event) => setMaxTokens(event.target.value)}
            data-testid="divergence-max-tokens"
          />
          <span className="divergence__hint">
            Each answer stops here, and only what was generated is compared.
          </span>
        </label>

        <label>
          Client key (optional)
          <input
            type="password"
            value={clientKey}
            onChange={(event) => setClientKey(event.target.value)}
            autoComplete="off"
            spellCheck="false"
            data-testid="divergence-client-key"
          />
          <span className="divergence__hint">
            Only if the proxy has client keys. Sent as a bearer token; kept in this page's memory and never stored.
          </span>
        </label>

        <button type="submit" disabled={phase === 'running' || blocked}>
          {phase === 'running' ? 'Asking both nodes…' : 'Compare'}
        </button>
      </form>

      {problemMessage && (
        <div className="divergence__problem" data-testid="divergence-problem" data-code={problem.code} role="alert">
          <p>{problemMessage}</p>
          <CorsHint pageOrigin={pageOrigin} diagnosis={requestDiagnosis} />
        </div>
      )}

      {comparison && (
        <>
          <p
            className="divergence__verdict"
            data-testid="divergence-verdict"
            data-verdict={comparison.verdict.kind || 'unknown'}
            data-tone={VERDICT_TONE[comparison.verdict.kind] || 'refused'}
            data-attributable={String(isAttributable(comparison))}
          >
            {verdictHeadline(comparison)}
          </p>

          <p className="divergence__cap" data-testid="divergence-token-cap">
            {cap || (
              <Unknown why="This proxy did not report the token cap it applied, so how much of each answer was compared is not known.">
                token cap unknown
              </Unknown>
            )}
          </p>

          <p className="divergence__plan">
            temperature {comparison.plan.temperature ?? <Unknown why="Not reported." />}
            {' · seed '}
            {comparison.plan.seed === null ? 'none' : comparison.plan.seed}
            {' · runs each '}
            {comparison.plan.repetitions ?? <Unknown why="Not reported." />}
            {' · prompt sha256 '}
            <code>{comparison.promptSha256 || '-'}</code>
          </p>

          {/* What the result rests on, stated on the result: an operator's claim
              about the weights must travel with every result that depends on it. */}
          <section className="divergence__basis" data-testid="divergence-basis" data-identity={identity.kind}>
            <p data-testid="divergence-identity" data-identity={identity.kind}>
              <strong>Model identity.</strong> {identity.text}
            </p>
            <div data-testid="divergence-uncontrolled">
              <p>
                <strong>Not controlled.</strong>{' '}
                {!comparison.uncontrolledReported && (
                  <Unknown why="This proxy did not report which controls it could not apply." />
                )}
                {comparison.uncontrolledReported && uncontrolled.length === 0
                  && 'Nothing: every control this comparison asked for was sent to both engines.'}
              </p>
              {uncontrolled.length > 0 && (
                <ul className="divergence__uncontrolled">
                  {uncontrolled.map((item) => (
                    <li key={item.name} data-uncontrolled={item.name}>
                      <strong>{item.name}</strong> — {item.text}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          {caveats.length > 0 && (
            <ul className="divergence__caveats" data-testid="divergence-caveats">
              {caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}
            </ul>
          )}

          <div className="divergence__sides">
            <Side side={comparison.left} />
            <Side side={comparison.right} />
          </div>

          <section className="divergence__diff" data-testid="divergence-diff" data-diff-kind={comparison.diff.kind || 'unknown'}>
            <h2>Difference</h2>
            {comparison.diff.kind === 'identical' && <p className="divergence-note">Byte-identical; there is nothing to show.</p>}
            {comparison.diff.kind === 'declined' && <p className="divergence-note">{comparison.diff.reason}</p>}
            {comparison.diff.kind === 'lines' && (
              <pre className="divergence-diff">
                {comparison.diff.lines.map((line, index) => <DiffLine key={index} line={line} />)}
              </pre>
            )}
            {!comparison.diff.kind && <p className="divergence-note"><Unknown why="The proxy sent a diff shape this build does not recognise." /></p>}
          </section>

          <section className="divergence__templates" data-testid="divergence-templates">
            <h2>Advertised chat templates</h2>
            <p className="divergence-note">
              What each engine publishes as this model&apos;s template. Not proof of the prompt it built.
            </p>
            {templateNote && (
              <p
                className={`divergence-note${templates?.differ || templates?.unexplained ? ' divergence-note--strong' : ''}`}
                data-testid="divergence-template-note"
                data-unexplained={String(Boolean(templates?.unexplained))}
              >
                {templateNote}
              </p>
            )}
            <div className="divergence__sides">
              <Template side={comparison.left} />
              <Template side={comparison.right} />
            </div>
          </section>

          <section className="divergence__templates" data-testid="divergence-rendered">
            <h2>Rendered prompts</h2>
            <p className="divergence-note">
              The prompt each engine built from exactly the messages this comparison sent — the only
              evidence here of what an engine applied.
              {rendered && (rendered.same
                ? ' Both rendered prompts are byte-identical.'
                : ' The two rendered prompts differ.')}
            </p>
            <div className="divergence__sides">
              <Rendered side={comparison.left} />
              <Rendered side={comparison.right} />
            </div>
          </section>
        </>
      )}
    </div>
  )
}
