import { useState } from 'react'
import { Unknown } from '../components/fabric/Unknown.jsx'
import { useDivergence } from '../hooks/useDivergence.js'
import { useFabric } from '../hooks/useFabric.js'
import {
  comparisonCaveats,
  isAttributable,
  modelChoices,
  modelLooksAbsent,
  needsIdentityAssertion,
  templateDivergence,
  verdictHeadline,
} from '../lib/divergenceModel.js'
import '../styles/divergence.css'

const VERDICT_TONE = {
  identical: 'same',
  divergent: 'differ',
  different_models: 'refused',
  not_attributable: 'refused',
}

/* One side's node and model. The model is a picker over what the proxy said
   that node holds, with free text as well: the list can be withheld, and the
   proxy remains the authority on what is really there. */
function SidePicker({ side, label, nodes, fabric, value, onChange }) {
  const choices = modelChoices(fabric, value.node)
  const absent = modelLooksAbsent(choices, value.model)

  return (
    <fieldset className="divergence-pick" data-side={side}>
      <legend>{label}</legend>

      <label>
        Node
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
            required
          />
        )}
      </label>

      <p className="divergence-pick__note" data-choices={choices.kind}>
        {choices.kind === 'listed' && `${choices.models.length} model(s) reported by this node`}
        {choices.kind === 'withheld' && 'This proxy does not disclose its node detail, so its models cannot be listed. Type the id.'}
        {choices.kind === 'not_ready' && `This node is not serving${choices.reason ? `: ${choices.reason}` : ''}, so it reports no models.`}
        {choices.kind === 'unknown' && <Unknown why="This node reported no model list." />}
        {choices.kind === 'no_node' && 'Choose a node to see what it holds.'}
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
  const { kind, source, template, detail } = side.template
  return (
    <section className="divergence-template" data-template-kind={kind || 'unknown'}>
      <h4>{side.label} · {kind === 'captured' ? source : 'no template'}</h4>
      {kind === 'captured' && <pre>{template}</pre>}
      {kind === 'not_exposed' && <p className="divergence-note">{detail}</p>}
      {kind === 'unavailable' && <p className="divergence-note">Could not be read: {detail}</p>}
      {!kind && <p className="divergence-note"><Unknown why="The proxy sent a template kind this build does not recognise." /></p>}
    </section>
  )
}

export default function DivergenceView() {
  const { endpoint, fabric } = useFabric()
  const [left, setLeft] = useState({ node: '', model: '' })
  const [right, setRight] = useState({ node: '', model: '' })
  const [prompt, setPrompt] = useState('What is 7 plus 5?')
  const [repetitions, setRepetitions] = useState(2)
  const [asserted, setAsserted] = useState(false)
  const { phase, comparison, problem, run } = useDivergence(endpoint)

  const nodes = fabric?.nodes || []
  const mustAssert = needsIdentityAssertion(left.model, right.model)
  const blocked = mustAssert && !asserted

  const caveats = comparison ? comparisonCaveats(comparison) : []
  const templates = comparison ? templateDivergence(comparison) : null

  return (
    <div className="camelid-page divergence">
      <header className="divergence__head">
        <h1>Compare two nodes</h1>
        <p className="divergence__lede">
          The same prompt on two machines. This never says which answer is correct — it reports
          what differed, how both sides were sampled, and, where the engine exposes one, the chat
          template each applied.
        </p>
      </header>

      <form
        className="divergence__form"
        onSubmit={(event) => {
          event.preventDefault()
          if (blocked) return
          run({
            left: left.node,
            right: right.node,
            model: left.model,
            left_model: left.model,
            right_model: right.model,
            prompt,
            repetitions: Number(repetitions) || 2,
            temperature: 0,
          })
        }}
      >
        <div className="divergence__picks">
          <SidePicker side="left" label="Left" nodes={nodes} fabric={fabric} value={left} onChange={setLeft} />
          <SidePicker side="right" label="Right" nodes={nodes} fabric={fabric} value={right} onChange={setRight} />
        </div>

        {nodes.length === 0 && (
          <p className="divergence__note-inline" data-testid="divergence-no-nodes">
            No nodes to choose from yet. Set the proxy address on the Cluster page first.
          </p>
        )}

        {/* Two names are not evidence of the same weights, so the pairing is
            offered and the claim stays the operator's to make. */}
        {mustAssert && (
          <label className="divergence__assert" data-testid="divergence-assert">
            <input type="checkbox" checked={asserted} onChange={(event) => setAsserted(event.target.checked)} />
            <span>
              These are two different ids — <code>{left.model}</code> and <code>{right.model}</code>.
              Engines name weights differently and none of them publishes a digest this fabric can
              check, so nothing here can verify they match. Tick to declare they are the same
              weights; the result records that you asserted it.
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

        <button type="submit" disabled={phase === 'running' || blocked}>
          {phase === 'running' ? 'Asking both nodes…' : 'Compare'}
        </button>
      </form>

      {problem && (
        <p className="divergence__problem" data-testid="divergence-problem">
          {problem.code === 'refused' && problem.detail}
          {problem.code === 'unreachable' && `No fabric proxy answered. ${problem.detail}`}
          {problem.code === 'malformed' && problem.detail}
          {problem.code === 'bad_endpoint' && 'That proxy address cannot be used.'}
        </p>
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

          <p className="divergence__plan">
            temperature {comparison.plan.temperature ?? <Unknown why="Not reported." />}
            {' · seed '}
            {comparison.plan.seed === null ? 'none' : comparison.plan.seed}
            {' · prompt sha256 '}
            <code>{comparison.promptSha256 || '-'}</code>
          </p>

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
                {comparison.diff.lines.map((line, index) => (
                  <span key={index} className="divergence-diff__line" data-op={line.op || 'unknown'}>
                    {line.op === 'removed' ? '-' : line.op === 'added' ? '+' : ' '} {line.text}{'\n'}
                  </span>
                ))}
              </pre>
            )}
            {!comparison.diff.kind && <p className="divergence-note"><Unknown why="The proxy sent a diff shape this build does not recognise." /></p>}
          </section>

          <section className="divergence__templates">
            <h2>Templates applied</h2>
            {templates?.differ && (
              <p className="divergence-note divergence-note--strong">
                These two templates are not the same, which is usually the explanation.
              </p>
            )}
            {templates && !templates.differ && (
              <p className="divergence-note">Both nodes applied the same template.</p>
            )}
            <div className="divergence__sides">
              <Template side={comparison.left} />
              <Template side={comparison.right} />
            </div>
          </section>
        </>
      )}
    </div>
  )
}
