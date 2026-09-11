import { Chip } from '../ui/Chip'
import { Unknown } from './Unknown'

/* The node list. Authoritative: this is the fabric, not a drawing of one.
   Every cell traces to a field in the proxy's `/v1/health` answer, and a field
   we were not sent renders as an explicit unknown rather than a plausible zero. */

const STATE_PRESENTATION = {
  ready: { tone: 'ready', label: 'ready' },
  not_ready: { tone: 'warn', label: 'not ready' },
  unreachable: { tone: 'error', label: 'unreachable' },
}

function StateCell({ node }) {
  const presentation = node.state ? STATE_PRESENTATION[node.state] : null
  if (!presentation) {
    return <Unknown why="The proxy reported a state this build does not recognise." />
  }
  return (
    <div className="fabric-row__state">
      <Chip tone={presentation.tone} dot>{presentation.label}</Chip>
      {node.reason && <span className="fabric-row__reason" title={node.reason}>{node.reason}</span>}
    </div>
  )
}

function EngineCell({ node }) {
  if (!node.engine) {
    return <Unknown why="This entry arrived without a declared engine." />
  }
  // `placeable` is the proxy's answer, not ours: a healthy node can still be
  // one this fabric does not route to. And one it does route to can still be
  // one it cannot fully see, when the proxy was told to accept that.
  const routed = node.placeable
  const accepting = node.placementBlockers ? node.placementBlockers.length : 0
  return (
    <span className="fabric-row__engine">
      <span className="fabric-row__engine-name">{node.engine}</span>
      {routed === false && (
        <span
          className="fabric-row__engine-note"
          title="This fabric reads this node but does not send work to it."
        >
          not routed to
        </span>
      )}
      {routed === true && accepting > 0 && (
        <span
          className="fabric-row__engine-note fabric-row__engine-note--accepting"
          title="This proxy sends work here and accepts what it cannot see about this engine. Open the node for the list."
        >
          routed to · accepting {accepting} {accepting === 1 ? 'limit' : 'limits'}
        </span>
      )}
    </span>
  )
}

function ModelCell({ node }) {
  if (node.state !== 'ready') {
    return <Unknown why="Only a ready node reports which models it is serving." />
  }
  const models = node.models
  if (!models || models.length === 0) {
    return <span className="fabric-row__muted">no model loaded</span>
  }
  if (models.length === 1) {
    return <span className="fabric-row__model" title={models[0]}>{models[0]}</span>
  }
  // Several models means the engine can serve any of them, not that it picked
  // one; naming a single "active" model here would be an invention.
  return (
    <span className="fabric-row__model" title={models.join('\n')}>
      {models.length} models
    </span>
  )
}

function LoadCell({ node }) {
  if (node.state !== 'ready') {
    return <Unknown why="Only a ready node reports load." />
  }
  if (node.inFlight === null) {
    return <Unknown why={`This engine (${node.engine || 'unknown'}) publishes no load figure.`} />
  }
  const waiting = node.waiting === null ? null : node.waiting
  return (
    <span
      className="fabric-row__load"
      title={
        'Jobs accepted and not yet finished. This is a gauge of work in flight, '
        + 'not a capacity: a node never reports its queue bound.'
      }
    >
      {node.inFlight} in flight
      {waiting !== null && waiting > 0 ? ` · ${waiting} waiting` : ''}
    </span>
  )
}

function LatencyCell({ node }) {
  if (node.latencyMs === null) return <Unknown why="No probe round-trip was recorded." />
  return <span className="fabric-row__latency">{node.latencyMs} ms</span>
}

/* A row is a row, so assistive tech can read the table as a table; the thing
   that opens a node's detail is a real button in its first cell. A button
   given `role="row"` loses its button semantics, so nobody using a screen
   reader was told the rows could be opened. The row keeps a click handler
   only as a larger mouse target for the same action. */
export function FabricNodeTable({ nodes, selectedLabel, onSelect }) {
  return (
    <div className="fabric-table" role="table" aria-label="Fabric nodes">
      <div className="fabric-table__head" role="row">
        <span role="columnheader">Node</span>
        <span role="columnheader">Engine</span>
        <span role="columnheader">State</span>
        <span role="columnheader">Address</span>
        <span role="columnheader">Serving</span>
        <span role="columnheader">Load</span>
        <span role="columnheader">Probe</span>
      </div>
      {nodes.map((node, index) => {
        const key = node.label || `node-${index}`
        const selected = Boolean(selectedLabel && selectedLabel === node.label)
        return (
          <div
            key={key}
            role="row"
            className={`fabric-row${selected ? ' is-selected' : ''}`}
            onClick={() => onSelect?.(node.label)}
            data-node-label={node.label || ''}
            data-node-state={node.state || 'unknown'}
            data-node-engine={node.engine || 'unknown'}
          >
            <span role="cell" data-label="Node" className="fabric-row__label">
              <button
                type="button"
                className="fabric-row__open"
                aria-expanded={selected}
                aria-label={`Show detail for node ${node.label || 'without a label'}`}
                onClick={(event) => {
                  event.stopPropagation()
                  onSelect?.(node.label)
                }}
              >
                {node.label || <Unknown why="This entry arrived without a label." />}
              </button>
            </span>
            <span role="cell" data-label="Engine"><EngineCell node={node} /></span>
            <span role="cell" data-label="State"><StateCell node={node} /></span>
            <span role="cell" data-label="Address" className="fabric-row__authority">
              {node.authority || <Unknown why="This entry arrived without an address." />}
            </span>
            <span role="cell" data-label="Serving"><ModelCell node={node} /></span>
            <span role="cell" data-label="Load"><LoadCell node={node} /></span>
            <span role="cell" data-label="Probe"><LatencyCell node={node} /></span>
          </div>
        )
      })}
    </div>
  )
}

export default FabricNodeTable
