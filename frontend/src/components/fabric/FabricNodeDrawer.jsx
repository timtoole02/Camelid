import { useEffect, useRef, useState } from 'react'
import { Button } from '../ui/Button'
import { Chip } from '../ui/Chip'
import { IconClose, IconCopy } from '../ui/icons'
import { copyText } from '../../lib/markdown.jsx'
import { Unknown } from './Unknown'

/* Everything the proxy told us about one node, and nothing else.

   Notably absent, because `/v1/health` does not carry them: the transport the
   proxy used to reach this node, and when the proxy last probed it. Inventing
   either would be the exact defect this view exists to remove. The one time we
   can speak is our own fetch, so "as of" is stamped by the client that asked. */

const STATE_TONE = { ready: 'ready', not_ready: 'warn', unreachable: 'error' }

const PROVENANCE_LABEL = {
  measured: 'measured here',
  declared: 'from the API',
  not_probed: 'not checked',
}

/* An answer and how the proxy came by it. `not checked` is deliberately not
   rendered as a "no": an unmeasured backend is not a broken one. */
function CapabilityRow({ capability }) {
  const verdict = capability.supported === null
    ? <Unknown why="Nobody has checked this, and this build will not guess.">unknown</Unknown>
    : <span className={capability.supported ? 'fabric-cap__yes' : 'fabric-cap__no'}>
      {capability.supported ? 'yes' : 'no'}
    </span>
  return (
    <li className="fabric-cap" data-capability={capability.name} data-provenance={capability.provenance || 'unknown'}>
      <span className="fabric-cap__name">{capability.name.replace(/_/g, ' ')}</span>
      <span className="fabric-cap__verdict">{verdict}</span>
      <span className="fabric-cap__how" title={capability.detail || undefined}>
        {capability.provenance
          ? (PROVENANCE_LABEL[capability.provenance] || capability.provenance)
          : <Unknown why="The proxy sent a provenance this build does not recognise." />}
      </span>
    </li>
  )
}

function Row({ label, children }) {
  return (
    <div className="fabric-detail__row">
      <span className="fabric-detail__key">{label}</span>
      <span className="fabric-detail__value">{children}</span>
    </div>
  )
}

function CopyButton({ value, label }) {
  const [done, setDone] = useState(false)
  const [failed, setFailed] = useState(false)
  return (
    <Button
      variant="ghost"
      size="sm"
      icon={<IconCopy size={14} />}
      onClick={async () => {
        const ok = await copyText(value)
        setDone(ok)
        setFailed(!ok)
        window.setTimeout(() => { setDone(false); setFailed(false) }, 2000)
      }}
    >
      {failed ? 'Copy failed' : (done ? 'Copied' : label)}
    </Button>
  )
}

export function FabricNodeDrawer({ node, checkedAt, onClose }) {
  const panelRef = useRef(null)
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose
  const label = node ? node.label : null
  const open = Boolean(node)

  // Focus follows the operator into the detail they asked for. Keyed on which
  // node is open, not on the node object: the 5s poll replaces that object, and
  // re-focusing on every poll would pull focus away from wherever it had gone.
  useEffect(() => {
    if (open) panelRef.current?.focus()
  }, [open, label])

  useEffect(() => {
    if (!open) return undefined
    const onKey = (event) => {
      if (event.key === 'Escape' && !event.defaultPrevented) onCloseRef.current?.()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open])

  if (!node) return null
  const tone = node.state ? STATE_TONE[node.state] : 'neutral'
  const healthUrl = node.authority ? `http://${node.authority}/v1/health` : null

  return (
    <aside
      ref={panelRef}
      tabIndex={-1}
      className="fabric-detail"
      aria-label={`Node ${node.label || 'detail'}`}
      data-testid="fabric-detail"
    >
      <header className="fabric-detail__head">
        <div>
          <h2 className="fabric-detail__title">{node.label || 'Unlabelled node'}</h2>
          <Chip tone={tone} dot>{(node.state || 'unknown').replace('_', ' ')}</Chip>
        </div>
        <Button variant="ghost" size="sm" icon={<IconClose size={15} />} onClick={onClose} aria-label="Close node detail" />
      </header>

      <div className="fabric-detail__body">
        <Row label="Address">
          {node.authority || <Unknown why="This entry arrived without an address." />}
        </Row>

        <Row label="Engine">
          {node.engine || <Unknown why="This entry arrived without a declared engine." />}
          {node.placeable === false && (
            <p className="fabric-detail__note">
              This fabric reads this node but does not send work to it
              {node.placementBlockers && node.placementBlockers.length > 0
                ? `: it ${node.placementBlockers.join('; it ')}.`
                : '.'}
            </p>
          )}
        </Row>

        {node.state !== 'ready' && (
          <Row label="Reason">
            {node.reason
              ? <code className="fabric-detail__reason">{node.reason}</code>
              : <Unknown why="The proxy reported this state without a reason." />}
          </Row>
        )}

        <Row label="Active model">
          {node.state !== 'ready'
            ? <Unknown why="Only a ready node reports which model it is serving." />
            : (node.activeModelId || <span className="fabric-row__muted">no model loaded</span>)}
        </Row>

        <Row label="Execution backend">
          {node.backend || <Unknown why="Only a ready node reports its execution backend." />}
        </Row>

        <Row label="Node build">
          {node.version || <Unknown why="Only a ready node reports its version." />}
        </Row>

        <Row label="Load">
          {node.state !== 'ready' || node.inFlight === null
            ? <Unknown why="Only a ready node reports load." />
            : (
              <>
                {node.inFlight} in flight
                {node.waiting !== null && <> · {node.waiting} waiting</>}
                <p className="fabric-detail__note">
                  A gauge of work in flight, not a capacity — a node never publishes its queue bound.
                </p>
              </>
            )}
        </Row>

        <Row label="Probe round-trip">
          {node.latencyMs === null
            ? <Unknown why="No probe round-trip was recorded." />
            : <>{node.latencyMs} ms <span className="fabric-detail__note">one sample; routing does not rank on it</span></>}
        </Row>

        <Row label="As of">
          {checkedAt
            ? new Date(checkedAt).toLocaleTimeString()
            : <Unknown why="This page has not completed a read yet." />}
          <p className="fabric-detail__note">
            When this page last read the proxy. The proxy does not publish when it last probed the node.
          </p>
        </Row>
      </div>

      {node.capabilities && (
        <section className="fabric-detail__caps" data-testid="fabric-capabilities">
          <h3>What this engine can be asked</h3>
          <ul className="fabric-cap-list">
            {node.capabilities.map((capability) => (
              <CapabilityRow key={capability.name} capability={capability} />
            ))}
          </ul>
          <p className="fabric-detail__note">
            A measurement is about the exact build it was taken on, so a neighbouring
            version inherits nothing from it.
          </p>
        </section>
      )}

      {healthUrl && (
        <footer className="fabric-detail__foot">
          <p className="fabric-detail__note">
            Ask this node directly — useful when the proxy cannot reach it but you can.
          </p>
          <div className="fabric-detail__actions">
            <code className="fabric-detail__cmd">{healthUrl}</code>
            <CopyButton value={healthUrl} label="Copy health URL" />
          </div>
        </footer>
      )}
    </aside>
  )
}

export default FabricNodeDrawer
