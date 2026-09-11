import { useEffect, useRef, useState } from 'react'
import { Button } from '../ui/Button'
import { Chip } from '../ui/Chip'
import { CopyableCommand } from './CopyableCommand'
import {
  mixedModeAcceptance,
  provenanceLabel,
  requirementLimits,
  routingCommand,
} from '../../lib/fabricModel.js'

/* Screen E: where this proxy places work, and what the other mode would mean.

   The page cannot switch a running proxy, and pretends nothing: choosing the
   other mode only decides which command is shown, after a confirmation that
   lists what that mode accepts. Every sentence describing the proxy's
   behaviour is one the proxy sent, rendered verbatim, because a sentence
   written here would describe this page's build rather than the proxy's.
   The choice lives in component state only; nothing is stored. */

/* Only a node that answered can have published no version. One that did not
   answer was never asked, so its version is unknown. */
function versionText(node) {
  if (node.version) return ` ${node.version}`
  return node.state === 'ready' ? ', version not published' : ', version unknown'
}

function NodeLine({ node }) {
  return (
    <li className="fabric-routing__node" data-node-label={node.label || ''}>
      <span className="fabric-routing__node-label">{node.label || 'unlabelled node'}</span>
      <span className="fabric-routing__node-engine">
        {node.engine || 'engine not stated'}
        {versionText(node)}
      </span>
      {node.detail && <span className="fabric-routing__node-detail">{node.detail}</span>}
      {node.provenance && (
        <span className="fabric-routing__node-how">{provenanceLabel(node.provenance)}</span>
      )}
    </li>
  )
}

/* What the proxy says it accepts, one section per reason, and what it will
   still never send. Shared by the confirmation and the live allowed state. */
function Acceptance({ groups, limits }) {
  return (
    <>
      {groups.map((group) => (
        <section key={group.key} className="fabric-routing__accept" data-blocker-key={group.key}>
          <h4 className="fabric-routing__blocker">{group.blocker}</h4>
          <p className="fabric-routing__consequence">{group.consequence}</p>
          <ul className="fabric-routing__nodes">
            {group.nodes.map((node) => <NodeLine key={`${group.key}-${node.label}`} node={node} />)}
          </ul>
        </section>
      ))}
      {limits.length > 0 && (
        <ul className="fabric-routing__limits" data-testid="fabric-routing-limits">
          {limits.flatMap((limit) => limit.nodes.map((node) => (
            <li key={`${limit.key}-${node.label}`} data-limit-key={limit.key} data-node-label={node.label || ''}>
              <span className="fabric-routing__node-label">{node.label || 'unlabelled node'}</span>
              {' '}
              {node.consequence}
            </li>
          )))}
        </ul>
      )}
    </>
  )
}

function focusables(root) {
  if (!root) return []
  return [...root.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
    .filter((element) => !element.disabled)
}

function Confirmation({ placement, groups, limits, onCancel, onConfirm }) {
  const dialogRef = useRef(null)

  useEffect(() => {
    focusables(dialogRef.current)[0]?.focus()
  }, [])

  const onKeyDown = (event) => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      onCancel()
      return
    }
    if (event.key !== 'Tab') return
    const items = focusables(dialogRef.current)
    if (items.length === 0) return
    const first = items[0]
    const last = items[items.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  const consequences = placement.consequences || []
  const adds = placement.modelsIfMixed || []
  return (
    <div className="fabric-routing__backdrop">
      <div
        ref={dialogRef}
        className="fabric-routing__dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="fabric-routing-confirm-title"
        data-testid="fabric-routing-confirm"
        onKeyDown={onKeyDown}
      >
        <h3 id="fabric-routing-confirm-title">Also placing on other engines would accept this</h3>
        {groups.length === 0 ? (
          <p className="fabric-note" data-testid="fabric-routing-nothing">
            The flag would change nothing about the nodes this proxy has now.
          </p>
        ) : (
          <Acceptance groups={groups} limits={limits} />
        )}
        {consequences.length > 0 && (
          <ul className="fabric-routing__consequences" data-testid="fabric-routing-consequences">
            {consequences.map((consequence) => (
              <li key={consequence.key} data-consequence-key={consequence.key}>{consequence.text}</li>
            ))}
          </ul>
        )}
        {adds.length > 0 && (
          <div className="fabric-routing__adds" data-testid="fabric-routing-adds">
            <p className="fabric-note">This would start serving:</p>
            <ul>
              {adds.map((model) => <li key={model}><code>{model}</code></li>)}
            </ul>
          </div>
        )}
        <div className="fabric-routing__actions">
          <Button variant="ghost" onClick={onCancel}>Cancel</Button>
          <Button variant="tonal" onClick={onConfirm}>Show the command</Button>
        </div>
      </div>
    </div>
  )
}

function CommandFor({ placement, target }) {
  const command = routingCommand(placement, target)
  if (!command) return null
  return (
    <div className="fabric-routing__command" data-testid="fabric-routing-command" data-target={target}>
      <p className="fabric-note">
        Restart the proxy with this, keeping the other flags it was started with.
        {target === 'camelid_only' ? ` Leave out ${placement.flag}.` : ''}
        {' '}This page cannot change a running proxy.
      </p>
      <CopyableCommand command={command} />
    </div>
  )
}

export function RoutingMode({ fabric }) {
  const [choice, setChoice] = useState('camelid_only')
  const [confirming, setConfirming] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const mixedRadioRef = useRef(null)

  const placement = fabric?.placement || null
  const nodes = fabric?.detail === 'disclosed' ? fabric.nodes : null
  // Every node must carry the field for the proxy to have described them all;
  // an older proxy sends none, and silence is not "nothing to accept".
  const described = Array.isArray(nodes) && nodes.every((node) => Array.isArray(node.placementBlockerDetail))
  const mode = nodes && described && placement ? placement.mixedEngines : null

  if (mode === null) {
    return (
      <section className="fabric-routing" data-testid="fabric-routing" data-mode="unknown">
        <h2>Routing</h2>
        <p className="fabric-note">
          This proxy did not say how it routes, so this page offers no way to change it.
        </p>
      </section>
    )
  }

  const groups = mixedModeAcceptance(nodes)
  const limits = requirementLimits(nodes)

  if (mode === 'allowed') {
    const added = placement.foreignAddedSinceStart || []
    return (
      <section className="fabric-routing" data-testid="fabric-routing" data-mode="allowed">
        <div className="fabric-routing__head">
          <h2>Routing</h2>
          <Chip tone="warn" dot data-testid="fabric-routing-live">Also placing on other engines</Chip>
        </div>
        <div className="fabric-routing__accepting" data-testid="fabric-routing-accepting">
          <h3>What this proxy is accepting now</h3>
          {groups.length === 0
            ? <p className="fabric-note">No node here has anything to accept.</p>
            : <Acceptance groups={groups} limits={limits} />}
        </div>
        {added.length > 0 && (
          <div className="fabric-routing__added" data-testid="fabric-routing-added-later">
            <h3>Added to the nodes file since this proxy started</h3>
            <ul>
              {added.map((node) => (
                <li key={node.label} data-node-label={node.label}>
                  <span className="fabric-routing__node-label">{node.label}</span>
                  {node.engine ? ` (${node.engine})` : ''}
                </li>
              ))}
            </ul>
          </div>
        )}
        <p className="fabric-note">To place on Camelid engines only again:</p>
        <CommandFor placement={placement} target="camelid_only" />
      </section>
    )
  }

  const choose = (next) => {
    setChoice(next)
    setConfirmed(false)
    setConfirming(next === 'mixed')
  }
  const cancel = () => {
    setConfirming(false)
    setConfirmed(false)
    setChoice('camelid_only')
    window.requestAnimationFrame(() => mixedRadioRef.current?.focus())
  }

  return (
    <section className="fabric-routing" data-testid="fabric-routing" data-mode="refused">
      <div className="fabric-routing__head">
        <h2>Routing</h2>
        <Chip tone="ready" dot data-testid="fabric-routing-live">Camelid engines only</Chip>
      </div>
      <div className="fabric-routing__choice" role="radiogroup" aria-label="Where this proxy places work" data-testid="fabric-routing-choice">
        <label className="fabric-routing__option">
          <input
            type="radio"
            name="fabric-routing"
            value="camelid_only"
            checked={choice === 'camelid_only'}
            onChange={() => choose('camelid_only')}
          />
          Camelid engines only
        </label>
        <label className="fabric-routing__option">
          <input
            ref={mixedRadioRef}
            type="radio"
            name="fabric-routing"
            value="mixed"
            checked={choice === 'mixed'}
            onChange={() => choose('mixed')}
          />
          Also place on other engines
        </label>
      </div>
      {confirming && (
        <Confirmation
          placement={placement}
          groups={groups}
          limits={limits}
          onCancel={cancel}
          onConfirm={() => { setConfirming(false); setConfirmed(true) }}
        />
      )}
      {choice === 'mixed' && confirmed && <CommandFor placement={placement} target="mixed" />}
    </section>
  )
}

export default RoutingMode
