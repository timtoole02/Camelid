import { useState } from 'react'
import { Button } from '../ui/Button'
import { Field } from '../ui/Field'

/* The second of the two clicks it takes to add a machine.
 *
 * Everything that will be written is on screen before anything is: both lines
 * exactly as they will appear, the file they go in, and the consequences the
 * *server* stated. Nothing here is derived from the engine name — whether a
 * node would be shown the fabric's bearer depends on how the proxy was
 * started, and only the proxy knows that.
 *
 * The host field holds the address, not a name the device chose for itself. A
 * router-supplied name is offered beside it as a separate, explicit choice,
 * with what that name actually proves. */

const WARNINGS = {
  cleartext: 'Prompts and answers to this machine will cross the network unencrypted.',
  bearer_will_be_sent: "This fabric's API key will be sent to this machine on every health check.",
  bearer_sent_if_configured: "If this fabric is started with an API key, that key will be sent to this machine.",
  name_resolves_to_several_addresses: 'That name resolves to more than one address; which one answers may change.',
  name_not_written: 'The name this machine was reached by is not one this build will write into a nodes file, so the address is proposed instead.',
}

export function DiscoveryConfirm({ finding, nodesFile, onWrite, onCancel, busy, problem }) {
  const proposal = finding.proposal
  const [label, setLabel] = useState(proposal.label)
  const [host, setHost] = useState(proposal.host)
  // Never defaulted for an ambiguous row: the first match is not a pick.
  const [engine, setEngine] = useState(
    proposal.engineChoices.length > 0 ? '' : proposal.engine,
  )

  const line = `${label}=${engine || '<engine>'}://${host}:${proposal.port}`
  const ready = Boolean(label && host && engine) && !busy

  return (
    <div className="discovery-confirm" data-testid="discovery-confirm" data-engine={engine || 'unchosen'}>
      <div className="discovery-confirm__fields">
        <Field label="Label">
          <input
            className="cx-input"
            value={label}
            spellCheck="false"
            autoComplete="off"
            onChange={(event) => setLabel(event.target.value)}
            data-testid="discovery-confirm-label"
          />
        </Field>
        <Field label="Host">
          <input
            className="cx-input"
            value={host}
            spellCheck="false"
            autoComplete="off"
            onChange={(event) => setHost(event.target.value)}
            data-testid="discovery-confirm-host"
          />
        </Field>
      </div>

      {proposal.hostAlternatives.length > 0 && (
        <div className="discovery-confirm__alternatives" data-testid="discovery-confirm-alternatives">
          {proposal.hostAlternatives.map((alternative) => (
            <div key={alternative.host} className="discovery-confirm__alternative">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setHost(alternative.host)
                  if (alternative.label) setLabel(alternative.label)
                }}
                data-testid="discovery-confirm-use-name"
              >
                Use the name {alternative.host}
              </Button>
              <p className="fabric-note">{alternative.warning}</p>
            </div>
          ))}
        </div>
      )}

      {proposal.engineChoices.length > 0 && (
        <fieldset className="discovery-confirm__engines" data-testid="discovery-confirm-engines">
          <legend>This machine answered like more than one engine. Which is it?</legend>
          {proposal.engineChoices.map((choice) => (
            <label key={choice}>
              <input
                type="radio"
                name={`engine-${finding.id}`}
                value={choice}
                checked={engine === choice}
                onChange={() => setEngine(choice)}
              />
              {choice}
            </label>
          ))}
        </fieldset>
      )}

      <div className="discovery-confirm__lines">
        <p className="fabric-note">
          These two lines will be added to <code>{nodesFile || 'the nodes file'}</code>:
        </p>
        <pre data-testid="discovery-confirm-lines">
          <code>{proposal.commentPreview}</code>
          {'\n'}
          <code>{line}</code>
        </pre>
        <p className="fabric-note">The time in the comment is filled in when it is written.</p>
      </div>

      {proposal.warnings.length > 0 && (
        <ul className="discovery-confirm__warnings" data-testid="discovery-confirm-warnings">
          {proposal.warnings.map((warning) => (
            <li key={warning} data-warning={warning}>
              {WARNINGS[warning] || warning}
            </li>
          ))}
        </ul>
      )}

      {problem && (
        <p className="fabric-note" data-testid="discovery-confirm-problem" data-code={problem.code}>
          {problem.message}
          {problem.code === 'name_reaches_another_address' && (
            <Button
              variant="tonal"
              size="sm"
              onClick={() => onWrite({ label, host: finding.address, engine })}
              data-testid="discovery-confirm-use-address"
            >
              Write the address instead
            </Button>
          )}
        </p>
      )}

      <div className="discovery-confirm__actions">
        <Button
          variant="primary"
          disabled={!ready}
          onClick={() => onWrite({ label, host, engine })}
          data-testid="discovery-confirm-write"
        >
          {busy ? 'Writing…' : 'Write to nodes file'}
        </Button>
        <Button variant="ghost" onClick={onCancel} data-testid="discovery-confirm-cancel">
          Cancel
        </Button>
      </div>
    </div>
  )
}

export default DiscoveryConfirm
