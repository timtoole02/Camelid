import { useEffect, useMemo, useState } from 'react'
import { Button } from '../ui/Button'
import { Chip } from '../ui/Chip'
import { Field } from '../ui/Field'
import { CopyableCommand } from './CopyableCommand'
import { CorsHint } from './CorsHint'
import { Unknown } from './Unknown'
import { DiscoveryConfirm } from './DiscoveryConfirm.jsx'
import { useDiscovery } from '../../hooks/useDiscovery.js'
import { groupFindings, problemMessage } from '../../lib/discoveryModel.js'

/* Screen B: finding machines to add.
 *
 * This panel renders route JSON and nothing else. It holds no discovery logic
 * of its own, because a page that re-derived a rule would drift from the proxy
 * enforcing it — which is exactly how a supported model once got shown as
 * unsupported in this same view.
 *
 * Three behaviours are load-bearing and are each pinned by a smoke:
 *   - it never scans on mount and never polls, because a scan sends traffic to
 *     machines nobody named;
 *   - "Add to fabric" exists only on a row the proxy actually identified;
 *   - a written node is drawn only once the proxy itself reports it. */

const NOTHING_LISTENING = `Both \`ollama serve\` and \`camelid serve\` listen on loopback by default,
so they cannot be seen from another machine unless they were started on an address the network can reach.`

function Count({ label, value }) {
  return (
    <span className="discovery-count">
      <strong>{value === null || value === undefined ? <Unknown why="The proxy did not report this count." /> : value}</strong>
      {' '}{label}
    </span>
  )
}

function Evidence({ finding }) {
  return (
    <details className="discovery-evidence">
      <summary>What it answered</summary>
      <ul>
        {finding.evidence.map((entry) => (
          <li key={entry.request}>
            <code>{entry.request}</code>
            {' → '}
            {entry.status === null ? <Unknown why="No HTTP answer." /> : entry.status}
            {' — '}
            {entry.fact}
          </li>
        ))}
      </ul>
    </details>
  )
}

function Row({ finding, nodesFile, canJoin, joinState, onAdd, onWrite, onCancel, confirming }) {
  const where = finding.addresses.length > 1
    ? `${finding.address} (also ${finding.addresses.slice(1).join(', ')})`
    : finding.address

  return (
    <li className="discovery-row" data-testid="discovery-row" data-kind={finding.kind} data-address={finding.address}>
      <div className="discovery-row__head">
        <span className="discovery-row__where">{where}:{finding.port}</span>
        {finding.engine && (
          <Chip tone="ready">
            {finding.engine}
            {finding.version ? ` ${finding.version}` : ''}
          </Chip>
        )}
        {finding.kind === 'answers_like' && !finding.version && (
          <Unknown why="This engine reported no version this build will repeat." />
        )}
      </div>

      {finding.summary && <p className="fabric-note">{finding.summary}</p>}
      {finding.notProposed && (
        <p className="fabric-note" data-testid="discovery-not-proposed">{finding.notProposed}</p>
      )}
      {finding.name.name
        ? <p className="fabric-note">Known here as <code>{finding.name.name}</code>, which resolves back to this address.</p>
        : finding.name.why && <p className="fabric-note">No usable name: {finding.name.why}.</p>}
      {finding.possiblySameAs.length > 0 && (
        <p className="fabric-note">
          Answers identically to another row; this build cannot tell whether they are one machine.
        </p>
      )}

      <Evidence finding={finding} />

      {joinState?.phase === 'waiting' && (
        <div data-testid="discovery-joined-waiting">
          <pre><code>{joinState.joined.appended}</code></pre>
          <p className="fabric-note">Written — waiting for the proxy to report it.</p>
        </div>
      )}
      {joinState?.phase === 'reported' && (
        <p className="fabric-note" data-testid="discovery-joined-reported">
          Added, and the proxy is reporting it.
        </p>
      )}
      {joinState?.phase === 'unreported' && (
        <p className="fabric-note" data-testid="discovery-joined-unreported">
          Written, but the proxy has not picked it up. Check its output.
        </p>
      )}

      {confirming ? (
        <DiscoveryConfirm
          finding={finding}
          nodesFile={nodesFile}
          busy={joinState?.phase === 'writing'}
          problem={joinState?.problem
            ? { code: joinState.problem.code, message: problemMessage(joinState.problem) }
            : null}
          onWrite={onWrite}
          onCancel={onCancel}
        />
      ) : (
        finding.canJoin && canJoin && !joinState && (
          <Button variant="tonal" size="sm" onClick={onAdd} data-testid="discovery-add">
            Add to fabric…
          </Button>
        )
      )}
    </li>
  )
}

export function DiscoveryPanel({ base, labels = [], pageOrigin }) {
  const { policy, policyProblem, loadPolicy, scan, start, cancel, joins, confirm, settle } = useDiscovery(base)
  const [clientKey, setClientKey] = useState('')
  const [range, setRange] = useState('')
  const [hosts, setHosts] = useState('')
  const [extraPorts, setExtraPorts] = useState('')
  const [includeThisMachine, setIncludeThisMachine] = useState(true)
  const [includeDefaultPorts, setIncludeDefaultPorts] = useState(true)
  const [confirming, setConfirming] = useState(null)

  // Read-only, and the only request made without a click: it is what says
  // whether there is anything on offer here at all.
  useEffect(() => { loadPolicy() }, [loadPolicy])

  // Pre-filled, never scanned until somebody presses Scan.
  useEffect(() => {
    if (policy?.suggestions?.length && !range) setRange(policy.suggestions[0].cidr)
  }, [policy, range])

  // A written row settles against the labels the Cluster view is already
  // polling, rather than against anything this panel assumes. Keyed on the
  // labels themselves rather than the array: the parent rebuilds that list on
  // every render, and counting those as polls would call a fresh write
  // unreported within a second of making it.
  const labelKey = labels.join(',')
  useEffect(() => { settle(labelKey ? labelKey.split(',') : []) }, [labelKey, settle])

  // ...and again on a timer while anything is still waiting. Keyed on the
  // labels alone, the effect above never re-runs in the one case the
  // "not picked up" state exists for — the proxy never reporting the label —
  // so the row would sit on "waiting" for ever, implying progress.
  const waiting = Object.values(joins).some((join) => join.phase === 'waiting')
  useEffect(() => {
    if (!waiting) return undefined
    const timer = setInterval(() => settle(labelKey ? labelKey.split(',') : []), 2000)
    return () => clearInterval(timer)
  }, [waiting, labelKey, settle])

  const suggestion = policy?.suggestions?.[0] ?? null
  const groups = useMemo(
    () => (scan.discovery ? groupFindings(scan.discovery.findings) : []),
    [scan.discovery],
  )

  const state = (() => {
    if (policyProblem?.code === 'discovery_disabled') return 'disabled'
    if (policyProblem?.code === 'old_build') return 'old_build'
    if (policyProblem?.code === 'key_required') return 'key_required'
    if (policyProblem?.code === 'key_refused') return 'key_refused'
    if (policyProblem?.code === 'loopback_only') return 'loopback_only'
    if (policyProblem?.cause === 'network') return 'origin_blocked'
    if (policyProblem) return 'failed'
    if (scan.phase === 'scanning') return 'scanning'
    if (scan.phase === 'failed') return 'failed'
    if (scan.phase === 'settled') return 'results'
    return policy ? 'ready' : 'loading'
  })()

  const offBox = range.trim().length > 0 || hosts.trim().length > 0
  const lanBlocked = offBox && policy?.transport?.lanPermitted === false
  const scanCommand = `camelid fabric discover --cidr ${range || '<range>'} --allow-cleartext-node-transport`
  const serveCommand = `camelid fabric serve --nodes-file PATH --discovery --cors-origin ${pageOrigin || '<this page>'}`

  const onScan = () => start(
    {
      ranges: range.trim() ? [range.trim()] : [],
      hosts: hosts.split(',').map((host) => host.trim()).filter(Boolean),
      ports: extraPorts.split(',').map((port) => Number(port.trim())).filter((port) => Number.isFinite(port) && port > 0),
      loopback: includeThisMachine,
      default_ports: includeDefaultPorts,
    },
    { clientKey },
  )

  return (
    <section className="discovery" data-testid="fabric-discovery" data-state={state}>
      <header className="discovery__head">
        <h2>Find machines</h2>
        <p className="cxv-sub">
          Look for inference engines this fabric could use. Nothing is added by looking.
        </p>
      </header>

      {(state === 'disabled' || state === 'old_build') && (
        <div className="discovery__blocked">
          <p>{problemMessage(policyProblem)}</p>
          <CopyableCommand command={serveCommand} />
          <p className="fabric-note">Or look from a terminal on that machine:</p>
          <CopyableCommand command={scanCommand} />
        </div>
      )}

      {state === 'loopback_only' && (
        <p data-testid="discovery-loopback-only">{problemMessage(policyProblem)}</p>
      )}

      {state === 'origin_blocked' && (
        <CorsHint pageOrigin={pageOrigin} diagnosis="possible" />
      )}

      {(state === 'key_required' || state === 'key_refused') && (
        <div className="discovery__key">
          <p>{problemMessage(policyProblem)}</p>
          <Field label="Client key">
            <input
              className="cx-input"
              type="password"
              value={clientKey}
              autoComplete="off"
              onChange={(event) => setClientKey(event.target.value)}
              data-testid="discovery-client-key"
            />
          </Field>
          <Button variant="tonal" onClick={() => loadPolicy({ clientKey })}>Use this key</Button>
        </div>
      )}

      {(state === 'ready' || state === 'scanning' || state === 'results' || state === 'failed') && policy && (
        <>
          <div className="discovery__scope">
            <Field label="Network range">
              <input
                className="cx-input"
                value={range}
                spellCheck="false"
                autoComplete="off"
                placeholder="100.64.0.0/24"
                onChange={(event) => setRange(event.target.value)}
                data-testid="discovery-range"
              />
            </Field>
            {suggestion && (
              <p className="fabric-note" data-testid="discovery-suggestion">
                Suggested from {suggestion.interface ? <code>{suggestion.interface}</code> : 'this machine'}
                {suggestion.prefixSource === 'interface_netmask' && ' using its own netmask'}
                {suggestion.prefixSource === 'narrowed' && ', narrowed to the block around this machine'}
                {suggestion.prefixSource === 'assumed_24' && ', assuming a /24'}.
              </p>
            )}
            <Field label="Named machines (comma separated)">
              <input
                className="cx-input"
                value={hosts}
                spellCheck="false"
                autoComplete="off"
                onChange={(event) => setHosts(event.target.value)}
                data-testid="discovery-hosts"
              />
            </Field>
            <Field label="Extra ports (comma separated)">
              <input
                className="cx-input"
                value={extraPorts}
                spellCheck="false"
                autoComplete="off"
                onChange={(event) => setExtraPorts(event.target.value)}
                data-testid="discovery-ports"
              />
            </Field>
            <label>
              <input
                type="checkbox"
                checked={includeThisMachine}
                onChange={(event) => setIncludeThisMachine(event.target.checked)}
                data-testid="discovery-include-loopback"
              />
              Include this machine
            </label>
            <label>
              <input
                type="checkbox"
                checked={includeDefaultPorts}
                onChange={(event) => setIncludeDefaultPorts(event.target.checked)}
                data-testid="discovery-include-defaults"
              />
              Include each engine&apos;s usual port ({policy.defaultPorts.join(', ')})
            </label>
          </div>

          <p className="fabric-note" data-testid="discovery-transport">
            Node transport: {policy.transport.description}. No credential is presented to any host.
          </p>

          {lanBlocked ? (
            <div data-testid="discovery-transport-blocked">
              <p className="fabric-note">
                Reaching another machine in the clear needs an explicit acknowledgement, so this
                proxy will refuse that range. Restart it with the flag, or look from a terminal:
              </p>
              <CopyableCommand command={scanCommand} />
            </div>
          ) : (
            <div className="discovery__actions">
              <Button
                variant="primary"
                onClick={onScan}
                disabled={state === 'scanning'}
                data-testid="discovery-scan"
              >
                {state === 'scanning' ? 'Looking…' : 'Scan'}
              </Button>
              {state === 'scanning' && (
                <Button variant="ghost" onClick={cancel} data-testid="discovery-cancel">Cancel</Button>
              )}
            </div>
          )}
        </>
      )}

      {state === 'failed' && (
        <p data-testid="discovery-problem" data-code={scan.problem?.code || policyProblem?.code}>
          {problemMessage(scan.problem || policyProblem)}
        </p>
      )}

      {state === 'results' && scan.discovery && (
        <div className="discovery__results" data-testid="discovery-results">
          <p className="discovery__summary" data-testid="discovery-summary">
            <Count label="planned" value={scan.discovery.planned} />
            <Count label="probed" value={scan.discovery.probes} />
            <Count label="refused" value={scan.discovery.notListed.refused} />
            <Count label="timed out" value={scan.discovery.notListed.timedOut} />
            <Count label="unreachable" value={scan.discovery.notListed.unreachable} />
            <Count label="not reached in time" value={scan.discovery.notScanned} />
          </p>
          {scan.discovery.hint && <p className="fabric-note" data-testid="discovery-hint">{scan.discovery.hint}</p>}

          {groups.length === 0 ? (
            <p className="fabric-note" data-testid="discovery-empty">{NOTHING_LISTENING}</p>
          ) : (
            groups.map((group) => (
              <div key={group.key} className="discovery__group" data-group={group.key}>
                <h3>{group.title}</h3>
                <ul>
                  {group.findings.map((finding) => (
                    <Row
                      key={finding.id}
                      finding={finding}
                      nodesFile={scan.discovery.nodesFile?.path}
                      canJoin={scan.discovery.canJoin}
                      joinState={joins[finding.id]}
                      confirming={confirming === finding.id}
                      onAdd={() => setConfirming(finding.id)}
                      onCancel={() => setConfirming(null)}
                      onWrite={({ label, host, engine }) => {
                        setConfirming(null)
                        confirm(
                          finding.id,
                          {
                            label,
                            host,
                            engine,
                            port: finding.port,
                            base_sha256: scan.discovery.nodesFile?.sha256,
                            /* The server's own spelling, sent back untouched.
                               Composed here it gets IPv6 wrong: `::1` and
                               `[::1]` are one machine, and only one of the two
                               is an address anything can read. */
                            scanned_address: finding.proposal?.scannedAddress,
                          },
                          { clientKey },
                        )
                      }}
                    />
                  ))}
                </ul>
              </div>
            ))
          )}
        </div>
      )}
    </section>
  )
}

export default DiscoveryPanel
