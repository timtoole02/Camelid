import { useMemo, useState } from 'react'
import { useFabric } from '../hooks/useFabric'
import { FabricNodeTable } from '../components/fabric/FabricNodeTable'
import { FabricNodeDrawer } from '../components/fabric/FabricNodeDrawer'
import { Unknown } from '../components/fabric/Unknown'
import { Button } from '../components/ui/Button'
import { Chip } from '../components/ui/Chip'
import { EmptyState } from '../components/ui/EmptyState'
import { Field } from '../components/ui/Field'
import { IconCopy, IconNetwork, IconRefresh, IconServer } from '../components/ui/icons'
import { copyText } from '../lib/markdown.jsx'
import { DETAIL_WITHHELD_REASON, fabricPosture, fabricProblemMessage } from '../lib/fabricModel.js'
import { endpointLabel, normalizeEndpoint } from '../lib/fabricClient.js'

/* The Cluster view reports one fabric proxy, live.

   It used to be a diagram: nodes were drawn by hand, saved to browser storage,
   and a green "live" chip meant a string in that storage said `running`. Nothing
   on this page is drawn any more. Every value comes from the answer to a request
   made moments ago, and anything the proxy did not tell us says so. */

const POSTURE = {
  ready: { tone: 'ready', label: 'serving' },
  degraded: { tone: 'warn', label: 'degraded' },
  not_ready: { tone: 'error', label: 'no node ready' },
  unknown: { tone: 'neutral', label: 'unknown' },
}

const SERVE_COMMAND = 'camelid fabric serve --node LABEL=HOST:PORT'

function CountCell({ label, value, tone }) {
  return (
    <div className={`fabric-count fabric-count--${tone}`}>
      <span className="fabric-count__value">
        {value === null || value === undefined
          ? <Unknown why="The proxy did not report this count." />
          : value}
      </span>
      <span className="fabric-count__label">{label}</span>
    </div>
  )
}

function CopyableCommand({ command }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="fabric-cmd">
      <code>{command}</code>
      <Button
        variant="ghost"
        size="sm"
        icon={<IconCopy size={14} />}
        onClick={async () => {
          const ok = await copyText(command)
          setCopied(ok)
          window.setTimeout(() => setCopied(false), 2000)
        }}
      >
        {copied ? 'Copied' : 'Copy'}
      </Button>
    </div>
  )
}

export default function ClusterView() {
  const { endpoint, setEndpoint, fabric, phase, checkedAt, refresh, valid } = useFabric()
  const [draft, setDraft] = useState(endpoint)
  const [selectedLabel, setSelectedLabel] = useState(null)

  const shown = endpointLabel(normalizeEndpoint(endpoint)) || endpoint
  const posture = POSTURE[fabricPosture(fabric)]
  const problem = fabric?.problem ? fabricProblemMessage(fabric.problem, shown) : null
  const nodes = fabric?.nodes ?? null
  const selected = useMemo(
    () => (nodes && selectedLabel ? nodes.find((node) => node.label === selectedLabel) || null : null),
    [nodes, selectedLabel],
  )

  const applyEndpoint = (event) => {
    event.preventDefault()
    const normalized = setEndpoint(draft)
    setSelectedLabel(null)
    if (normalized) refresh(draft.trim())
  }

  return (
    <div className="fabric-view">
      <header className="fabric-header">
        <div className="fabric-header__copy">
          <p className="cxv-kicker"><IconNetwork size={14} /> Cluster</p>
          <h1>Cluster</h1>
          <p className="cxv-sub">
            The machines one Camelid fabric proxy routes across, as that proxy reports them right now.
          </p>
        </div>

        <form className="fabric-header__endpoint" onSubmit={applyEndpoint}>
          <Field label="Fabric proxy address">
            <input
              className="cx-input"
              value={draft}
              spellCheck="false"
              autoComplete="off"
              onChange={(event) => setDraft(event.target.value)}
              placeholder="127.0.0.1:8282"
              aria-invalid={normalizeEndpoint(draft) === null}
              data-testid="fabric-endpoint-input"
            />
          </Field>
          <div className="fabric-header__endpoint-actions">
            <Button type="submit" variant="tonal" disabled={normalizeEndpoint(draft) === null}>Use</Button>
            <Button
              variant="ghost"
              icon={<IconRefresh size={15} />}
              onClick={() => refresh()}
              disabled={!valid}
              data-testid="fabric-refresh"
            >
              Refresh
            </Button>
          </div>
        </form>
      </header>

      <section className="fabric-status" data-testid="fabric-status" data-phase={phase}>
        <div className="fabric-status__headline">
          <Chip tone={posture.tone} dot data-testid="fabric-posture">{posture.label}</Chip>
          <span className="fabric-status__endpoint">{shown}</span>
          <span className="fabric-status__asof">
            {phase === 'never' || phase === 'first'
              ? 'reading…'
              : (checkedAt ? `as of ${new Date(checkedAt).toLocaleTimeString()}` : null)}
          </span>
        </div>
        {fabric?.build && <p className="fabric-status__build">Proxy build <code>{fabric.build}</code></p>}
      </section>

      {problem && (
        <div className="fabric-panel fabric-panel--problem" data-testid="fabric-problem" role="status">
          <p>{problem}</p>
          {fabric.problem.code === 'unreachable' && (
            <>
              <p className="fabric-note">Start one on this machine, then Refresh:</p>
              <CopyableCommand command={SERVE_COMMAND} />
            </>
          )}
        </div>
      )}

      {fabric?.detail === 'withheld' && (
        <div className="fabric-panel fabric-panel--withheld" data-testid="fabric-withheld" role="status">
          <p>{DETAIL_WITHHELD_REASON}</p>
          <p className="fabric-note">
            What it will still say: it is{' '}
            {fabric.ready === null
              ? <Unknown why="The proxy did not report readiness." />
              : (fabric.ready ? 'serving at least one ready node' : 'serving no ready node')}.
          </p>
        </div>
      )}

      {fabric?.detail === 'disclosed' && (
        <>
          <section className="fabric-counts" data-testid="fabric-counts">
            <CountCell label="nodes" value={fabric.counts?.total ?? null} tone="total" />
            <CountCell label="ready" value={fabric.counts?.ready ?? null} tone="ready" />
            <CountCell label="not ready" value={fabric.counts?.notReady ?? null} tone="warn" />
            <CountCell label="unreachable" value={fabric.counts?.unreachable ?? null} tone="error" />
          </section>

          {nodes && nodes.length > 0 ? (
            <FabricNodeTable nodes={nodes} selectedLabel={selectedLabel} onSelect={setSelectedLabel} />
          ) : (
            <EmptyState
              icon={<IconServer size={22} />}
              title="This proxy has no nodes"
              description="It answered, and it is configured with an empty fabric. Give it a node and it picks the change up without a restart."
              action={<CopyableCommand command={SERVE_COMMAND} />}
            />
          )}

          <section className="fabric-models" data-testid="fabric-models">
            <h2>Models being served</h2>
            {fabric.placements && fabric.placements.length > 0 ? (
              <ul className="fabric-models__list">
                {fabric.placements.map(({ model, labels }) => (
                  <li key={model}>
                    <span className="fabric-models__id">{model}</span>
                    <span className="fabric-models__on">{labels.join(', ')}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="fabric-note">No node is currently serving a model.</p>
            )}
          </section>
        </>
      )}

      {selected && (
        <FabricNodeDrawer node={selected} checkedAt={checkedAt} onClose={() => setSelectedLabel(null)} />
      )}
    </div>
  )
}
