import { useEffect, useMemo, useState } from 'react'
import {
  exportTelemetryJson,
  getTelemetrySnapshot,
  perModelBreakdown,
  subscribeTelemetry,
  summarizeTelemetry,
} from '../lib/telemetryLog'
import { Button } from '../components/ui/Button'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { EmptyState } from '../components/ui/EmptyState'
import { IconChart } from '../components/ui/icons'

/* Session telemetry dashboard (Phase 6).

   Every number on this screen is computed from real requests made in this
   browser session — chat sends, workbench try-its, health polls. There is no
   seed data and nothing persists across reloads. The page-level chip states
   that caveat once (I4 framing: operational telemetry, never compatibility or
   support evidence) — it is deliberately not repeated per panel. Bounded
   perf/RSS evidence from the contract lives in the Compatibility ledger, not
   here. */

const fmtMs = (value) => (Number.isFinite(value) ? (value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`) : '—')
const fmtRate = (value) => (Number.isFinite(value) ? `${value >= 10 ? Math.round(value) : value.toFixed(1)} tok/s` : '—')
const fmtTime = (at) => new Date(at).toLocaleTimeString()

function Sparkline({ values, width = 220, height = 36, ariaLabel }) {
  const points = values.filter((v) => Number.isFinite(v))
  if (points.length < 2) return <span className="tele-spark tele-spark--empty">Needs at least 2 requests</span>
  const min = Math.min(...points)
  const max = Math.max(...points)
  const span = max - min || 1
  const step = width / (points.length - 1)
  const path = points.map((v, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${(height - 4 - ((v - min) / span) * (height - 8)).toFixed(1)}`).join(' ')
  return (
    <svg className="tele-spark" width={width} height={height} role="img" aria-label={ariaLabel}>
      <path d={path} fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

export default function TelemetryView() {
  const [snapshot, setSnapshot] = useState(getTelemetrySnapshot)
  const [revealPrompts, setRevealPrompts] = useState(false)

  useEffect(() => subscribeTelemetry(() => setSnapshot(getTelemetrySnapshot())), [])

  const { requests, health } = snapshot
  const summary = useMemo(() => summarizeTelemetry(requests), [requests])
  const models = useMemo(() => perModelBreakdown(requests), [requests])
  const generationRequests = useMemo(() => requests.filter((r) => r.kind === 'chat'), [requests])
  const recentRequests = useMemo(() => requests.slice(-60).reverse(), [requests])
  const healthOk = health.filter((h) => h.ok).length

  const downloadExport = () => {
    const blob = new Blob([exportTelemetryJson()], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'camelid-session-telemetry.json'
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="telemetry-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconChart size={14} /> Telemetry</p>
          <h1>Telemetry</h1>
          <p className="cxv-sub">Computed from the requests this browser made this session — chat sends, workbench try-its, health polls.</p>
        </div>
        <div className="cxv-head__actions">
          <EvidenceChip
            state="neutral"
            label="Live session stats — measured in this browser, nothing persisted"
            source={{ note: 'Every number here comes from requests this browser made this session. Nothing is seeded, stored, or shared, and it all resets on reload.' }}
            size="sm"
          />
        </div>
      </header>

      {requests.length === 0 ? (
        <EmptyState
          className="cx-empty--inline"
          icon={<IconChart size={22} />}
          title="No session traffic yet"
          description="Send a chat or run a workbench try-it and this dashboard fills in from those real requests."
        />
      ) : (
        <>
          <div className="cxv-stat-grid cxv-stat-grid--five">
            <div className="cxv-stat"><span>Requests</span><strong>{summary.total}</strong><small>{generationRequests.length} chat · {summary.total - generationRequests.length} workbench</small></div>
            <div className="cxv-stat"><span>Errors</span><strong>{summary.errors}</strong><small>{summary.errorRate !== null ? `${Math.round(summary.errorRate * 100)}% of session requests` : '—'}</small></div>
            <div className="cxv-stat"><span>TTFT median</span><strong>{fmtMs(summary.medianTtftMs)}</strong><small>client-measured</small></div>
            <div className="cxv-stat"><span>Decode median</span><strong>{fmtRate(summary.medianTokensPerSec)}</strong><small>client-measured</small></div>
            <div className="cxv-stat"><span>Duration median</span><strong>{fmtMs(summary.medianDurationMs)}</strong><small>client-measured</small></div>
          </div>

          <div className="cxv-grid cxv-grid--two">
            <section className="cxv-card cxv-panel">
              <div className="cxv-section__head"><h2>Session trend</h2><span className="cxv-section__count">latest {Math.min(requests.length, 500)} requests</span></div>
              <div className="tele-spark-row"><span className="tele-spark-label">web</span><Sparkline values={requests.map((r) => r.webResearchMs)} ariaLabel="Web research latency trend" /></div>
              <div className="tele-spark-row"><span className="tele-spark-label">TTFT</span><Sparkline values={requests.map((r) => r.ttftMs)} ariaLabel="TTFT trend" /></div>
              <div className="tele-spark-row"><span className="tele-spark-label">tok/s</span><Sparkline values={requests.map((r) => r.tokensPerSec)} ariaLabel="Tokens per second trend" /></div>
              <div className="tele-spark-row"><span className="tele-spark-label">duration</span><Sparkline values={requests.map((r) => r.durationMs)} ariaLabel="Request duration trend" /></div>
            </section>

            <section className="cxv-card cxv-panel">
              <div className="cxv-section__head"><h2>Per-model</h2><span className="cxv-section__count">{models.length} model id{models.length === 1 ? '' : 's'}</span></div>
              {models.length ? (
                <table className="cxv-table">
                  <thead><tr><th>model id</th><th>req</th><th>err</th><th>TTFT med</th><th>tok/s med</th></tr></thead>
                  <tbody>
                    {models.map((row) => (
                      <tr key={row.modelId}>
                        <td><code>{row.modelId}</code></td>
                        <td>{row.total}</td>
                        <td>{row.errors}</td>
                        <td>{fmtMs(row.medianTtftMs)}</td>
                        <td>{fmtRate(row.medianTokensPerSec)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="tele-note">No model-tagged requests yet.</p>
              )}
            </section>
          </div>
        </>
      )}

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head">
          <h2>Backend reachability</h2>
          <span className="cxv-section__count">{health.length ? `${healthOk}/${health.length} polls ok this session` : 'no polls yet'}</span>
        </div>
        <p className="cxv-sub">Live /v1/health polls — 2.5s cadence while the backend answers, backing off to 20s on consecutive failures.</p>
        <div className="tele-health" role="img" aria-label="Backend reachability history">
          {health.slice(-120).map((poll, index) => (
            <span
              key={`${poll.at}-${index}`}
              className={`tele-health__cell ${poll.ok ? 'is-ok' : 'is-fail'}`}
              title={`${fmtTime(poll.at)} · ${poll.ok ? `ok ${fmtMs(poll.latencyMs)}` : 'unreachable'}`}
            />
          ))}
        </div>
      </section>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head">
          <h2>Request log</h2>
          <div className="tele-log-actions">
            <label className="tele-reveal">
              <input type="checkbox" checked={revealPrompts} onChange={(event) => setRevealPrompts(event.target.checked)} />
              Reveal prompt content (this session only)
            </label>
            <Button variant="outline" size="sm" onClick={downloadExport} disabled={!requests.length}>
              Export JSON
            </Button>
          </div>
        </div>
        <p className="cxv-sub">Local-only. Prompts stay redacted unless revealed for this session; exports exclude prompt content and file paths by construction.</p>
        {recentRequests.length ? (
          <table className="cxv-table cxv-table--log">
            <thead><tr><th>time</th><th>endpoint</th><th>model</th><th>outcome</th><th>web</th><th>duration</th><th>tokens</th><th>prompt</th></tr></thead>
            <tbody>
              {recentRequests.map((record) => (
                <tr key={record.id} className={record.outcome !== 'ok' ? 'is-error' : ''}>
                  <td>{fmtTime(record.at)}</td>
                  <td><code>{record.endpoint}</code></td>
                  <td><code>{record.modelId || '—'}</code></td>
                  <td>{record.outcome}{record.httpStatus ? ` · ${record.httpStatus}` : ''}</td>
                  <td>{fmtMs(record.webResearchMs)}</td>
                  <td>{fmtMs(record.durationMs)}</td>
                  <td>{Number.isFinite(record.promptTokens) ? `${record.promptTokens}→${record.completionTokens ?? '—'}` : '—'}</td>
                  <td className="tele-prompt">{record.promptText ? (revealPrompts ? record.promptText.slice(0, 80) : '•••• redacted') : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="tele-note">Nothing logged yet this session.</p>
        )}
      </section>
    </section>
  )
}
