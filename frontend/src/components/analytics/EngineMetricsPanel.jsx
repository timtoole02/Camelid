import { useEffect, useState } from 'react'
import {
  LANE_RISK,
  LANE_RISK_COPY,
  formatBytes,
  formatCount,
  formatRate,
  formatSeconds,
  formatShare,
  readActiveLane,
  summarizeMetrics,
} from '../../lib/runtimeMetrics'

/* Engine metrics — what the SERVER reports about itself.

   Distinct from the Telemetry view, which times requests in this browser and is
   therefore client-measured. These numbers come from the engine's own /metrics
   surface, and they include quantities a browser cannot observe at all: how often
   the prompt-prefix cache was reused, how forward time split between evaluating
   prompts and decoding, how deep the job queue got.

   COPY RULE. Every counter here is cumulative since the engine started, so every
   derived figure is a LIFETIME AVERAGE and is labelled as one. A server that ran
   fast for an hour and has been slow for a minute still reports the hour, and a
   panel that called that "current" would be lying with true numbers.

   The second section is the quarantined half: which lane the engine is decoding
   on, and whether that lane carries parity evidence here. It offers no controls,
   because the engine has none — one HTTP route mutates runtime configuration and
   the rest are read once at process start. A toggle would save a value and do
   nothing. */

const REFRESH_MS = 5000

function Stat({ label, value, hint }) {
  return (
    <div className="engmetrics__stat">
      <dt>{label}</dt>
      <dd title={hint}>{value}</dd>
    </div>
  )
}

export function EngineMetricsPanel({ apiBase, runtime, initialMetricsText = null }) {
  const [summary, setSummary] = useState(() => (initialMetricsText ? summarizeMetrics(initialMetricsText) : null))
  const [error, setError] = useState(null)
  const base = (apiBase || '').replace(/\/$/, '')

  useEffect(() => {
    if (initialMetricsText) return undefined
    let cancelled = false
    const load = async () => {
      try {
        const response = await fetch(`${base}/metrics`)
        if (!response.ok) {
          if (!cancelled) setError(`The engine returned ${response.status} for /metrics.`)
          return
        }
        const text = await response.text()
        if (cancelled) return
        const next = summarizeMetrics(text)
        setSummary(next)
        /* A 200 that is not Prometheus text is a different fault from an engine
           with nothing to report — most often a proxy answering /metrics itself.
           Saying which one keeps a routing problem from reading as an idle engine. */
        setError(next
          ? null
          : /^\s*#\s*(HELP|TYPE)/m.test(text)
            ? 'The engine reported no metrics.'
            : 'The response to /metrics was not Prometheus text — something between this page and the engine answered it.')
      } catch {
        if (!cancelled) setError('The engine is not reachable.')
      }
    }
    load()
    const timer = setInterval(load, REFRESH_MS)
    return () => { cancelled = true; clearInterval(timer) }
  }, [base, initialMetricsText])

  const lane = readActiveLane(runtime)
  const laneCopy = LANE_RISK_COPY[lane.risk]

  return (
    <div className="engmetrics">
      <section className="engmetrics__section">
        <h2>Engine metrics</h2>
        <p className="engmetrics__intro">
          Reported by the engine about itself, not measured in this browser. Counters are
          cumulative since the engine started, so every average below covers its whole run.
        </p>

        {error && <p className="engmetrics__error">{error}</p>}

        {summary && (
          <>
            <dl className="engmetrics__stats">
              <Stat
                label="Prefix cache reuse"
                value={formatShare(summary.lifetime.promptCacheHitRate)}
                hint="Share of completed generations that reused a cached prompt prefix instead of re-evaluating it. Lifetime average."
              />
              <Stat
                label="Time spent on prompts"
                value={formatShare(summary.lifetime.prefillShare)}
                hint="Share of forward time spent evaluating prompts rather than decoding. Lifetime average."
              />
              <Stat
                label="Decode rate"
                value={formatRate(summary.lifetime.decodeTokensPerSecond)}
                hint="Decoded tokens divided by decode time, over the engine's whole run. Not a current rate."
              />
              <Stat
                label="Weights reused"
                value={formatShare(summary.lifetime.weightCacheHitRate)}
                hint="Share of generations that reused already-loaded weights. Lifetime average."
              />
            </dl>

            <dl className="engmetrics__stats engmetrics__stats--muted">
              <Stat label="Generations" value={formatCount(summary.counters.genRequests)} />
              <Stat label="Failed" value={formatCount(summary.counters.genFailures)} />
              <Stat label="Prompt tokens" value={formatCount(summary.counters.promptTokens)} />
              <Stat label="Decoded tokens" value={formatCount(summary.counters.decodeTokens)} />
              <Stat label="Prompt time" value={formatSeconds(summary.lifetime.promptSeconds)} />
              <Stat label="Decode time" value={formatSeconds(summary.lifetime.decodeSeconds)} />
            </dl>

            <dl className="engmetrics__stats engmetrics__stats--muted">
              <Stat label="Queue depth" value={formatCount(summary.live.queueDepth)} hint="Queued plus active engine jobs, right now." />
              <Stat label="Waiting" value={formatCount(summary.live.queuedTasks)} />
              <Stat label="Resident memory" value={formatBytes(summary.memory.residentBytes)} />
              <Stat
                label="VRAM in use"
                value={summary.memory.vramUsedBytes === null
                  ? '—'
                  : `${formatBytes(summary.memory.vramUsedBytes)} of ${formatBytes(summary.memory.vramTotalBytes)}`}
                hint="From the engine's cached hardware probe."
              />
            </dl>

            {summary.lifetime.promptCacheAttempts === 0 && (
              <p className="engmetrics__note">
                The prefix cache has not been consulted yet, so its reuse rate is unknown rather than zero.
              </p>
            )}
            {summary.missing.length > 0 && (
              <p className="engmetrics__note">
                {summary.missing.length} metric{summary.missing.length === 1 ? '' : 's'} this page knows about
                {summary.missing.length === 1 ? ' is' : ' are'} not reported by this engine build.
              </p>
            )}
          </>
        )}
      </section>

      <section className="engmetrics__section">
        <h2>Active configuration</h2>
        <div className={`engmetrics__lane engmetrics__lane--${lane.risk}`}>
          <span className="engmetrics__lane-label">{laneCopy.label}</span>
          <p className="engmetrics__lane-detail">{laneCopy.detail}</p>
        </div>

        {lane.rows.length > 0 && (
          <table className="cxv-table engmetrics__table">
            <thead>
              <tr>
                <th scope="col">Setting</th>
                <th scope="col">Active value</th>
                <th scope="col">Changing it</th>
              </tr>
            </thead>
            <tbody>
              {lane.rows.map((row) => (
                <tr key={row.key}>
                  <td>
                    {row.label}
                    <span className="engmetrics__row-note">{row.note}</span>
                  </td>
                  <td className="engmetrics__value">{row.value}</td>
                  <td className="engmetrics__change">
                    {row.changeable === 'gpu_toggle'
                      ? 'Settings → GPU acceleration'
                      : 'Restart required'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* The reason this section has no switches. Stated once, plainly, rather
            than implied by their absence. */}
        <p className="engmetrics__note">
          These are read once when the engine starts, and several are latched for the life of the
          process — so they are shown here rather than offered as controls. Changing one means
          restarting the engine with a different setting. The GPU acceleration toggle in Settings is
          the one exception the engine accepts while running.
        </p>

        {lane.reasons.length > 0 && (
          <details className="engmetrics__reasons">
            <summary>Why the engine chose this lane</summary>
            <ul>
              {lane.reasons.map((reason) => <li key={reason}>{reason}</li>)}
            </ul>
          </details>
        )}
      </section>
    </div>
  )
}

export default EngineMetricsPanel
