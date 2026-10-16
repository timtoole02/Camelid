import { Fragment, useEffect, useMemo, useState } from 'react'
import FlowBench from '../components/observatory/FlowBench'
import NeuralField from '../components/observatory/NeuralField'
import DetailsPanel from '../components/observatory/DetailsPanel'
import { useInferenceTelemetry } from '../hooks/useInferenceTelemetry'
import { getTelemetrySnapshot, rateSourceLabel, subscribeTelemetry, summarizeTelemetry } from '../lib/telemetryLog'
import { getChatGateState } from '../lib/chatGate'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { IconChevronRight, IconObservatory } from '../components/ui/icons'
import { fmtMs, fmtRate } from '../components/observatory/format'
import { appStorage } from '../lib/appStorage.js'

/* Observatory (Phase 6.1 — "The Flow Bench"): inference rendered as liquid.
   The centerpiece canvas and the instrument rail consume the SAME lifecycle
   bus as the Telemetry view, so the art and the numbers cannot disagree. The
   backend-reported run panel (camelid.telemetry/v1 SSE) remains below as a
   separate, explicitly backend-side instrument. */

/* Renderer mode. Neural Field is the default: its Phase 5 gate passed
   2026-07-02 (frames + PERF p95 2.3ms@DPR1 + truthfulness audit + build —
   see design-evidence/neural-field/). A stored choice always wins. */
const RENDERER_KEY = 'camelid.observatory.renderer'
const RENDERERS = ['flowbench', 'neuralfield']

function initialRenderer() {
  try {
    const stored = appStorage.getItem(RENDERER_KEY)
    return RENDERERS.includes(stored) ? stored : 'neuralfield'
  } catch {
    return 'neuralfield'
  }
}

export default function InferenceObservatoryView({ apiBase, runtime = null, selectedModel = null, capabilities = null }) {
  const store = useInferenceTelemetry(apiBase)
  const [snapshot, setSnapshot] = useState(getTelemetrySnapshot)
  const [highlightId, setHighlightId] = useState(null)
  const [detailsCollapsed, setDetailsCollapsed] = useState(true)
  const [systemReduced] = useState(() => typeof window !== 'undefined' && Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches))
  const [manualReduced, setManualReduced] = useState(false)
  const reducedMotion = systemReduced || manualReduced
  const [renderer, setRenderer] = useState(initialRenderer)

  const pickRenderer = (mode) => {
    setRenderer(mode)
    try {
      appStorage.setItem(RENDERER_KEY, mode)
    } catch { /* persistence is best-effort */ }
  }

  useEffect(() => subscribeTelemetry(() => setSnapshot(getTelemetrySnapshot())), [])

  const { requests } = snapshot
  const summary = useMemo(() => summarizeTelemetry(requests), [requests])
  const recent = useMemo(() => requests.slice(-9).reverse(), [requests])
  const gate = getChatGateState(capabilities, selectedModel, runtime)
  const activeModelId = runtime?.active_model_id || null

  return (
    <section className="observatory-view cxv flowbench-view">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconObservatory size={14} /> Observatory</p>
          <h1>Observatory</h1>
          <p className="cxv-sub">A live picture of inference on this machine, drawn from the real requests made in this session. When nothing is running, the view stays still.</p>
        </div>
        <div className="cxv-head__actions">
          <div className="observatory-renderer-toggle" role="group" aria-label="Centerpiece renderer">
            <button
              type="button"
              className={renderer === 'flowbench' ? 'is-active' : ''}
              aria-pressed={renderer === 'flowbench'}
              onClick={() => pickRenderer('flowbench')}
            >
              Flow Bench
            </button>
            <button
              type="button"
              className={renderer === 'neuralfield' ? 'is-active' : ''}
              aria-pressed={renderer === 'neuralfield'}
              onClick={() => pickRenderer('neuralfield')}
            >
              Neural Field
            </button>
          </div>
          <EvidenceChip
            state="neutral"
            label="live session stats"
            source={{ note: 'Drawn from this session’s real requests only — counts and timings, never message text. Verified model support lives on the Compatibility page.' }}
            size="sm"
          />
        </div>
      </header>

      <div className="flowbench-stage">
        {renderer === 'neuralfield'
          ? <NeuralField apiBase={apiBase} reducedMotion={reducedMotion} />
          : <FlowBench reducedMotion={reducedMotion} highlightId={highlightId} />}
        <aside className="flowbench-rail" aria-label="Live instruments">
          <div className="flowbench-rail__tiles">
            <div className="cxv-stat"><span>Requests</span><strong>{summary.total}</strong><small>{summary.errors} error{summary.errors === 1 ? '' : 's'}</small></div>
            <div className="cxv-stat"><span>TTFT med</span><strong>{fmtMs(summary.medianTtftMs)}</strong><small>client-measured</small></div>
            <div className="cxv-stat"><span>Output rate medians</span>{summary.outputRates.length ? summary.outputRates.map((rate) => <Fragment key={rate.source}><strong>{fmtRate(rate.median)}</strong><small>{rateSourceLabel(rate.source)}</small></Fragment>) : <strong>—</strong>}</div>
          </div>
          <div className="flowbench-rail__model">
            <span className="flowbench-rail__model-id">{activeModelId || 'no model loaded'}</span>
            {gate.hint?.target && (
              <EvidenceChip
                status={gate.hint.target.status}
                state={gate.contractSupported ? 'supported' : null}
                source={{ rowId: gate.hint.target.id }}
                size="sm"
              />
            )}
          </div>
          <ol className="flowbench-rail__log" aria-label="Recent requests — hover to highlight the ink thread">
            {recent.length === 0 && <li className="flowbench-rail__empty">No session traffic yet — the bench stays still until a real request runs.</li>}
            {recent.map((record) => (
              <li
                key={record.id}
                className={`flowbench-rail__row ${record.outcome !== 'ok' ? 'is-error' : ''} ${highlightId === record.id ? 'is-highlit' : ''}`}
                onMouseEnter={() => setHighlightId(record.id)}
                onMouseLeave={() => setHighlightId(null)}
              >
                <code>{record.id}</code>
                <span>{record.kind === 'chat' ? 'chat' : record.endpoint}</span>
                <span>{record.outcome}</span>
                <span>{fmtMs(record.durationMs)}</span>
              </li>
            ))}
          </ol>
          <div className="flowbench-rail__foot">
            <button type="button" className="cxturn__action" onClick={() => setManualReduced((value) => !value)} aria-pressed={manualReduced}>
              {reducedMotion ? 'Motion: static field' : 'Motion: live'}
            </button>
            <a className="flowbench-rail__link" href="#telemetry">full request log &amp; health history <IconChevronRight size={12} /></a>
          </div>
        </aside>
      </div>

      <div className="flowbench-backend">
        <DetailsPanel store={store} collapsed={detailsCollapsed} onToggle={() => setDetailsCollapsed((value) => !value)} />
        <p className="tele-note">Backend stream · camelid.telemetry/v1</p>
      </div>
    </section>
  )
}
