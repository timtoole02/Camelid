/* StarNursery — canvas host for the Star Nursery renderer: a nebula in which
   every decoded token ignites a star, so a finished reply is a constellation.
   Same rAF/DPR/ResizeObserver/cleanup skeleton as the Neural Field host; the
   data path is the backend SSE store (useInferenceTelemetry +
   store.drainEvents), never the client-side telemetryLog bus.

   Idle = the nebula at rest; the only standing motion is the explicitly-idle
   ambient treatment (slow gas drift + twinkle), never an inference signal.
   Reduced motion: no drift, no dust, no flares; stars simply appear.

   The floating readouts are plain DOM over the canvas and read the same
   store, so the picture and the numbers cannot disagree. */

import { useEffect, useRef } from 'react'
import { useInferenceTelemetry } from '../../hooks/useInferenceTelemetry'
import { createStarNurseryRenderer } from '../../lib/observatory/starNursery/renderer'
import { fmtRate } from './format'

const DPR_CAP = 2
const CONSUMER_ID = 'starnursery'

const PHASE_LABEL = {
  idle: 'Idle',
  running: 'Starting',
  prefill: 'Reading prompt',
  decode: 'Generating',
  finished: 'Complete',
  error: 'Error',
}

export default function StarNursery({ apiBase, reducedMotion = false }) {
  const canvasRef = useRef(null)
  const store = useInferenceTelemetry(apiBase)
  const storeRef = useRef(store)
  storeRef.current = store

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d')
    const renderer = createStarNurseryRenderer({ reducedMotion })

    let raf = 0
    let lastT = performance.now()
    let dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP)

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP)
      const rect = canvas.getBoundingClientRect()
      canvas.width = Math.max(1, Math.round(rect.width * dpr))
      canvas.height = Math.max(1, Math.round(rect.height * dpr))
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(canvas)

    const frameFor = (t, dt) => {
      const s = storeRef.current
      return {
        w: canvas.width / dpr,
        h: canvas.height / dpr,
        t,
        dt,
        run: s.getRun(),
        connection: s.getConnection(),
        runStale: s.isRunStale(),
      }
    }

    /* On mount, the first drain is backlog: real events received while this
       canvas was not on screen. They rebuild the resting picture, flagged as
       a replay so nothing re-animates as if it were happening now. Where the
       store keeps a per-consumer ring, rewind it first to recover the whole
       latest run. */
    storeRef.current.rewindConsumer?.(CONSUMER_ID)
    const mountFrame = frameFor(lastT, 0)
    for (const evt of storeRef.current.drainEvents(CONSUMER_ID)) renderer.onEvent(evt, mountFrame, { replay: true })

    const tick = (t) => {
      raf = window.requestAnimationFrame(tick)
      const dt = Math.min(t - lastT, 50)
      lastT = t
      const frame = frameFor(t, dt)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      // Real events drained once per frame — the only animation source.
      for (const evt of storeRef.current.drainEvents(CONSUMER_ID)) renderer.onEvent(evt, frame)
      renderer.draw(ctx, frame)
    }
    raf = window.requestAnimationFrame(tick)

    const onVisibility = () => {
      if (document.hidden) {
        window.cancelAnimationFrame(raf)
        raf = 0
      } else if (!raf) {
        lastT = performance.now()
        raf = window.requestAnimationFrame(tick)
      }
    }
    document.addEventListener('visibilitychange', onVisibility)

    return () => {
      if (raf) window.cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', onVisibility)
      observer.disconnect()
    }
  }, [reducedMotion])

  const run = store.getRun()
  const unavailable = store.getConnection() === 'unavailable'
  const phase = unavailable ? 'unavailable' : run.phase
  const label = unavailable ? 'Stream unavailable' : PHASE_LABEL[run.phase] || 'Idle'
  const rate = run.active ? store.liveDecodeTps() : run.finish?.decodeTps
  const hasContext = run.kv.capacity > 0

  return (
    <div className="flowbench starnursery" data-phase={phase}>
      <canvas
        ref={canvasRef}
        className="flowbench__canvas"
        aria-label="Star Nursery — every decoded token of a real request ignites a star"
      />
      <div className="starnursery__hud starnursery__hud--phase">
        <span className="starnursery__dot" aria-hidden="true" />
        <span>{label}</span>
        {run.receipt && !unavailable && <span className="starnursery__sealed">receipt sealed</span>}
      </div>
      <div className="starnursery__hud starnursery__hud--readout">
        <div className="starnursery__rate">
          <strong>{fmtRate(rate)}</strong>
        </div>
        <div className="starnursery__sub">
          {run.decode.tokens} token{run.decode.tokens === 1 ? '' : 's'}
          {hasContext && <> · context {run.kv.position} / {run.kv.capacity}</>}
        </div>
      </div>
      <p className="starnursery__hud starnursery__hud--legend">one star = one decoded token</p>
    </div>
  )
}
