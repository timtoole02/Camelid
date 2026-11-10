/* Star Nursery renderer — inference drawn as a nebula in which every decoded
   token ignites a star. By the end of a run the reply is a constellation.
   Standard module interface (`onEvent(evt, frame, opts)`, `draw(ctx, frame)`),
   same as the Neural Field renderer.

   Truthfulness: every lit element traces to a camelid.telemetry/v1 event or a
   store field. Nothing is spawned in draw(); an idle store leaves the nebula
   at rest with only the declared ambient treatment (slow gas drift, twinkle).

   | Element                               | Driven by                                   |
   | ------------------------------------- | ------------------------------------------- |
   | Nebula wake (brightness ramp)         | inference_started / inference_finished      |
   | Previous constellation fades out      | inference_started                           |
   | Inbound dust motes (<= 96)            | prefill_started.prefill_tokens              |
   | Protostar size                        | prefill_progress tokens_done / tokens_total |
   | Protostar release                     | decode_started                              |
   | Condensing ring at the next position  | layer_started / layer_completed (CPU lanes) |
   | One new star + flare, linked to prior | token_decoded (one star per token)          |
   | Nebula brightness swell               | token_decoded cadence (decaying envelope)   |
   | Nebula spread                         | run.kv.position / run.kv.capacity           |
   | Red shift, flares stop                | inference_error, non-ok inference_finished  |
   | Single exhale pulse                   | inference_finished { status: "ok" }         |
   | Constellation lines turn gold + burst | receipt_written                             |
   | Dimmed field                          | connection === 'unavailable'                |

   Declared abstractions: star positions follow a fixed open curve indexed by
   token order (position carries order only, never token identity or text);
   star size/tint and gas placement use visual randomness for layout only.
   Sampler candidates are not rendered. Only the most recent STAR_CAP tokens
   of a run are kept on screen.

   Palette: this canvas is its own deliberately colourful, always-dark world
   (like Fusion Core), so its inks are literal rather than theme tokens. */

const TAU = Math.PI * 2
export const MOTE_CAP = 96
export const STAR_CAP = 800

const WAKE_RAMP_MS = 400
const SETTLE_MS = 900
const FLARE_MS = 550
const EXHALE_MS = 900
const GHOST_FADE_MS = 1200
const SEAL_MS = 1100
const ACTIVITY_TAU_MS = 450
const PAST_FLARE = FLARE_MS + 1

const INK = {
  white: '255,255,255',
  cream: '255,240,205',
  gold: '255,180,84',
  pink: '255,77,157',
  violet: '143,107,255',
  blue: '84,130,255',
  cyan: '95,224,255',
  teal: '53,230,192',
  error: '255,90,90',
}
const GAS = [INK.pink, INK.violet, INK.teal, INK.blue, INK.gold, INK.violet, INK.pink]
const STAR_INKS = [INK.cream, INK.white, INK.cyan, INK.gold, INK.pink, INK.cream]

const rand = (a, b) => a + Math.random() * (b - a)

const sprites = new Map()
function sprite(ink) {
  let s = sprites.get(ink)
  if (s) return s
  s = document.createElement('canvas')
  s.width = 64
  s.height = 64
  const g = s.getContext('2d')
  const gradient = g.createRadialGradient(32, 32, 0, 32, 32, 32)
  gradient.addColorStop(0, `rgba(${ink},1)`)
  gradient.addColorStop(0.18, `rgba(${ink},0.6)`)
  gradient.addColorStop(0.5, `rgba(${ink},0.15)`)
  gradient.addColorStop(1, `rgba(${ink},0)`)
  g.fillStyle = gradient
  g.fillRect(0, 0, 64, 64)
  sprites.set(ink, s)
  return s
}

function glow(ctx, x, y, r, ink, alpha) {
  if (alpha <= 0.004 || r <= 0) return
  ctx.globalAlpha = Math.min(1, alpha)
  ctx.drawImage(sprite(ink), x - r, y - r, r * 2, r * 2)
}

/* Open curve the constellation follows: token order -> canvas fraction. The
   two frequencies are incommensurate, so long replies keep finding new sky. */
function starPosition(i, phase) {
  const a = i * 0.045 + phase
  return {
    x: 0.5 + 0.39 * Math.sin(a) + 0.018 * Math.sin(i * 1.7),
    y: 0.5 + 0.3 * Math.sin(a * 1.618 + 1.3) + 0.03 * Math.cos(i * 2.3) + 0.014 * Math.sin(i * 5.1),
  }
}

export function createStarNurseryRenderer({ reducedMotion = false } = {}) {
  let reduced = reducedMotion
  const field = Array.from({ length: 220 }, () => ({
    x: Math.random(),
    y: Math.random(),
    s: Math.pow(Math.random(), 3) * 1.5 + 0.4,
    p: rand(0, TAU),
    ink: Math.random() < 0.14 ? INK.gold : Math.random() < 0.3 ? INK.cyan : INK.white,
  }))
  const gas = Array.from({ length: 14 }, (_, i) => ({
    x: rand(0.08, 0.92),
    y: rand(0.18, 0.82),
    r: rand(0.24, 0.46),
    ink: GAS[i % GAS.length],
    p: rand(0, TAU),
    sp: rand(0.06, 0.16),
  }))

  const state = {
    wake: 0,
    awake: false,
    activity: 0,
    error: 0,
    errorLatched: false,
    exhaleAt: null,
    sealed: false,
    seal: 0,
    prefill: { fill: 0, glow: 0, released: false },
    dust: [], // inbound prompt motes { a, d, t } (t < 0 = still queued)
    stars: [], // current run { x, y, s, ink, age }
    ghosts: [], // previous run, fading out
    ghostFade: 0,
    index: 0,
    phase: rand(0, TAU),
    forming: null, // 0..1 layer progress toward the next star (CPU lanes)
    bursts: [], // receipt flares { x, y, age }
  }

  function beginRun(replay) {
    state.awake = true
    state.errorLatched = false
    state.exhaleAt = null
    state.sealed = false
    state.seal = 0
    state.prefill = { fill: 0, glow: 0, released: false }
    state.dust = []
    state.bursts = []
    state.ghosts = replay ? [] : state.stars
    state.ghostFade = replay || !state.stars.length ? 0 : 1
    state.stars = []
    state.index = 0
    state.forming = null
    state.phase = rand(0, TAU)
  }

  /* `opts.replay` marks events drained from the store's ring on mount: they
     rebuild the last run's resting picture without re-animating it. */
  function onEvent(evt, frame, opts = {}) {
    const replay = Boolean(opts.replay)
    const quiet = replay || reduced
    switch (evt.event) {
      case 'inference_started':
        beginRun(replay)
        break
      case 'prefill_started': {
        state.awake = true
        state.prefill = { fill: 0, glow: 1, released: false }
        if (!quiet) {
          const count = Math.min(evt.prefill_tokens || 0, MOTE_CAP)
          for (let i = 0; i < count; i += 1) {
            state.dust.push({ a: rand(0, TAU), d: rand(0.25, 0.5), t: -(i / count) * 0.7 })
          }
        }
        break
      }
      case 'prefill_progress': {
        const total = evt.tokens_total || frame.run.prefill.tokens || 0
        if (total > 0) state.prefill.fill = Math.min((evt.tokens_done || 0) / total, 1)
        break
      }
      case 'decode_started':
        state.prefill.released = true
        break
      case 'layer_started':
      case 'layer_completed': {
        const total = evt.layers_total || frame.run.layersTotal
        if (total > 0 && typeof evt.layer === 'number') state.forming = Math.min((evt.layer + 1) / total, 1)
        break
      }
      case 'token_decoded': {
        state.awake = state.awake || !replay
        state.prefill.released = true
        const p = starPosition(state.index, state.phase)
        state.stars.push({
          x: p.x,
          y: p.y,
          s: rand(0.6, 1.9),
          ink: STAR_INKS[(Math.random() * STAR_INKS.length) | 0],
          age: quiet ? PAST_FLARE : 0,
        })
        if (state.stars.length > STAR_CAP) state.stars.shift()
        state.index += 1
        state.forming = null
        if (!replay) state.activity = Math.min(1, state.activity + 0.14)
        break
      }
      case 'inference_error':
        state.errorLatched = true
        break
      case 'inference_finished':
        state.awake = false
        state.forming = null
        state.prefill.released = true
        state.errorLatched = evt.status !== 'ok'
        if (evt.status === 'ok' && !quiet) state.exhaleAt = frame.t
        break
      case 'receipt_written': {
        state.sealed = true
        if (replay) state.seal = 1
        const last = state.stars[state.stars.length - 1]
        if (last && !quiet) state.bursts.push({ x: last.x, y: last.y, age: 0 })
        break
      }
      default:
        break
    }
  }

  function toward(value, target, dt, ms) {
    const stepSize = dt / ms
    return value < target ? Math.min(target, value + stepSize) : Math.max(target, value - stepSize)
  }

  function drawConstellation(ctx, stars, w, h, k, fade, calm) {
    if (!stars.length || fade <= 0.01) return
    if (stars.length > 1) {
      ctx.globalAlpha = Math.min(1, (0.16 + 0.55 * state.seal) * fade)
      ctx.lineWidth = 1 + state.seal * 0.6
      ctx.strokeStyle = state.error > 0.5 ? `rgb(${INK.error})` : state.seal > 0 ? `rgb(${INK.cream})` : 'rgb(170,190,255)'
      ctx.beginPath()
      for (let i = 0; i < stars.length; i += 1) {
        const s = stars[i]
        if (i) ctx.lineTo(s.x * w, s.y * h)
        else ctx.moveTo(s.x * w, s.y * h)
      }
      ctx.stroke()
    }
    for (const s of stars) {
      const x = s.x * w
      const y = s.y * h
      const r = (4 + 3.5 * s.s) * k + 1.5
      glow(ctx, x, y, r, s.ink, (0.75 + 0.25 * state.seal) * fade * (1 - 0.65 * state.error))
      if (state.error > 0.02) glow(ctx, x, y, r, INK.error, 0.7 * state.error * fade)
      if (!calm && s.age < FLARE_MS) {
        const f = 1 - s.age / FLARE_MS
        const len = (26 * f + 4) * k * s.s
        glow(ctx, x, y, (30 * f + 8) * k, INK.white, f)
        glow(ctx, x, y, 60 * f * k, INK.gold, f * 0.5)
        ctx.globalAlpha = f
        ctx.strokeStyle = 'rgb(255,255,255)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(x - len, y)
        ctx.lineTo(x + len, y)
        ctx.moveTo(x, y - len)
        ctx.lineTo(x, y + len)
        ctx.stroke()
      }
    }
  }

  function draw(ctx, frame) {
    const { w, h, t, dt, run, connection } = frame
    // Width the constellation may use: an overlaid instrument rail (if any)
    // is kept clear. The sky and gas still fill the whole canvas.
    const sw = Math.max(w - (frame.insetRight || 0), w * 0.5)
    const unavailable = connection === 'unavailable'

    // A run whose events stopped without inference_finished settles via the
    // store's RUN_STALE_MS path, exactly like the Neural Field.
    if (frame.runStale && state.awake) state.awake = false

    // ---- advance state (decay toward rest; nothing is created here) ----
    state.wake = toward(state.wake, state.awake ? 1 : 0, dt, state.awake ? WAKE_RAMP_MS : SETTLE_MS)
    state.activity *= Math.exp(-dt / ACTIVITY_TAU_MS)
    state.error = toward(state.error, state.errorLatched ? 1 : 0, dt, 300)
    if (state.sealed) state.seal = toward(state.seal, 1, dt, SEAL_MS)
    if (state.ghostFade > 0) {
      state.ghostFade = Math.max(0, state.ghostFade - dt / GHOST_FADE_MS)
      if (state.ghostFade === 0) state.ghosts = []
    }
    if (state.prefill.released) state.prefill.glow = Math.max(0, state.prefill.glow - dt / 500)
    for (const s of state.stars) s.age += dt

    let pulse = 0
    if (state.exhaleAt != null) {
      const u = (t - state.exhaleAt) / EXHALE_MS
      if (u >= 1) state.exhaleAt = null
      else pulse = Math.sin(Math.PI * Math.max(u, 0)) * 0.25
    }

    const m = Math.min(w, h)
    const k = m / 620
    const dim = unavailable ? 0.25 : 1
    const clock = reduced ? 0 : t / 1000
    const kvFrac = run.kv.capacity > 0 ? Math.min(run.kv.position / run.kv.capacity, 1) : 0
    const energy = (0.1 + 0.2 * state.wake + 0.5 * state.activity + pulse) * dim

    // ---- sky ----
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1
    ctx.fillStyle = '#05060d'
    ctx.fillRect(0, 0, w, h)
    const vignette = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.max(w, h) * 0.7)
    vignette.addColorStop(0, 'rgba(24,28,66,0.6)')
    vignette.addColorStop(1, 'rgba(5,6,13,0)')
    ctx.fillStyle = vignette
    ctx.fillRect(0, 0, w, h)

    ctx.globalCompositeOperation = 'lighter'
    for (const s of field) {
      ctx.globalAlpha = (0.3 + 0.3 * Math.sin(clock * 0.8 + s.p)) * dim
      ctx.fillStyle = `rgb(${s.ink})`
      ctx.fillRect(s.x * w, s.y * h, s.s, s.s)
    }

    // ---- nebula gas: brightness = wake + token cadence, spread = KV fill ----
    for (const b of gas) {
      const x = (b.x + 0.04 * Math.sin(clock * b.sp + b.p)) * w
      const y = (b.y + 0.05 * Math.cos(clock * b.sp * 1.3 + b.p)) * h
      const r = b.r * m * (0.85 + 1.1 * kvFrac + 0.1 * Math.sin(clock * 0.3 + b.p))
      glow(ctx, x, y, r, b.ink, (0.09 + 0.17 * energy) * dim * (1 - 0.8 * state.error))
      if (state.error > 0.02) glow(ctx, x, y, r, INK.error, 0.1 * state.error * dim)
    }

    // ---- prompt dust falling into the protostar where the reply begins ----
    const origin = starPosition(0, state.phase)
    const ox = origin.x * sw
    const oy = origin.y * h
    if (state.prefill.glow > 0.01) {
      const fill = state.prefill.fill
      glow(ctx, ox, oy, m * (0.03 + 0.09 * fill), INK.cyan, (0.35 + 0.5 * fill) * state.prefill.glow)
      glow(ctx, ox, oy, m * 0.02, INK.white, fill * state.prefill.glow)
    }
    if (state.dust.length) {
      state.dust = state.dust.filter((d) => {
        d.t += dt / 950
        if (d.t >= 1) return false
        if (d.t < 0) return true
        const e = 1 - d.t * d.t
        const a = d.a + d.t * 2.2
        glow(ctx, ox + Math.cos(a) * d.d * m * e, oy + Math.sin(a) * d.d * m * e * 0.7, 4 * k + 2, INK.cyan, 0.9 * Math.min(1, d.t * 4))
        return true
      })
    }

    // ---- constellations: the fading previous run, then the current one ----
    drawConstellation(ctx, state.ghosts, sw, h, k, state.ghostFade * 0.6 * dim, true)
    drawConstellation(ctx, state.stars, sw, h, k, dim, reduced)

    // Next star condensing as layers complete (only where layer events exist).
    if (state.forming != null && state.awake && !reduced) {
      const p = starPosition(state.index, state.phase)
      ctx.globalAlpha = 0.25 + 0.45 * state.forming
      ctx.strokeStyle = `rgb(${INK.cream})`
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(p.x * sw, p.y * h, ((1 - state.forming) * 22 + 4) * k, 0, TAU)
      ctx.stroke()
    }

    // Receipt burst at the final star.
    if (state.bursts.length) {
      state.bursts = state.bursts.filter((b) => {
        b.age += dt
        const f = 1 - b.age / 1200
        if (f <= 0) return false
        glow(ctx, b.x * sw, b.y * h, (40 + 140 * (1 - f)) * k, INK.gold, f * 0.7)
        glow(ctx, b.x * sw, b.y * h, 24 * k, INK.white, f)
        return true
      })
    }

    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1
  }

  return {
    onEvent,
    draw,
    setReducedMotion(value) {
      reduced = Boolean(value)
    },
    // Test/debug seam: current state (read-only use).
    _state: state,
  }
}
