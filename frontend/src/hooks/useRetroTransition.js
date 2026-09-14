import { useCallback, useLayoutEffect, useRef } from 'react'

export const RETRO_TRANSITION_KEY = 'camelid.retroTransitions'
export const MOSAIC_STEPS = [48, 32, 24, 16, 12, 8, 4, 2]
const DIVE_MS = 900
const REVEAL_MS = 560

// Prefer a visible text pixel so the destination is part of the screen's
// content. Use source-pixel centers so the final sample matches the zoom target.
function pickPixel(view) {
  const bounds = view.getBoundingClientRect()
  let point = { x: bounds.width * .5, y: bounds.height * .45 }
  for (const node of view.querySelectorAll('h2, p')) {
    if (node.closest('[hidden]')) continue
    const range = document.createRange()
    range.selectNodeContents(node)
    const rect = [...range.getClientRects()].find(rect => rect.width > 8 && rect.height > 8 && rect.top > bounds.top + 16 && rect.bottom < bounds.bottom - 16)
    if (!rect) continue
    const x = rect.left + rect.width * .4
    const y = rect.top + rect.height * .55
    if (x > bounds.left + 16 && x < bounds.right - 16) {
      point = { x: x - bounds.left, y: y - bounds.top }
      break
    }
  }
  const snap = (position, limit) => Math.max(4.5, Math.min(Math.floor((limit - 8) / 8) * 8 + 4.5, Math.floor(position / 8) * 8 + 4.5))
  return { x: snap(point.x, bounds.width), y: snap(point.y, bounds.height) }
}

/** Animate only the content surface; navigation stays available to cancel it. */
export function useRetroTransition({ tab, setTab, enabled, viewRef, stageRef, filterPrefix }) {
  const runRef = useRef(null)
  const tabRef = useRef(tab)
  tabRef.current = tab

  const cancel = useCallback(() => {
    const run = runRef.current
    if (!run) return
    runRef.current = null
    clearTimeout(run.timer)
    clearTimeout(run.pixelTimer)
    clearTimeout(run.watchdog)
    cancelAnimationFrame(run.frame)
    run.observer?.disconnect()
    run.animation?.cancel()
    run.view.style.removeProperty('visibility')
    run.view.style.removeProperty('filter')
    run.view.style.removeProperty('transform-origin')
    run.stage.style.removeProperty('--retro-target-x')
    run.stage.style.removeProperty('--retro-target-y')
    run.stage.style.removeProperty('--retro-dive-duration')
    run.view.inert = false
    delete run.stage.dataset.retroPhase
  }, [])

  const navigate = useCallback((next) => {
    const current = tabRef.current
    const running = runRef.current
    if (running?.to === next) return
    cancel()
    const pair = (current === 'chat' && next === 'code') || (current === 'code' && next === 'chat')
    const view = viewRef.current
    const stage = stageRef.current
    if (!enabled || !pair || !view?.animate || !stage || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setTab(next)
      return
    }

    const run = { from: current, to: next, view, stage, phase: 'dive' }
    runRef.current = run
    stage.dataset.retroPhase = 'dive'
    view.inert = true
    const pixel = pickPixel(view)
    view.style.transformOrigin = `${pixel.x}px ${pixel.y}px`
    stage.style.setProperty('--retro-target-x', `${pixel.x}px`)
    stage.style.setProperty('--retro-target-y', `${pixel.y}px`)
    stage.style.setProperty('--retro-dive-duration', `${DIVE_MS}ms`)
    const sample = document.getElementById(`${filterPrefix}-sample`)
    sample?.setAttribute('x', String(pixel.x - .5))
    sample?.setAttribute('y', String(pixel.y - .5))
    // Keep the source view unfiltered during the zoom. Finish by expanding
    // the exact sampled pixel into the final solid-color frame.
    const pixelScale = Math.max(2, Math.min(32, 16384 / Math.max(view.clientWidth, view.clientHeight, 1)))
    run.animation = view.animate([
      { transform: 'scale(1) perspective(1000px) rotateX(0deg) rotateZ(0deg)', offset: 0 },
      { transform: 'scale(1) perspective(1000px) rotateX(0deg) rotateZ(0deg)', offset: .18 },
      { transform: 'scale(2) perspective(1000px) rotateX(-18deg) rotateZ(-24deg)', offset: .42 },
      { transform: 'scale(10) perspective(1000px) rotateX(-32deg) rotateZ(145deg)', offset: .76 },
      { transform: `scale(${pixelScale}) perspective(1000px) rotateX(0deg) rotateZ(360deg)`, offset: 1 },
    ], { duration: DIVE_MS, easing: 'cubic-bezier(.55, 0, .8, .45)', fill: 'forwards' })
    run.pixelTimer = window.setTimeout(() => {
      if (runRef.current !== run) return
      run.animation.cancel()
      stage.dataset.retroPhase = 'pixel'
      view.style.filter = `url(#${filterPrefix}-pixel)`
    }, DIVE_MS - 100)
    run.timer = window.setTimeout(() => {
      if (runRef.current !== run) return
      run.phase = 'waiting'
      stage.dataset.retroPhase = 'waiting'
      view.style.visibility = 'hidden'
      run.animation.cancel()
      setTab(next)
    }, DIVE_MS)
    // A lazy view, suspended tab or missed animation event must never trap UI.
    run.watchdog = window.setTimeout(() => {
      if (runRef.current !== run) return
      const destination = run.to
      cancel()
      setTab(destination)
    }, 3500)
  }, [cancel, enabled, filterPrefix, setTab, stageRef, viewRef])

  useLayoutEffect(() => {
    const run = runRef.current
    if (!run) return
    if (!enabled || (run.phase === 'dive' ? tab !== run.from : tab !== run.to)) {
      cancel()
      return
    }
    if (run.phase !== 'waiting') return
    const reveal = () => {
      if (runRef.current !== run || run.view.querySelector('.view-loading')) return
      run.observer?.disconnect()
      run.phase = 'mosaic'
      run.stage.dataset.retroPhase = 'mosaic'
      run.view.style.filter = `url(#${filterPrefix}-48)`
      run.view.style.removeProperty('visibility')
      const started = performance.now()
      let lastStep = 48
      const tick = (now) => {
        if (runRef.current !== run) return
        const progress = (now - started) / REVEAL_MS
        if (progress >= 1) { cancel(); return }
        const size = MOSAIC_STEPS[Math.floor(progress * MOSAIC_STEPS.length)]
        if (size !== lastStep) {
          run.view.style.filter = `url(#${filterPrefix}-${size})`
          lastStep = size
        }
        run.frame = requestAnimationFrame(tick)
      }
      run.frame = requestAnimationFrame(tick)
    }
    run.observer = new MutationObserver(reveal)
    run.observer.observe(run.view, { childList: true, subtree: true })
    reveal()
  }, [tab, enabled, cancel, filterPrefix])

  useLayoutEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const skip = () => {
      const destination = runRef.current?.to
      cancel()
      if (destination) setTab(destination)
    }
    const onMotion = () => { if (media.matches) skip() }
    const onKey = (event) => { if (event.key === 'Escape') skip() }
    const onVisibility = () => { if (document.hidden) skip() }
    media.addEventListener('change', onMotion)
    window.addEventListener('keydown', onKey)
    window.addEventListener('resize', skip)
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      cancel()
      media.removeEventListener('change', onMotion)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('resize', skip)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [cancel, setTab])

  return { navigate, cancel }
}
