import { useCallback, useLayoutEffect, useRef } from 'react'

export const RETRO_TRANSITION_KEY = 'camelid.retroTransitions'
export const MOSAIC_STEPS = [48, 32, 24, 16, 12, 8, 4, 2]
const DIVE_MS = 640
const REVEAL_MS = 560

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
    clearTimeout(run.watchdog)
    cancelAnimationFrame(run.frame)
    run.observer?.disconnect()
    run.animation?.cancel()
    run.view.style.removeProperty('visibility')
    run.view.style.removeProperty('filter')
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
    const pixelScale = 1 / Math.max(view.clientWidth, view.clientHeight, 1)
    run.animation = view.animate([
      { transform: 'perspective(1000px) rotateX(0deg) rotateZ(0deg) scale(1)', offset: 0 },
      { transform: 'perspective(1000px) rotateX(48deg) rotateZ(-24deg) scale(.88)', offset: .26 },
      { transform: 'perspective(1000px) rotateX(58deg) rotateZ(145deg) scale(.35)', offset: .65 },
      { transform: `perspective(1000px) rotateX(0deg) rotateZ(360deg) scale(${pixelScale})`, offset: 1 },
    ], { duration: DIVE_MS, easing: 'cubic-bezier(.55, 0, .8, .45)', fill: 'forwards' })
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
    }, 2500)
  }, [cancel, enabled, setTab, stageRef, viewRef])

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
