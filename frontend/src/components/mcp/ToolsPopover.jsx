import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

const FOCUSABLE = 'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary, [href]'

// Portal out of the composer's scrolling toolbar. Position against the visual
// viewport so zoom, short windows, and an on-screen keyboard keep it reachable.
export function ToolsPopover({ anchorRef, id, titleId, onClose, children }) {
  const panelRef = useRef(null)
  const closeRef = useRef(onClose)
  closeRef.current = onClose
  const [position, setPosition] = useState(null)
  useLayoutEffect(() => {
    const panel = panelRef.current
    const anchor = anchorRef.current
    if (!panel || !anchor) return undefined
    const update = () => {
      const viewport = window.visualViewport
      const left = viewport?.offsetLeft || 0
      const top = viewport?.offsetTop || 0
      const width = viewport?.width || window.innerWidth
      const height = viewport?.height || window.innerHeight
      const mobile = width <= 600
      const rect = anchor.getBoundingClientRect()
      const box = anchor.closest('.cxcomposer__box')?.getBoundingClientRect() || rect
      const above = Math.max(0, box.top - top - 20)
      const below = Math.max(0, top + height - rect.bottom - 20)
      const atTop = above >= Math.min(380, below)
      const maxHeight = Math.max(0, Math.min(height - 24, mobile ? 680 : atTop ? above : below))
      const panelWidth = Math.min(388, width - 24)
      const actualHeight = Math.min(panel.scrollHeight, maxHeight)
      const next = {
        mobile,
        width: mobile ? width - 24 : panelWidth,
        maxHeight,
        left: mobile ? left + 12 : Math.min(Math.max(left + 12, rect.left), left + width - panelWidth - 12),
        top: mobile ? top + height - actualHeight - 12
          : Math.min(Math.max(top + 12, atTop ? box.top - actualHeight - 8 : rect.bottom + 8), top + height - actualHeight - 12),
      }
      setPosition(previous => JSON.stringify(previous) === JSON.stringify(next) ? previous : next)
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(panel)
    observer.observe(anchor)
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    window.visualViewport?.addEventListener('resize', update)
    window.visualViewport?.addEventListener('scroll', update)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
      window.visualViewport?.removeEventListener('resize', update)
      window.visualViewport?.removeEventListener('scroll', update)
    }
  }, [anchorRef])
  useEffect(() => {
    const panel = panelRef.current
    const anchor = anchorRef.current
    const frame = requestAnimationFrame(() => panel?.querySelector('input[type="search"]')?.focus({ preventScroll: true }))
    const outside = event => {
      if (!panel?.contains(event.target) && !anchor?.contains(event.target)) closeRef.current(event.target.classList?.contains('mcp-picker-scrim'))
    }
    const key = event => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeRef.current(true); return }
      if (event.key !== 'Tab' || !panel?.classList.contains('is-mobile')) return
      const elements = [...panel.querySelectorAll(FOCUSABLE)].filter(element => element.getClientRects().length)
      const first = elements[0], last = elements.at(-1)
      if (event.shiftKey && (document.activeElement === first || !panel.contains(document.activeElement))) { event.preventDefault(); last?.focus() }
      if (!event.shiftKey && (document.activeElement === last || !panel.contains(document.activeElement))) { event.preventDefault(); first?.focus() }
    }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('focusin', outside)
    document.addEventListener('keydown', key, true)
    return () => {
      cancelAnimationFrame(frame)
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('focusin', outside)
      document.removeEventListener('keydown', key, true)
    }
  }, [anchorRef])
  return createPortal(<>
    {position?.mobile && <div className="mcp-picker-scrim" aria-hidden="true" />}
    <section ref={panelRef} id={id} role="dialog" aria-modal={position?.mobile || undefined} aria-labelledby={titleId}
      className={'mcp-picker' + (position?.mobile ? ' is-mobile' : '')}
      style={{ left: position?.left, top: position?.top, width: position?.width, maxHeight: position?.maxHeight, visibility: position ? 'visible' : 'hidden' }}>
      {children}
    </section>
  </>, document.body)
}
