/* Desktop-shell detection, shared so the browser build never pays for it. */
export function isDesktopShell() {
  if (typeof window === 'undefined') return false
  return Boolean(window.__TAURI__?.core?.invoke)
}
