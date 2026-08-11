/* Desktop-shell detection, shared so the browser build never pays for it.

   Two different questions get asked about the desktop app and they are not the
   same: whether we are inside Tauri at all (BackendBanner asks this, to know a
   page cannot restart its own engine), and whether this window draws its own
   title bar. Only macOS uses the overlay style, where the traffic lights float
   over our content instead of sitting in a strip the OS owns — so only macOS
   needs the window to reserve room for them and provide a drag region. */
export function isDesktopShell() {
  if (typeof window === 'undefined') return false
  return Boolean(window.__TAURI__?.core?.invoke)
}

export function isMacPlatform() {
  if (typeof navigator === 'undefined') return false
  const platform = navigator.userAgentData?.platform || navigator.platform || ''
  return /mac/i.test(platform)
}

/* True only where the window chrome is ours to draw. Keep this the single
   source: styling that assumes the traffic lights are overlapping content must
   never switch on in the browser build, where there are none. */
export function hasOverlayTitleBar() {
  return isDesktopShell() && isMacPlatform()
}
