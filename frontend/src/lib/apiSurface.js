const LAN_CHAT_TABS = new Set(['chat', 'projects', 'history', 'memory', 'settings'])

export function isLanChatOnly(apiSurface) {
  return apiSurface === 'lan_chat_only'
}

export function apiSurfaceAllowsTab(apiSurface, tab) {
  return !isLanChatOnly(apiSurface) || LAN_CHAT_TABS.has(tab)
}
