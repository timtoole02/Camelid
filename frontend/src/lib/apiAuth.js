export const API_BASE_STORAGE_KEY = 'camelid.apiBase'
export const API_KEY_STORAGE_KEY = 'camelid.apiKey'

export function getStoredApiKey() {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(API_KEY_STORAGE_KEY) || ''
}

export function setStoredApiKey(value) {
  if (typeof window === 'undefined') return
  const next = String(value || '').trim()
  if (next) window.localStorage.setItem(API_KEY_STORAGE_KEY, next)
  else window.localStorage.removeItem(API_KEY_STORAGE_KEY)
}

export function installApiAuthFetch() {
  if (typeof window === 'undefined' || window.__camelidApiAuthFetchInstalled) return
  const nativeFetch = window.fetch.bind(window)

  window.fetch = (input, init = {}) => {
    const apiKey = getStoredApiKey()
    if (!apiKey) return nativeFetch(input, init)

    try {
      const requestUrl = new URL(typeof input === 'string' || input instanceof URL ? input : input.url, window.location.href)
      const configuredBase = window.localStorage.getItem(API_BASE_STORAGE_KEY) || window.location.origin
      const apiOrigin = new URL(configuredBase, window.location.href).origin
      if (requestUrl.origin !== apiOrigin) return nativeFetch(input, init)

      const headers = new Headers(init.headers || (typeof Request !== 'undefined' && input instanceof Request ? input.headers : undefined))
      if (!headers.has('authorization') && !headers.has('x-api-key')) headers.set('x-api-key', apiKey)
      return nativeFetch(input, { ...init, headers })
    } catch {
      return nativeFetch(input, init)
    }
  }

  window.__camelidApiAuthFetchInstalled = true
}
