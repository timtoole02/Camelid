import { apiUrl, getApiBase } from './apiBase.js'
import { getStoredApiKey } from './apiAuth.js'

/* Library requests use the same selected engine and explicit credentials as chat. */
export function apiFetch(path, init = {}, apiBase = getApiBase()) {
  const url = apiUrl(path, apiBase)
  const headers = new Headers(init.headers)
  const href = typeof window !== 'undefined' ? window.location.href : 'http://127.0.0.1:8181/'
  // Another tab may have changed the saved connection. Never send its key to
  // the different backend still selected in this view.
  const key = new URL(url, href).origin === new URL(getApiBase(), href).origin ? getStoredApiKey() : ''
  if (key && !headers.has('authorization') && !headers.has('x-api-key')) headers.set('x-api-key', key)
  return fetch(url, { ...init, headers })
}
