import { apiUrl } from './apiBase.js'
import { getStoredApiKey } from './apiAuth.js'

/* Library requests use the same selected engine and explicit credentials as chat. */
export function apiFetch(path, init = {}) {
  const headers = new Headers(init.headers)
  const key = getStoredApiKey()
  if (key && !headers.has('authorization') && !headers.has('x-api-key')) headers.set('x-api-key', key)
  return fetch(apiUrl(path), { ...init, headers })
}
