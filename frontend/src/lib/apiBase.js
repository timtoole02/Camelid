import { appStorage } from './appStorage.js'

export const API_BASE_STORAGE_KEY = 'camelid.apiBase'

export function defaultApiBase() {
  if (import.meta.env?.VITE_CAMELID_API_BASE) return import.meta.env.VITE_CAMELID_API_BASE
  if (typeof window !== 'undefined' && window.location?.origin) return window.location.origin
  return 'http://127.0.0.1:8181'
}

export function getApiBase() {
  return (typeof window !== 'undefined' && appStorage.getItem(API_BASE_STORAGE_KEY)) || defaultApiBase()
}

export function normalizeApiBase(value) {
  return (value || defaultApiBase()).trim().replace(/\/$/, '')
}

export function apiUrl(path, apiBase = getApiBase()) {
  return `${normalizeApiBase(apiBase)}${path}`
}
