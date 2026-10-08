/* Watched folders: a collection can watch a folder on this computer and holds
   every document in it, kept current by the server. The folder routes answer
   only this web UI on the computer running Camelid. */

import { browseWorkspaceFolders } from './workspaceAgent.js'
import { apiFetch } from './apiRequest.js'
import { getApiBase } from './apiBase.js'

const JSON_HEADERS = { 'Content-Type': 'application/json' }

async function request(path, options = {}, apiBase = getApiBase()) {
  const response = await apiFetch(path, options, apiBase)
  if (response.status === 204) return null
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new Error(body?.error?.message || `The folder request failed (${response.status}).`)
  return body
}

const count = value => (Number.isFinite(Number(value)) && Number(value) > 0 ? Number(value) : 0)

function normalizeChanges(value) {
  if (!value || typeof value !== 'object') return null
  return {
    added: count(value.added),
    updated: count(value.updated),
    removed: count(value.removed),
    unchanged: count(value.unchanged),
    skipped: count(value.skipped),
  }
}

export function normalizeFolder(value) {
  const id = typeof value?.id === 'string' ? value.id : ''
  const path = typeof value?.path === 'string' ? value.path : ''
  const collectionId = typeof value?.collection_id === 'string' ? value.collection_id : ''
  if (!id || !path || !collectionId) return null
  const progress = value.progress && typeof value.progress === 'object'
    ? { done: count(value.progress.done), total: count(value.progress.total) }
    : null
  return {
    id,
    path,
    collectionId,
    lastScanAt: Number(value.last_scan_at) || null,
    lastError: typeof value.last_error === 'string' ? value.last_error : '',
    lastChanges: normalizeChanges(value.last_changes),
    documents: count(value.documents),
    skippedCount: count(value.skipped_count),
    skipped: (Array.isArray(value.skipped) ? value.skipped : [])
      .filter(item => typeof item?.path === 'string')
      .map(item => ({ path: item.path, reason: typeof item.reason === 'string' ? item.reason : '' })),
    scanning: Boolean(value.scanning),
    queued: Boolean(value.queued),
    progress,
  }
}

export const listFolders = async (apiBase = getApiBase()) => (await request('/api/folders', {}, apiBase) || []).map(normalizeFolder).filter(Boolean)

export const watchFolder = async (path, collectionId, apiBase = getApiBase()) => normalizeFolder(await request('/api/folders', {
  method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ path, collection_id: collectionId }),
}, apiBase))

export const scanFolder = async (id, apiBase = getApiBase()) => normalizeFolder(await request(`/api/folders/${encodeURIComponent(id)}/scan`, { method: 'POST' }, apiBase))

export const unwatchFolder = (id, apiBase = getApiBase()) => request(`/api/folders/${encodeURIComponent(id)}`, { method: 'DELETE' }, apiBase)

export const browseFolders = (path, apiBase = getApiBase()) => browseWorkspaceFolders(apiBase, path)

export const SKIP_REASONS = {
  too_large: 'larger than 64 MB',
  unreadable: 'could not be read',
  no_text: 'no readable text',
  extract_failed: 'could not be parsed',
}

/** One line on what a folder's scanner is doing or last did. */
export function folderStatus(folder, now = Date.now()) {
  if (folder.scanning) {
    const { done = 0, total = 0 } = folder.progress || {}
    return total ? `Checking ${done.toLocaleString()} of ${total.toLocaleString()} files…` : 'Checking the folder…'
  }
  if (folder.queued) return 'Waiting to check the folder…'
  if (!folder.lastScanAt) return 'Not checked yet.'
  const seconds = Math.max(0, Math.round(now / 1000 - folder.lastScanAt))
  const when = seconds < 60 ? 'just now' : seconds < 3600 ? `${Math.round(seconds / 60)} min ago` : `${Math.round(seconds / 3600)} h ago`
  return `Checked ${when}.`
}

/** What the last scan changed, in words, or '' when it changed nothing. */
export function describeChanges(changes) {
  if (!changes) return ''
  const parts = [
    [changes.added, 'added'],
    [changes.updated, 'updated'],
    [changes.removed, 'removed'],
  ].filter(([value]) => value > 0).map(([value, label]) => `${value.toLocaleString()} ${label}`)
  return parts.length ? `Last check: ${parts.join(', ')}.` : ''
}
