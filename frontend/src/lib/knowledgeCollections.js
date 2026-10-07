/* Knowledge Library collections: named sets of library documents that a chat
   or a project searches when a message is sent. The server owns them; the chat
   and project context store only collection ids. */

import { apiFetch } from './apiRequest.js'
import { getApiBase } from './apiBase.js'

const JSON_HEADERS = { 'Content-Type': 'application/json' }

export const DOCUMENT_ACCEPT = '.pdf,.docx,.md,.txt,.csv,.json'

/* The types the server reads, the same list a watched folder picks up. */
const LIBRARY_EXTENSIONS = ['pdf', 'docx', 'md', 'txt', 'csv', 'json', 'rs', 'py', 'js']

export function isLibraryDocument(name) {
  const dot = String(name || '').lastIndexOf('.')
  return dot > 0 && LIBRARY_EXTENSIONS.includes(name.slice(dot + 1).toLowerCase())
}

const readAsBase64 = blob => new Promise((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = () => {
    const result = String(reader.result || '')
    const comma = result.indexOf(',')
    resolve(comma !== -1 ? result.slice(comma + 1) : result)
  }
  reader.onerror = () => reject(reader.error || new Error('Could not read the document.'))
  reader.readAsDataURL(blob)
})

/** Uploads one file into the library, optionally straight into collections.
    `name` defaults to the file's own; a file from a dropped folder passes its
    path inside that folder. */
export async function ingestLibraryFile(file, collectionIds = [], name = file.name, apiBase = getApiBase()) {
  const lowerName = name.toLowerCase()
  const isBinary = lowerName.endsWith('.pdf') || lowerName.endsWith('.docx')
  const response = await apiFetch('/api/documents/ingest', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({
      filename: name,
      content: isBinary ? await readAsBase64(file) : await file.text(),
      is_base64: isBinary,
      ...(collectionIds.length ? { collection_ids: collectionIds } : {}),
    }),
  }, apiBase)
  if (!response.ok) {
    const failure = await response.json().catch(() => null)
    throw new Error(failure?.error?.message || failure?.message || `Could not index ${name}.`)
  }
  return response.json()
}

const readEntries = reader => new Promise((resolve, reject) => reader.readEntries(resolve, reject))
const entryFile = entry => new Promise((resolve, reject) => entry.file(resolve, reject))

/** The files of a drop as `{ file, name }`, walking into dropped folders.
    `name` is the path inside the drop; hidden entries are left out. Call it
    from the drop handler itself: the browser forgets the items afterwards. */
export async function filesFromDrop(dataTransfer) {
  const entries = Array.from(dataTransfer?.items || [])
    .filter(item => item.kind === 'file')
    .map(item => item.webkitGetAsEntry?.())
    .filter(Boolean)
  if (!entries.length) return Array.from(dataTransfer?.files || []).map(file => ({ file, name: file.name }))
  const found = []
  const walk = async (entry, prefix) => {
    if (entry.name.startsWith('.')) return
    if (entry.isFile) {
      found.push({ file: await entryFile(entry), name: `${prefix}${entry.name}` })
    } else if (entry.isDirectory) {
      const reader = entry.createReader()
      for (let batch = await readEntries(reader); batch.length; batch = await readEntries(reader)) {
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`)
      }
    }
  }
  for (const entry of entries) await walk(entry, '')
  return found
}

async function request(path, options = {}, apiBase = getApiBase()) {
  const response = await apiFetch(path, options, apiBase)
  if (response.status === 204) return null
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new Error(body?.error?.message || `The library request failed (${response.status}).`)
  return body
}

const collectionPath = (id, rest = '') => `/api/collections/${encodeURIComponent(id)}${rest}`

export function normalizeCollection(value) {
  const id = typeof value?.id === 'string' ? value.id : ''
  const name = typeof value?.name === 'string' ? value.name : ''
  if (!id || !name) return null
  return {
    id,
    name,
    created_at: Number(value.created_at) || 0,
    doc_ids: Array.isArray(value.doc_ids) ? value.doc_ids.filter(docId => typeof docId === 'string') : [],
  }
}

export const normalizeCollections = value => (Array.isArray(value) ? value : []).map(normalizeCollection).filter(Boolean)

export const listCollections = async (apiBase = getApiBase()) => normalizeCollections(await request('/api/collections', {}, apiBase))

export const createCollection = async (name, apiBase = getApiBase()) => normalizeCollection(await request('/api/collections', {
  method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ name }),
}, apiBase))

export const renameCollection = async (id, name, apiBase = getApiBase()) => normalizeCollection(await request(collectionPath(id), {
  method: 'PATCH', headers: JSON_HEADERS, body: JSON.stringify({ name }),
}, apiBase))

export const deleteCollection = (id, apiBase = getApiBase()) => request(collectionPath(id), { method: 'DELETE' }, apiBase)

export const addCollectionDocuments = async (id, docIds, apiBase = getApiBase()) => normalizeCollection(await request(collectionPath(id, '/documents'), {
  method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ doc_ids: docIds }),
}, apiBase))

export const removeCollectionDocument = (id, docId, apiBase = getApiBase()) => request(collectionPath(id, `/documents/${encodeURIComponent(docId)}`), { method: 'DELETE' }, apiBase)

export async function listLibraryDocuments(apiBase = getApiBase()) {
  const documents = await request('/api/documents', {}, apiBase)
  return (Array.isArray(documents) ? documents : []).filter(doc => typeof doc?.id === 'string' && typeof doc?.filename === 'string')
}

/** Coverage for a set of documents from `GET /api/documents/index-status`, keyed by document id. */
export function documentsCoverage(docIds, coverageById = {}) {
  let indexable = 0
  let done = 0
  for (const docId of docIds) {
    const coverage = coverageById[docId]
    if (!coverage) continue
    indexable += coverage.indexable_chunks
    done += coverage.indexed_chunks + coverage.skipped_chunks
  }
  return { indexable, done, pending: done < indexable }
}
