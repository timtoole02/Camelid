/* Knowledge Library collections: named sets of library documents that a chat
   or a project searches when a message is sent. The server owns them; the chat
   and project context store only collection ids. */

const JSON_HEADERS = { 'Content-Type': 'application/json' }

export const DOCUMENT_ACCEPT = '.pdf,.docx,.md,.txt,.csv,.json'

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

/** Uploads one file into the library, optionally straight into collections. */
export async function ingestLibraryFile(file, collectionIds = []) {
  const lowerName = file.name.toLowerCase()
  const isBinary = lowerName.endsWith('.pdf') || lowerName.endsWith('.docx')
  const response = await fetch('/api/documents/ingest', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({
      filename: file.name,
      content: isBinary ? await readAsBase64(file) : await file.text(),
      is_base64: isBinary,
      ...(collectionIds.length ? { collection_ids: collectionIds } : {}),
    }),
  })
  if (!response.ok) {
    const failure = await response.json().catch(() => null)
    throw new Error(failure?.error?.message || failure?.message || `Could not index ${file.name}.`)
  }
  return response.json()
}

async function request(path, options = {}) {
  const response = await fetch(path, options)
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

export const listCollections = async () => normalizeCollections(await request('/api/collections'))

export const createCollection = async name => normalizeCollection(await request('/api/collections', {
  method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ name }),
}))

export const renameCollection = async (id, name) => normalizeCollection(await request(collectionPath(id), {
  method: 'PATCH', headers: JSON_HEADERS, body: JSON.stringify({ name }),
}))

export const deleteCollection = id => request(collectionPath(id), { method: 'DELETE' })

export const addCollectionDocuments = async (id, docIds) => normalizeCollection(await request(collectionPath(id, '/documents'), {
  method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ doc_ids: docIds }),
}))

export const removeCollectionDocument = (id, docId) => request(collectionPath(id, `/documents/${encodeURIComponent(docId)}`), { method: 'DELETE' })

export async function listLibraryDocuments() {
  const documents = await request('/api/documents')
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
