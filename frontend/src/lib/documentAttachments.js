import { appStorage } from './appStorage.js'

export const ATTACHED_DOCUMENTS_STORAGE_KEY = 'camelid.attachedDocuments'

export function normalizeAttachedDocuments(value) {
  const documents = Array.isArray(value) ? value : []
  const unique = new Map()
  for (const document of documents) {
    const docId = typeof document?.doc_id === 'string' ? document.doc_id.trim() : ''
    const filename = typeof document?.filename === 'string' ? document.filename.trim() : ''
    if (!docId || !filename) continue
    unique.set(docId, {
      doc_id: docId,
      filename,
      chunk_count: Math.max(0, Number(document.chunk_count) || 0),
      byte_size: Math.max(0, Number(document.byte_size) || 0),
    })
  }
  return [...unique.values()]
}

export function readAttachedDocuments() {
  try {
    const stored = appStorage.getItem(ATTACHED_DOCUMENTS_STORAGE_KEY)
    return normalizeAttachedDocuments(stored ? JSON.parse(stored) : [])
  } catch {
    appStorage.removeItem(ATTACHED_DOCUMENTS_STORAGE_KEY)
    return []
  }
}

export function writeAttachedDocuments(value) {
  const documents = normalizeAttachedDocuments(value)
  if (documents.length) {
    appStorage.setItem(ATTACHED_DOCUMENTS_STORAGE_KEY, JSON.stringify(documents))
  } else {
    appStorage.removeItem(ATTACHED_DOCUMENTS_STORAGE_KEY)
  }
  return documents
}
