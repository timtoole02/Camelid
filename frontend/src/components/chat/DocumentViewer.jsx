import { useEffect, useState } from 'react'
import { Modal } from '../ui/Modal'
import { apiFetch } from '../../lib/apiRequest.js'
import { getApiBase } from '../../lib/apiBase.js'

const formatBytes = (bytes) => (bytes >= 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${bytes} bytes`)

/* The text is shown only after the server confirms it still hashes to the
   digest recorded at ingest -- the same bar a citation from it must clear. */
export function DocumentViewer({ document: doc, onClose }) {
  const apiBase = getApiBase()
  const [view, setView] = useState({ status: 'loading' })

  useEffect(() => {
    setView({ status: 'loading' })
    const controller = new AbortController()
    const refuse = (code, message) => {
      if (!controller.signal.aborted) setView({ status: 'refused', code, message })
    }
    apiFetch(`/api/documents/${encodeURIComponent(doc.doc_id)}/source`, { signal: controller.signal })
      .then(async (res) => {
        const payload = await res.json().catch(() => null)
        if (controller.signal.aborted) return
        if (res.ok && typeof payload?.text === 'string') {
          setView({ status: 'shown', data: payload, bytes: new TextEncoder().encode(payload.text).length })
        } else {
          refuse(
            payload?.error?.code || 'document_source_refused',
            payload?.error?.message || 'This document could not be opened.',
          )
        }
      })
      .catch(() => refuse('document_source_unreachable', 'The library could not be reached to open this document.'))
    return () => controller.abort()
  }, [doc.doc_id, apiBase])

  const text = view.status === 'shown' ? view.data.text : null
  return (
    <Modal open onClose={onClose} title={doc.filename} labelledById="document-viewer-title" size="lg" className="document-viewer">
      <div className="document-viewer__meta">
        {view.status === 'shown' ? (
          <span className="citation-modal__badge citation-modal__badge--verified" title={`sha256 ${view.data.doc_sha256}`}>
            Verified &middot; {formatBytes(view.bytes)}
          </span>
        ) : view.status === 'refused' ? (
          <span className="citation-modal__badge citation-modal__badge--refused">Refused</span>
        ) : (
          <span className="citation-modal__badge">Checking&hellip;</span>
        )}
        {doc.chunk_count > 0 && <span>{doc.chunk_count} chunks indexed</span>}
      </div>
      {view.status === 'shown' ? (
        <pre className="document-viewer__text" tabIndex={0} aria-label={`Text of ${doc.filename}`}>{text}</pre>
      ) : view.status === 'refused' ? (
        <div className="citation-modal__refusal">
          <p className="citation-modal__refusal-title">Document not shown</p>
          <p className="citation-modal__refusal-message">{view.message}</p>
          <p className="citation-modal__refusal-code">{view.code}</p>
        </div>
      ) : (
        <p className="document-viewer__pending">Checking this document against its recorded hash&hellip;</p>
      )}
    </Modal>
  )
}
