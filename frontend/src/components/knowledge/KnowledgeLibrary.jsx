import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Modal } from '../ui/Modal'
import { IconCollection, IconFile, IconPlus } from '../ui/icons'
import {
  addCollectionDocuments,
  createCollection,
  deleteCollection,
  DOCUMENT_ACCEPT,
  filesFromDrop,
  ingestLibraryFile,
  isLibraryDocument,
  listLibraryDocuments,
  removeCollectionDocument,
  renameCollection,
} from '../../lib/knowledgeCollections.js'
import { WatchedFolders } from './WatchedFolders.jsx'
import { getApiBase } from '../../lib/apiBase.js'
import '../../styles/project-context.css'
import '../../styles/knowledge.css'

const countLabel = (count, one, many) => `${count.toLocaleString()} ${count === 1 ? one : many}`
// A dropped folder is uploaded one file at a time; a larger tree belongs in a watched folder.
const MAX_DROPPED_DOCUMENTS = 500
const byFilename = (a, b) => a.filename.localeCompare(b.filename, undefined, { sensitivity: 'base' })

/* Manage collections and their documents. Opened from the chat, it can also
   turn searching a collection on or off for that chat. */
export function KnowledgeLibrary({ collections, refresh, initialCollectionId = null, searchedIds = null, onToggleSearch = null, onClose, busy = false }) {
  const apiBase = getApiBase()
  const [documents, setDocuments] = useState(null)
  const [selectedId, setSelectedId] = useState(initialCollectionId)
  const [newName, setNewName] = useState('')
  const [renameDraft, setRenameDraft] = useState(null)
  const [picked, setPicked] = useState(() => new Set())
  const [working, setWorking] = useState('')
  const [error, setError] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [dropping, setDropping] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null)
  const uploadRef = useRef(null)

  const loadDocuments = useCallback(async () => {
    try {
      const next = await listLibraryDocuments()
      if (getApiBase() === apiBase) setDocuments(next)
    } catch (failure) {
      if (getApiBase() === apiBase) setError(failure.message)
    }
  }, [apiBase])
  useEffect(() => { setDocuments(null); loadDocuments() }, [loadDocuments])
  const refreshAll = useCallback(() => Promise.all([refresh(), loadDocuments()]), [refresh, loadDocuments])

  const list = collections || []
  const selected = list.find(item => item.id === selectedId) || list[0] || null
  useEffect(() => { setPicked(new Set()); setRenameDraft(null); setConfirmDelete(false) }, [selected?.id])

  const documentsById = useMemo(() => new Map((documents || []).map(doc => [doc.id, doc])), [documents])
  const members = useMemo(() => (selected?.doc_ids || []).map(id => documentsById.get(id)).filter(Boolean).sort(byFilename), [selected, documentsById])
  const candidates = useMemo(() => (documents || []).filter(doc => !selected?.doc_ids.includes(doc.id)).sort(byFilename), [documents, selected])

  const run = async (label, action) => {
    setWorking(label)
    setError('')
    try { await action() } catch (failure) { setError(failure.message) } finally { setWorking('') }
  }

  const create = event => {
    event.preventDefault()
    const name = newName.trim()
    if (!name) return
    run('create', async () => {
      const created = await createCollection(name)
      await refresh()
      setSelectedId(created.id)
      setNewName('')
    })
  }
  const saveRename = event => {
    event.preventDefault()
    run('rename', async () => {
      await renameCollection(selected.id, renameDraft)
      await refresh()
      setRenameDraft(null)
    })
  }
  const addPicked = () => run('add', async () => {
    await addCollectionDocuments(selected.id, [...picked])
    await refresh()
    setPicked(new Set())
  })
  const remove = doc => run(`remove:${doc.id}`, async () => {
    await removeCollectionDocument(selected.id, doc.id)
    await refresh()
  })
  const upload = (items, skippedTypes = 0) => run('upload', async () => {
    const failures = []
    try {
      for (const [index, { file, name }] of items.entries()) {
        setUploadProgress({ done: index, total: items.length })
        try { await ingestLibraryFile(file, [selected.id], name) } catch (failure) { failures.push(failure.message) }
      }
    } finally {
      setUploadProgress(null)
    }
    await Promise.all([refresh(), loadDocuments()])
    if (skippedTypes) failures.push(`Skipped ${countLabel(skippedTypes, 'file', 'files')} of a type the library does not read.`)
    if (failures.length) throw new Error(failures.join(' '))
  })
  const pickFiles = files => upload(Array.from(files || []).map(file => ({ file, name: file.name })))
  const dropFiles = event => {
    event.preventDefault()
    setDropping(false)
    if (working) return
    filesFromDrop(event.dataTransfer)
      .then(found => {
        const readable = found.filter(item => isLibraryDocument(item.name))
        if (readable.length > MAX_DROPPED_DOCUMENTS) {
          setError(`The drop holds ${readable.length} documents. Drop at most ${MAX_DROPPED_DOCUMENTS} at a time, or watch the folder instead.`)
          return
        }
        if (found.length) upload(readable, found.length - readable.length)
      })
      .catch(failure => setError(failure?.message || 'Could not read the dropped files.'))
  }
  const dragOver = event => {
    if (!Array.from(event.dataTransfer?.types || []).includes('Files')) return
    event.preventDefault()
    setDropping(true)
  }
  const removeCollection = () => run('delete', async () => {
    setConfirmDelete(false)
    await deleteCollection(selected.id)
    const next = await refresh()
    setSelectedId(next?.[0]?.id || null)
  })

  const searched = Boolean(selected && searchedIds?.has(selected.id))
  const disabled = Boolean(working)

  return (
    <Modal open title="Knowledge library" onClose={onClose} labelledById="knowledge-library-title" className="context-modal knowledge-modal"
      footer={<button type="button" className="context-primary" onClick={onClose}>Done</button>}>
      <p className="context-muted">A collection groups documents from your library. A chat or project that uses a collection searches every document in it when you send a message, and cites the passages it uses.</p>
      <div className="knowledge-layout">
        <nav className="knowledge-collections" aria-label="Collections">
          {collections === null && <p className="context-muted">Loading collections…</p>}
          {collections !== null && !list.length && <p className="context-muted">No collections yet.</p>}
          <ul>{list.map(item => <li key={item.id}>
            <button type="button" className="knowledge-collection" aria-pressed={item.id === selected?.id} onClick={() => setSelectedId(item.id)}>
              <IconCollection size={15} />
              <span className="knowledge-collection__name">{item.name}</span>
              <small>{countLabel(item.doc_ids.length, 'doc', 'docs')}</small>
              {searchedIds?.has(item.id) && <span className="knowledge-badge">In this chat</span>}
            </button>
          </li>)}</ul>
          <form className="knowledge-create" onSubmit={create}>
            <input aria-label="New collection name" placeholder="New collection" maxLength={80} value={newName} disabled={disabled} onChange={event => setNewName(event.target.value)} />
            <button type="submit" className="cxturn__action" disabled={disabled || !newName.trim()} aria-label="Create collection"><IconPlus size={14} />Create</button>
          </form>
        </nav>
        {selected ? <section className={`knowledge-detail${dropping ? ' is-dropping' : ''}`} aria-label={`Collection ${selected.name}`}
          onDragOver={dragOver} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget)) setDropping(false) }} onDrop={dropFiles}>
          <div className="context-row knowledge-detail__head">
            {renameDraft === null
              ? <h3>{selected.name}</h3>
              : <form className="knowledge-rename" onSubmit={saveRename}>
                <input aria-label="Collection name" maxLength={80} autoFocus value={renameDraft} onChange={event => setRenameDraft(event.target.value)} />
                <button type="submit" className="cxturn__action" disabled={disabled || !renameDraft.trim()}>Save name</button>
                <button type="button" className="cxturn__action" onClick={() => setRenameDraft(null)}>Cancel</button>
              </form>}
            {renameDraft === null && <div className="knowledge-detail__actions">
              <button type="button" className="cxturn__action" disabled={disabled} onClick={() => setRenameDraft(selected.name)} aria-label={`Rename ${selected.name}`}>Rename</button>
              <button type="button" className="cxturn__action" disabled={disabled} onClick={() => setConfirmDelete(true)} aria-label={`Delete ${selected.name}`}>Delete</button>
            </div>}
          </div>
          {confirmDelete && <div className="knowledge-confirm" role="group" aria-label={`Confirm deleting ${selected.name}`}>
            <p>Delete {selected.name}? Its documents stay in your library, and the folders it watches stop being watched. Chats and projects that searched it show it as unavailable.</p>
            <button type="button" className="context-primary knowledge-danger" disabled={disabled} onClick={removeCollection}>Delete collection</button>
            <button type="button" className="cxturn__action" onClick={() => setConfirmDelete(false)}>Cancel</button>
          </div>}
          {onToggleSearch && <label className="context-check knowledge-search-toggle">
            <input type="checkbox" checked={searched} disabled={busy || disabled} onChange={event => run('toggle', async () => onToggleSearch(selected.id, event.target.checked))} />
            Search this collection in this chat
          </label>}
          <h4>{countLabel(selected.doc_ids.length, 'document', 'documents')}</h4>
          {documents === null && <p className="context-muted">Loading documents…</p>}
          {documents !== null && !members.length && <p className="context-muted">No documents yet. Add some from your library or upload new files.</p>}
          <ul className="knowledge-docs">{members.map(doc => <li key={doc.id}>
            <IconFile size={14} /><span className="knowledge-doc__name">{doc.filename}</span>
            <small>{countLabel(doc.chunk_count, 'chunk', 'chunks')}</small>
            <button type="button" className="cxturn__action" disabled={disabled} onClick={() => remove(doc)} aria-label={`Remove ${doc.filename} from ${selected.name}`}>Remove</button>
          </li>)}</ul>
          <div className="knowledge-add">
            <details open={documents !== null && !members.length && candidates.length > 0}>
              <summary>Add from library ({candidates.length.toLocaleString()})</summary>
              {candidates.length === 0 && <p className="context-muted">Every document in the library is already in this collection.</p>}
              <ul className="knowledge-pick">{candidates.map(doc => <li key={doc.id}>
                <label className="context-check">
                  <input type="checkbox" checked={picked.has(doc.id)} disabled={disabled} onChange={event => setPicked(current => {
                    const next = new Set(current)
                    if (event.target.checked) next.add(doc.id); else next.delete(doc.id)
                    return next
                  })} />
                  <span className="knowledge-doc__name">{doc.filename}</span><small>{countLabel(doc.chunk_count, 'chunk', 'chunks')}</small>
                </label>
              </li>)}</ul>
              {candidates.length > 0 && <button type="button" className="context-primary" disabled={disabled || !picked.size} onClick={addPicked}>
                {picked.size ? `Add ${countLabel(picked.size, 'document', 'documents')}` : 'Add documents'}
              </button>}
            </details>
            <button type="button" className="cxturn__action knowledge-upload" disabled={disabled} onClick={() => uploadRef.current?.click()}>
              {working === 'upload'
                ? (uploadProgress?.total > 1 ? `Indexing ${uploadProgress.done + 1} of ${uploadProgress.total}…` : 'Indexing…')
                : `Upload files into ${selected.name}`}
            </button>
            <input ref={uploadRef} type="file" multiple hidden accept={DOCUMENT_ACCEPT} aria-label={`Upload files into ${selected.name}`} onChange={event => { pickFiles(event.target.files); event.target.value = '' }} />
            <p className="context-muted knowledge-drop-hint">{dropping ? `Drop to add to ${selected.name}` : 'Or drop files or folders here.'}</p>
          </div>
          <WatchedFolders collection={selected} onChanged={refreshAll} disabled={disabled} />
        </section> : <section className="knowledge-detail knowledge-detail--empty">
          <p className="context-muted">Create a collection to group documents, then use it in a chat or a project.</p>
        </section>}
      </div>
      {error && <p role="alert" className="context-error">{error}</p>}
    </Modal>
  )
}
