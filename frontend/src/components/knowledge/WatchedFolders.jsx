import { useCallback, useEffect, useRef, useState } from 'react'
import { IconFolder } from '../ui/icons'
import { getApiBase } from '../../lib/apiBase.js'
import {
  browseFolders,
  describeChanges,
  folderStatus,
  listFolders,
  scanFolder,
  SKIP_REASONS,
  unwatchFolder,
  watchFolder,
} from '../../lib/knowledgeFolders.js'

const BUSY_POLL_MS = 1500
const IDLE_POLL_MS = 10000

const changedSomething = changes => Boolean(changes && (changes.added || changes.updated || changes.removed))

/* The folders a collection watches, and a form to watch another. The server
   keeps each folder's documents current; this polls to show what it is doing
   and calls `onChanged` when a poll shows a finished check that changed the
   library, so the collection's document list catches up. A check the server's
   timer runs usually starts and ends between two polls. */
export function WatchedFolders({ collection, onChanged, disabled = false, apiBase = getApiBase() }) {
  const currentBase = useRef(apiBase)
  currentBase.current = apiBase
  const [folders, setFolders] = useState(null)
  const [error, setError] = useState('')
  // Kept apart from `error` so the next successful poll clears a failed one
  // without hiding the result of an action.
  const [loadError, setLoadError] = useState('')
  const [working, setWorking] = useState('')
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState('')
  const [browser, setBrowser] = useState(null)
  const [confirmId, setConfirmId] = useState('')
  const seenChecks = useRef(null)

  const mine = (folders || []).filter(folder => folder.collectionId === collection.id)
  const busy = mine.some(folder => folder.scanning || folder.queued)

  const load = useCallback(async () => {
    try {
      const next = await listFolders(apiBase)
      if (currentBase.current !== apiBase) return
      setFolders(next)
      setLoadError('')
    } catch (failure) {
      if (currentBase.current === apiBase) setLoadError(failure.message)
    }
  }, [apiBase])
  useEffect(() => { setFolders(null); seenChecks.current = null; load() }, [load])
  useEffect(() => {
    const timer = window.setInterval(load, busy ? BUSY_POLL_MS : IDLE_POLL_MS)
    return () => window.clearInterval(timer)
  }, [busy, load])
  useEffect(() => {
    if (folders === null) return
    const watched = folders.filter(folder => folder.collectionId === collection.id)
    const previous = seenChecks.current?.collectionId === collection.id ? seenChecks.current.checks : null
    const signature = folder => `${folder.lastScanAt}|${folder.documents}|${JSON.stringify(folder.lastChanges)}`
    seenChecks.current = { collectionId: collection.id, checks: new Map(watched.map(folder => [folder.id, signature(folder)])) }
    if (!previous) return
    const changedLibrary = watched.some(folder => folder.lastScanAt && previous.get(folder.id) !== signature(folder)
      && changedSomething(folder.lastChanges))
    if (changedLibrary) onChanged?.()
  }, [folders, collection.id, onChanged])
  useEffect(() => { setAdding(false); setBrowser(null); setConfirmId(''); setError('') }, [collection.id])

  const run = async (label, action) => {
    setWorking(label)
    setError('')
    try { await action() } catch (failure) { setError(failure.message) } finally { setWorking('') }
  }
  const browse = path => run('browse', async () => {
    const view = await browseFolders(path, apiBase)
    setBrowser(view)
    if (view.path) setDraft(view.path)
  })
  const watch = event => {
    event.preventDefault()
    const path = draft.trim()
    if (!path) return
    run('watch', async () => {
      await watchFolder(path, collection.id, apiBase)
      setAdding(false)
      setBrowser(null)
      setDraft('')
      await load()
    })
  }
  const checkNow = folder => run(`scan:${folder.id}`, async () => {
    await scanFolder(folder.id, apiBase)
    await load()
  })
  const stopWatching = folder => run(`remove:${folder.id}`, async () => {
    setConfirmId('')
    await unwatchFolder(folder.id, apiBase)
    await load()
    onChanged?.()
  })

  const locked = disabled || Boolean(working)

  return (
    <div className="knowledge-folders">
      <h4>Watched folders</h4>
      {folders === null && !error && <p className="context-muted">Loading folders…</p>}
      {mine.length > 0 && <ul className="knowledge-folder-list">{mine.map(folder => <li key={folder.id} className="knowledge-folder">
        <div className="knowledge-folder__head">
          <IconFolder size={14} />
          <code className="knowledge-folder__path" title={folder.path}>{folder.path}</code>
        </div>
        <p className="knowledge-folder__status" aria-live="polite">
          {`${folder.documents.toLocaleString()} ${folder.documents === 1 ? 'document' : 'documents'}. ${folderStatus(folder)}`}
          {!folder.scanning && describeChanges(folder.lastChanges) ? ` ${describeChanges(folder.lastChanges)}` : ''}
        </p>
        {folder.lastError && <p className="knowledge-folder__error" role="status">{folder.lastError}</p>}
        {folder.skippedCount > 0 && <details className="knowledge-folder__skipped">
          <summary>{`${folder.skippedCount.toLocaleString()} ${folder.skippedCount === 1 ? 'file' : 'files'} skipped`}</summary>
          <ul>{folder.skipped.map(item => <li key={item.path}><span>{item.path}</span><small>{SKIP_REASONS[item.reason] || item.reason}</small></li>)}</ul>
          {folder.skippedCount > folder.skipped.length && <p className="context-muted">{`…and ${(folder.skippedCount - folder.skipped.length).toLocaleString()} more.`}</p>}
        </details>}
        <div className="knowledge-folder__actions">
          <button type="button" className="cxturn__action" disabled={locked || folder.scanning} onClick={() => checkNow(folder)} aria-label={`Check ${folder.path} now`}>Check now</button>
          <button type="button" className="cxturn__action" disabled={locked} onClick={() => setConfirmId(folder.id)} aria-label={`Stop watching ${folder.path}`}>Stop watching</button>
        </div>
        {confirmId === folder.id && <div className="knowledge-confirm" role="group" aria-label={`Confirm stopping ${folder.path}`}>
          <p>{`Stop watching ${folder.path}? Its ${folder.documents.toLocaleString()} ${folder.documents === 1 ? 'document leaves' : 'documents leave'} the library. The files on disk are not touched.`}</p>
          <button type="button" className="context-primary knowledge-danger" disabled={locked} onClick={() => stopWatching(folder)}>Stop watching</button>
          <button type="button" className="cxturn__action" onClick={() => setConfirmId('')}>Cancel</button>
        </div>}
      </li>)}</ul>}
      {!adding
        ? <button type="button" className="cxturn__action knowledge-folder-add" disabled={locked} onClick={() => setAdding(true)}>
          <IconFolder size={14} />Watch a folder
        </button>
        : <form className="knowledge-folder-form" onSubmit={watch}>
          <label htmlFor="knowledge-folder-path">Folder on this computer</label>
          <div className="knowledge-folder-form__row">
            <input id="knowledge-folder-path" value={draft} placeholder="Full path of a folder" disabled={locked} onChange={event => setDraft(event.target.value)} />
            <button type="button" className="cxturn__action" disabled={locked} onClick={() => browse(draft.trim() || null)}>Browse</button>
          </div>
          {browser && <div className="knowledge-browser" role="group" aria-label="Choose a folder">
            <div className="knowledge-browser__bar">
              <button type="button" className="cxturn__action" disabled={locked || (browser.parent === null && !(browser.hasRoots && browser.path !== null))} onClick={() => browse(browser.parent)}>Up</button>
              <code title={browser.path || ''}>{browser.path || 'This PC'}</code>
            </div>
            <ul>{browser.entries.map(entry => <li key={entry.path}>
              <button type="button" className="knowledge-browser__entry" disabled={locked} onClick={() => browse(entry.path)}><IconFolder size={14} /><span>{entry.name}</span></button>
            </li>)}</ul>
            {!browser.entries.length && <p className="context-muted">No folders inside.</p>}
            {browser.truncated && <p className="context-muted">{`Showing the first ${browser.entries.length} folders.`}</p>}
          </div>}
          <div className="knowledge-folder-form__row">
            <button type="submit" className="context-primary" disabled={locked || !draft.trim()}>{working === 'watch' ? 'Starting…' : `Watch into ${collection.name}`}</button>
            <button type="button" className="cxturn__action" onClick={() => { setAdding(false); setBrowser(null) }}>Cancel</button>
          </div>
          <p className="context-muted">Camelid reads the PDF, Word, Markdown, text, CSV, JSON and code files in the folder and its subfolders, then checks it for changes every 30 seconds while it runs. Hidden files and links are skipped.</p>
        </form>}
      {(error || loadError) && <p role="alert" className="context-error">{error || loadError}</p>}
    </div>
  )
}
