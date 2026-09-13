import { useCallback, useEffect, useRef, useState } from 'react'
import { changeRequest } from '../lib/changeReviews.js'
import { appStorage } from '../lib/appStorage.js'
import { downloadOutput, MAX_OUTPUT_BYTES, outputBytes, textOutput } from '../lib/outputFiles.js'
import { FolderPicker } from './WorkspaceView'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { IconReceipt, IconPlus } from '../components/ui/icons'
import '../styles/changes.css'

const STATUS = { pending: 'Needs approval', applied: 'Applied', rejected: 'Rejected', undone: 'Undone', conflict: 'Needs attention' }
export default function ChangesView({ apiBase, draft, onConsumeDraft }) {
  const [reviews, setReviews] = useState([])
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [showForm, setShowForm] = useState(Boolean(draft))
  const [workspace, setWorkspace] = useState(() => appStorage.getItem('camelid.workspacePath') || '')
  const [path, setPath] = useState(draft?.name || '')
  const [content, setContent] = useState(draft?.text || '')
  const [source, setSource] = useState(draft ? 'Generated output' : 'Manual proposal')
  const [mode, setMode] = useState('diff')
  const [filter, setFilter] = useState('all')
  const [folderPicker, setFolderPicker] = useState(false)
  const [forget, setForget] = useState(null)
  const epoch = useRef(0)
  const selection = useRef(0)
  const refresh = useCallback(async signal => {
    const at = epoch.current
    const data = await changeRequest(apiBase, '', { signal })
    if (!signal?.aborted && epoch.current === at) setReviews(data.reviews || [])
  }, [apiBase])
  useEffect(() => {
    const controller = new AbortController(); epoch.current++; selection.current++
    setSelected(null); setReviews([]); setError(''); setBusy(false)
    refresh(controller.signal).catch(e => { if (!controller.signal.aborted) setError(e.message) })
    return () => { controller.abort(); epoch.current++; selection.current++ }
  }, [refresh])
  useEffect(() => {
    if (!draft) return
    setPath(draft.name); setContent(draft.text); setSource('Generated output'); setShowForm(true); setSelected(null)
    onConsumeDraft?.()
  }, [draft, onConsumeDraft])
  const open = async id => {
    const requestId = ++selection.current
    try {
      setError(''); const data = await changeRequest(apiBase, '/' + id)
      if (requestId !== selection.current) return
      setSelected(data); setMode('diff'); setShowForm(false)
    } catch (e) { if (requestId === selection.current) setError(e.message) }
  }
  const mutate = async (suffix, body, method = 'POST') => {
    if (busy) return
    const at = epoch.current; selection.current++; setBusy(true); setError('')
    try {
      const data = await changeRequest(apiBase, suffix, { method, body })
      if (epoch.current !== at) return
      if (method === 'DELETE') setSelected(null)
      else { setSelected(data); setShowForm(false); setMode('diff') }
      await refresh()
    } catch (e) {
      if (epoch.current === at) {
        setError(e.message)
        // A lost response is not proof that a write failed. Reconcile the
        // saved receipt without ever repeating the mutation.
        if (selected?.id) {
          try {
            const latest = await changeRequest(apiBase, '/' + selected.id)
            if (epoch.current === at) {
              setSelected(latest)
              if (latest.status !== selected.status) setError(e.message + ' Current saved status: ' + (STATUS[latest.status] || latest.status) + '. No action was repeated.')
            }
          } catch { /* preserve the original error until the engine returns */ }
        }
        if (epoch.current === at) await refresh().catch(() => {})
      }
    }
    finally { if (epoch.current === at) setBusy(false) }
  }
  const prepare = e => {
    e.preventDefault()
    if (outputBytes(content) > MAX_OUTPUT_BYTES) { setError('File outputs are limited to 256 KB.'); return }
    mutate('', { workspace: workspace.trim(), path: path.trim(), content, source })
  }
  const visible = reviews.filter(r => filter === 'all' || r.status === filter)
  return <section className="cxv changes-view">
    <header className="cxv-head"><div className="cxv-head__copy"><p className="cxv-kicker"><IconReceipt size={14} /> File review</p><h1>Changes</h1><p className="cxv-sub">Review generated files before applying them, and undo changes with their saved originals.</p></div><Button icon={<IconPlus size={16} />} disabled={busy} onClick={() => { selection.current++; setShowForm(true); setSelected(null); setPath(''); setContent(''); setSource('Manual proposal') }}>New review</Button></header>
    {error && <p className="changes-error" role="alert">{error}</p>}
    {showForm && <form className="change-proposal cxv-card" onSubmit={prepare}>
      <h2>Prepare a file change</h2><p>Choose its destination and complete file contents. Preparing a review leaves the destination untouched.</p>
      <label>Workspace folder<div className="change-folder"><input required value={workspace} onChange={e => setWorkspace(e.target.value)} placeholder="Choose a local project folder" /><Button onClick={() => setFolderPicker(true)}>Browse</Button></div></label>
      <label>File path<input required value={path} onChange={e => setPath(e.target.value)} placeholder="src/example.js" maxLength={1024} /><small>Relative to the workspace. Its parent folder must already exist.</small></label>
      <label>Proposed file contents<textarea value={content} onChange={e => setContent(e.target.value)} spellCheck={false} rows={12} /></label>
      <div className="change-actions"><small>{outputBytes(content).toLocaleString()} / 262,144 bytes</small><Button type="submit" variant="primary" loading={busy}>Prepare review</Button><Button disabled={busy} onClick={() => setShowForm(false)}>Cancel</Button></div>
    </form>}
    <div className="changes-layout"><aside className="change-list" aria-label="File review history">
      <label>Show<select value={filter} onChange={e => setFilter(e.target.value)}><option value="all">All reviews ({reviews.length})</option><option value="pending">Needs approval</option><option value="applied">Applied</option><option value="undone">Undone</option><option value="rejected">Rejected</option></select></label>
      {!visible.length && <p>No reviews here yet. Use “Review file change” on a generated code block, or prepare a file above.</p>}
      {visible.map(r => <button type="button" key={r.id} disabled={busy} className="change-list-item" aria-pressed={selected?.id === r.id} onClick={() => open(r.id)}><strong>{r.path}</strong><span>{STATUS[r.status] || r.status}{r.created ? ' · New file' : ''}</span><small>{new Date(r.created_at).toLocaleString()}</small></button>)}
      <Button variant="ghost" disabled={busy} onClick={() => refresh().catch(e => setError(e.message))}>Refresh history</Button>
    </aside>
    {selected && !showForm ? <article className="change-detail cxv-card" aria-label="Selected file review">
      <header><span className={'change-status is-' + selected.status}>{STATUS[selected.status] || selected.status}</span><h2>{selected.path}</h2><p className="change-root">{selected.workspace}</p><small>{selected.source} · {selected.before_bytes.toLocaleString()} → {selected.after_bytes.toLocaleString()} bytes</small></header>
      <div className="change-actions"><div className="change-tabs" aria-label="File comparison">{['diff','before','after'].map(tab => <Button key={tab} aria-pressed={mode === tab} onClick={() => setMode(tab)}>{tab === 'diff' ? 'Diff summary' : tab === 'before' ? 'Before' : 'After'}</Button>)}</div><Button variant="ghost" onClick={() => downloadOutput(textOutput(selected.after, '', selected.path))}>Download proposed file</Button></div>
      {mode === 'diff' ? <><p className="change-caption">Changed lines are summarized below. Before and After contain the complete file contents.</p><pre className="change-diff">{selected.diff.split('\n').map((line,i) => <span key={i} className={line.startsWith('+ ') ? 'diff-add' : line.startsWith('- ') ? 'diff-remove' : ''}>{line + '\n'}</span>)}</pre></> : <pre className="change-source">{mode === 'before' ? selected.before === null ? '(New file — no previous contents)' : selected.before || '(Empty file)' : selected.after || '(Empty file)'}</pre>}
      {selected.status === 'pending' && <div className="change-decision"><p>{selected.created ? 'Approval creates this file.' : 'Approval replaces this file with the exact After contents.'} Its original state is saved for Undo.</p><div className="change-actions"><Button variant="primary" loading={busy} onClick={() => mutate('/' + selected.id + '/decision', { approved: true })}>Approve &amp; apply</Button><Button disabled={busy} onClick={() => mutate('/' + selected.id + '/decision', { approved: false })}>Reject</Button></div></div>}
      {selected.status === 'applied' && <div className="change-decision"><p>{selected.created ? 'Undo removes the file created by this change.' : 'Undo restores the original contents.'} If the file has changed since application, Camelid will refuse to overwrite it.</p><Button loading={busy} onClick={() => mutate('/' + selected.id + '/undo')}>Undo this change</Button></div>}
      {selected.status === 'conflict' && <p>The journal could not confirm the outcome after an interrupted operation. Inspect the current file and saved Before/After versions before preparing another review.</p>}
      <Button variant="ghost" disabled={busy} onClick={() => setForget(selected.id)}>Remove saved review</Button>
    </article> : !showForm && <div className="change-empty"><IconReceipt size={32} /><h2>Every file change starts with a review</h2><p>Select a review to inspect its before and after versions. File changes require explicit approval; external MCP actions keep their own approval flow.</p></div>}
    </div>
    {folderPicker && <FolderPicker apiBase={apiBase} initialPath={workspace || null} onClose={() => setFolderPicker(false)} onPick={value => { setWorkspace(value); setFolderPicker(false) }} />}
    <ConfirmDialog open={Boolean(forget)} title="Remove saved review?" detail="This removes the review history and its undo snapshot. It does not change the destination file." confirmLabel="Remove review" onCancel={() => setForget(null)} onConfirm={async () => { await mutate('/' + forget, undefined, 'DELETE'); setForget(null) }} />
  </section>
}
