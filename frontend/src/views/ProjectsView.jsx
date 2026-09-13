import { useCallback, useState } from 'react'
import { Modal } from '../components/ui/Modal'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { IconFolder } from '../components/ui/icons'
import { ReferenceEditor } from '../components/context/ContextEditors'
import { MAX_INSTRUCTION_CHARS, MAX_PROJECTS } from '../lib/projectContext.js'

function ProjectEditor({ project, onSave, onClose, busy }) {
  const [draft, setDraft] = useState(() => project || { name: '', instructions: '', references: [] })
  const [error, setError] = useState('')
  const [readingFiles, setReadingFiles] = useState(false)
  const patch = values => setDraft(current => ({ ...current, ...values }))
  const save = () => {
    try { onSave(draft); onClose() } catch (failure) { setError(failure.message) }
  }
  return <Modal open title={project ? 'Edit project' : 'New project'} onClose={onClose} labelledById="project-editor-title" className="context-modal"
    footer={<><button type="button" className="cxturn__action" onClick={onClose}>Cancel</button><button type="button" className="context-primary" disabled={busy || readingFiles || !draft.name.trim()} onClick={save}>Save project</button></>}>
    <div className="context-form"><fieldset disabled={busy}>
      <label className="context-field">Project name<input autoFocus aria-label="Project name" maxLength={80} value={draft.name} onChange={event => patch({ name: event.target.value })} placeholder="Website redesign" /></label>
      <label className="context-field">Project instructions<textarea aria-label="Project instructions" maxLength={MAX_INSTRUCTION_CHARS} rows={5} value={draft.instructions} onChange={event => patch({ instructions: event.target.value })} placeholder="Describe the project, its goals, and preferences shared by its conversations…" /></label>
      <ReferenceEditor references={draft.references} disabled={busy} onReadingChange={setReadingFiles} onChange={references => patch({ references })} />
    </fieldset>
    <p className="context-muted">Saved on this device. Future messages in this project's chats use the latest instructions and files, unless excluded in conversation context.</p>
    {error && <p role="alert" className="context-error">{error}</p>}</div>
  </Modal>
}

export default function ProjectsView({ projects, conversations, onSave, onDelete, onNewChat, onOpenConversation, busy }) {
  const [editing, setEditing] = useState(null)
  const [deleting, setDeleting] = useState(null)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const closeEditor = useCallback(() => setEditing(null), [])
  const filtered = projects.filter(project => `${project.name} ${project.instructions}`.toLowerCase().includes(search.toLowerCase()))
  return <section className="cxv projects-view" aria-label="Projects">
    <header className="context-page-head"><div><h2>Projects</h2><p>Keep related chats together with shared instructions and reference files.</p></div>
      <button type="button" className="context-primary" disabled={busy || projects.length >= MAX_PROJECTS} onClick={() => setEditing({})}>New project</button></header>
    {projects.length > 0 && <input className="context-search" aria-label="Search projects" placeholder="Search projects…" value={search} onChange={event => setSearch(event.target.value)} />}
    {!projects.length && <div className="context-empty"><IconFolder size={30} /><h3>A starting point for every conversation</h3><p>Add the brief, style guide, or background you usually repeat. Choose the project when you start a chat.</p></div>}
    {projects.length > 0 && !filtered.length && <p className="context-muted">No projects match your search.</p>}
    <div className="project-grid">{filtered.map(project => {
      const chats = conversations.filter(conversation => conversation.context?.project_id === project.id).sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)))
      return <article key={project.id} className="project-card">
        <div className="context-row"><h3><IconFolder size={19} />{project.name}</h3><span className="context-muted">{chats.length} chats</span></div>
        <p className="project-description">{project.instructions || 'No shared instructions yet.'}</p>
        <p className="context-muted">{project.references.length} reference {project.references.length === 1 ? 'file' : 'files'}</p>
        <div className="context-row project-actions"><button type="button" className="context-primary" disabled={busy} onClick={() => onNewChat(project.id)} aria-label={`New chat in ${project.name}`}>New chat</button>
          <button type="button" className="cxturn__action" disabled={busy} onClick={() => setEditing(project)} aria-label={`Edit ${project.name}`}>Edit</button>
          <button type="button" className="cxturn__action" disabled={busy} onClick={() => { setError(''); setDeleting(project) }} aria-label={`Delete ${project.name}`}>Delete</button></div>
        {chats.length > 0 && <details className="project-chats"><summary>Conversations ({chats.length})</summary><ul>{chats.map(chat => <li key={chat.id}><button type="button" onClick={() => onOpenConversation(chat.id)}>{chat.title}{chat.archived_at ? ' · Archived' : ''}</button></li>)}</ul></details>}
      </article>
    })}</div>
    {editing && <ProjectEditor project={editing.id ? editing : null} onSave={onSave} onClose={closeEditor} busy={busy} />}
    <ConfirmDialog open={Boolean(deleting)} title={`Delete ${deleting?.name || 'project'}?`} detail="This removes the shared instructions and files. Its conversations remain available in Chat history and will show that their project is unavailable." confirmLabel="Delete project" onCancel={() => setDeleting(null)} onConfirm={() => {
      try { onDelete(deleting.id); setDeleting(null) } catch (failure) { setError(failure.message); setDeleting(null) }
    }} />
    {error && <p role="alert" className="context-error">{error}</p>}
  </section>
}
