import { useCallback, useRef, useState } from 'react'
import { Modal } from '../ui/Modal'
import { IconFolder, IconFile } from '../ui/icons'
import { buildContextSources, contextBytes, estimateContextTokens, MAX_INSTRUCTION_CHARS, MAX_REFERENCES, normalizeChatContext, readContextFile, validateContextDraft, withCollection, withoutCollection } from '../../lib/projectContext.js'
import { CollectionChecklist } from '../knowledge/CollectionChecklist'
import '../../styles/project-context.css'

export function ReferenceEditor({ references, onChange, disabled = false, onReadingChange = () => {} }) {
  const [error, setError] = useState('')
  const [reading, setReading] = useState(false)
  const inputRef = useRef(null)
  const addFiles = async event => {
    const files = [...event.target.files]
    event.target.value = ''
    setError('')
    setReading(true)
    onReadingChange(true)
    try {
      if (references.length + files.length > MAX_REFERENCES) throw new Error(`Keep at most ${MAX_REFERENCES} reference files.`)
      const added = await Promise.all(files.map(readContextFile))
      const next = [...references, ...added]
      validateContextDraft({ references: next })
      onChange(next)
    } catch (failure) { setError(failure.message) }
    finally { setReading(false); onReadingChange(false) }
  }
  return <section className="context-references" aria-label="Reference files">
    <div className="context-row"><h3>Reference files <span className="context-muted">{references.length}/{MAX_REFERENCES}</span></h3>
      <button type="button" className="cxturn__action" disabled={disabled || reading || references.length >= MAX_REFERENCES} onClick={() => inputRef.current?.click()}>{reading ? 'Reading…' : 'Add files'}</button>
      <input ref={inputRef} type="file" multiple hidden aria-label="Add reference files" disabled={disabled || reading} onChange={addFiles} />
    </div>
    <p className="context-muted">Text, Markdown, code, CSV, or JSON · up to 32 KB each. A copy is saved here and included in full; files do not update automatically.</p>
    {references.map(reference => <details key={reference.id} className="context-reference">
      <summary><IconFile size={15} /><span>{reference.name}</span><small>{Math.ceil(contextBytes(reference.content) / 1024)} KB</small></summary>
      <pre>{reference.content}</pre>
      <button type="button" className="cxturn__action" disabled={disabled || reading} onClick={() => onChange(references.filter(item => item.id !== reference.id))} aria-label={`Remove ${reference.name}`}>Remove file</button>
    </details>)}
    {error && <p role="alert" className="context-error">{error}</p>}
  </section>
}

export function ContextSourceList({ sources }) {
  const tokens = estimateContextTokens(sources)
  return <section className="context-sources" aria-label="Included context">
    <h3>Included in the next request</h3>
    <p className="context-muted">{sources.length} sources · ~{tokens.toLocaleString()} tokens, estimated. Your conversation history and next message are added separately.</p>
    {sources.map((source, index) => <details className="context-reference" key={`${source.id}-${index}`}>
      <summary><span>{source.label}</span><small>{source.id === 'memory' ? 'Memory' : source.role === 'system' ? 'Instructions' : 'Reference'}</small></summary>
      <pre>{source.content}</pre>
    </details>)}
    {!sources.length && <p className="context-muted">No extra instructions or reference files.</p>}
  </section>
}

function ContextDialog({ context, projects, globalPrompt, automaticCodePrompt, onSave, onClose, busy, collections, memories = null }) {
  const [draft, setDraft] = useState(() => normalizeChatContext(context))
  const [readingFiles, setReadingFiles] = useState(false)
  const [error, setError] = useState('')
  const project = projects.find(item => item.id === draft.project_id)
  const patch = values => setDraft(current => ({ ...current, ...values }))
  const sources = buildContextSources({ context: draft, projects, globalPrompt, codePrompt: automaticCodePrompt, memories })
  const save = () => {
    try { onSave(draft); onClose() } catch (failure) { setError(failure.message) }
  }
  return <Modal open onClose={onClose} title="Conversation context" labelledById="conversation-context-title" className="context-modal"
    footer={<><button type="button" className="cxturn__action" onClick={onClose}>Cancel</button><button type="button" className="context-primary" disabled={busy || readingFiles} onClick={save}>Save context</button></>}>
    <div className="context-form">
      <p className="context-muted">Saved on this device. Changes apply to future messages; previous replies stay as they are.</p>
      <fieldset disabled={busy}>
        <label className="context-field">Project<select aria-label="Conversation project" value={draft.project_id} onChange={event => patch({ project_id: event.target.value, excluded_reference_ids: [], excluded_collection_ids: [] })}>
          <option value="">No project</option>
          {draft.project_id && !project && <option value={draft.project_id}>Project unavailable</option>}
          {projects.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select></label>
        {draft.project_id && !project && <p role="status" className="context-error">This project was removed. Its instructions and files are no longer included. Select another project or choose No project.</p>}
        <label className="context-check"><input type="checkbox" checked={draft.use_global_instructions} onChange={event => patch({ use_global_instructions: event.target.checked })} />Use global instructions</label>
        {project && <><label className="context-check"><input type="checkbox" checked={draft.use_project_instructions} onChange={event => patch({ use_project_instructions: event.target.checked })} />Use project instructions</label>
          {project.references.length > 0 && <div className="context-inheritance"><h3>Project reference files</h3>{project.references.map(reference => <label className="context-check" key={reference.id}>
            <input type="checkbox" checked={!draft.excluded_reference_ids.includes(reference.id)} onChange={event => patch({ excluded_reference_ids: event.target.checked
              ? draft.excluded_reference_ids.filter(id => id !== reference.id) : [...draft.excluded_reference_ids, reference.id] })} />{reference.name}
          </label>)}</div>}
        </>}
        <label className="context-field">Conversation instructions<textarea aria-label="Conversation instructions" rows={4} maxLength={MAX_INSTRUCTION_CHARS} value={draft.instructions} onChange={event => patch({ instructions: event.target.value })} placeholder="Goals, constraints, tone, or decisions to keep in mind for this chat…" /></label>
        <p className="context-muted">Conversation instructions take precedence over project and global preferences. Turn off inheritance above to replace those instructions.</p>
        {memories && <>
          <label className="context-check"><input type="checkbox" checked={draft.use_memory} onChange={event => patch({ use_memory: event.target.checked })} />Use memory in this chat</label>
          <p className="context-muted">Memory is what you chose to keep about yourself. With this off, the chat neither uses it nor suggests anything new to remember.</p>
        </>}
        <ReferenceEditor references={draft.references} disabled={busy} onReadingChange={setReadingFiles} onChange={references => patch({ references })} />
        {collections !== undefined && <section className="context-collections" aria-label="Knowledge collections">
          <h3>Knowledge collections</h3>
          <p className="context-muted">Searched when you send a message. Passages they supply are cited in the reply.</p>
          {collections === null ? <p className="context-muted">Loading collections…</p> : <>
            {project?.collection_ids.length > 0 && <div className="context-inheritance"><h3>From {project.name}</h3>
              <CollectionChecklist collections={collections.filter(item => project.collection_ids.includes(item.id))}
                checkedIds={project.collection_ids.filter(id => !draft.excluded_collection_ids.includes(id))}
                onToggle={(id, on) => patch({ excluded_collection_ids: (on ? withCollection : withoutCollection)(draft, projects, id).excluded_collection_ids })} />
            </div>}
            <CollectionChecklist collections={collections} excludeIds={project?.collection_ids || []} checkedIds={draft.collection_ids}
              onToggle={(id, on) => patch({ collection_ids: on ? [...draft.collection_ids, id] : draft.collection_ids.filter(item => item !== id) })} />
            {!collections.length && <p className="context-muted">No collections yet. Create one from Attach → Collections.</p>}
          </>}
        </section>}
      </fieldset>
      <ContextSourceList sources={sources} />
      <p className="context-muted">Context is sent to the selected chat endpoint. Conversation exports contain the transcript only; project and conversation context are kept on this device.</p>
      {error && <p className="context-error" role="alert">{error}</p>}
    </div>
  </Modal>
}

export function ConversationContext({ context, projects, sources, globalPrompt, onSave, onManageProjects, busy, compact = false, collections, memories = null }) {
  const [open, setOpen] = useState(false)
  const close = useCallback(() => setOpen(false), [])
  const project = projects.find(item => item.id === context.project_id)
  const tokens = estimateContextTokens(sources)
  return <div className={`conversation-context ${compact ? 'conversation-context--compact' : ''}`}>
    <button type="button" className="context-trigger" onClick={() => setOpen(true)} aria-label="Edit conversation context">
      <IconFolder size={16} /><strong>{compact ? 'Context' : project?.name || (context.project_id ? 'Project unavailable' : 'Conversation context')}</strong>
      <span>{sources.length ? `${sources.length} sources · ~${tokens.toLocaleString()} tokens` : 'Add instructions or files'}</span>
    </button>
    <button type="button" className="cxturn__action" onClick={onManageProjects}>Projects</button>
    {open && <ContextDialog key={context.project_id} context={context} projects={projects} globalPrompt={globalPrompt} automaticCodePrompt={sources.find(source => source.id === 'automatic-code')?.content || ''} onSave={onSave} onClose={close} busy={busy} collections={collections} memories={memories} />}
  </div>
}
