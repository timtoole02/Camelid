import { useEffect, useMemo, useRef, useState } from 'react'
import { clampText, formatDate } from '../lib/formatters'
import { copyText } from '../lib/markdown'
import { MAX_MEMORY_CHARS, MAX_PROMPT_MEMORIES, cleanMemoryText, memoriesForPrompt } from '../lib/memory.js'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { EmptyState } from '../components/ui/EmptyState'
import { IconMemory, IconPin, IconCopy, IconCheck, IconClose, IconEdit, IconTrash, IconChat, IconPlus, IconSearch, IconFile } from '../components/ui/icons'
import '../styles/memory.css'

export default function MemoryView({
  memories = [],
  memoryEnabled = false,
  setMemoryEnabled,
  addMemory,
  editMemory,
  setMemoryInUse,
  forgetMemory,
  forgetAllMemories,
  conversations = [],
  openMemorySource,
  notes = [],
  noteSearch,
  setNoteSearch,
  selectedConversation,
  latestAssistantMessage,
  saveReplyAsNote,
  createNote,
  updateNote,
  deleteNote,
  setTab,
}) {
  const [section, setSection] = useState('memories')
  const [draft, setDraft] = useState('')
  const [draftSource, setDraftSource] = useState({ kind: 'manual' })
  const draftRef = useRef(null)

  /* A saved note becomes a memory only when the user writes it as one. */
  const noteToMemory = (note) => {
    setDraft(cleanMemoryText(note.body || note.title))
    setDraftSource({ kind: 'note', note_title: note.title || '' })
    setSection('memories')
    window.requestAnimationFrame(() => draftRef.current?.focus())
  }

  const inUse = memories.filter((memory) => memory.enabled).length

  return (
    <section className="memory-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconMemory size={14} /> Memory</p>
          <h1>Memory</h1>
          <p className="cxv-sub">What you’ve chosen to keep about yourself for the model, and notes you’ve saved for you.</p>
        </div>
        <div className="cxv-stats">
          <div className="cxv-stat">
            <span>Memories</span>
            <strong>{memories.length}</strong>
            <small>{memoryEnabled ? `${inUse} in use` : 'Memory is off'}</small>
          </div>
          <div className="cxv-stat">
            <span>Saved notes</span>
            <strong>{notes.length}</strong>
            <small>Never sent to the model</small>
          </div>
        </div>
      </header>

      <div className="memory-tabs" role="tablist" aria-label="Memory sections">
        <button type="button" role="tab" id="memory-tab-memories" aria-controls="memory-panel-memories"
          aria-selected={section === 'memories'} className={section === 'memories' ? 'is-active' : ''} onClick={() => setSection('memories')}>
          Memories <span>{memories.length}</span>
        </button>
        <button type="button" role="tab" id="memory-tab-notes" aria-controls="memory-panel-notes"
          aria-selected={section === 'notes'} className={section === 'notes' ? 'is-active' : ''} onClick={() => setSection('notes')}>
          Saved notes <span>{notes.length}</span>
        </button>
      </div>

      {section === 'memories' ? (
        <div role="tabpanel" id="memory-panel-memories" aria-labelledby="memory-tab-memories">
          <MemoriesSection
            memories={memories}
            memoryEnabled={memoryEnabled}
            setMemoryEnabled={setMemoryEnabled}
            addMemory={addMemory}
            editMemory={editMemory}
            setMemoryInUse={setMemoryInUse}
            forgetMemory={forgetMemory}
            forgetAllMemories={forgetAllMemories}
            conversations={conversations}
            openMemorySource={openMemorySource}
            draft={draft}
            setDraft={setDraft}
            draftSource={draftSource}
            setDraftSource={setDraftSource}
            draftRef={draftRef}
          />
        </div>
      ) : (
        <div role="tabpanel" id="memory-panel-notes" aria-labelledby="memory-tab-notes">
          <NotesSection
            notes={notes}
            noteSearch={noteSearch}
            setNoteSearch={setNoteSearch}
            selectedConversation={selectedConversation}
            latestAssistantMessage={latestAssistantMessage}
            saveReplyAsNote={saveReplyAsNote}
            createNote={createNote}
            updateNote={updateNote}
            deleteNote={deleteNote}
            setTab={setTab}
            onMakeMemory={noteToMemory}
          />
        </div>
      )}
    </section>
  )
}

function sourceLabel(source, conversations) {
  if (source.kind === 'note') return { text: source.note_title ? `From your note “${clampText(source.note_title, 48)}”` : 'From a saved note' }
  if (source.kind !== 'chat') return { text: 'Added by you' }
  const title = clampText(source.conversation_title || 'Untitled chat', 48)
  const where = source.turn ? `, message ${source.turn}` : ''
  if (!conversations.some((conversation) => conversation.id === source.conversation_id)) {
    return { text: `From “${title}”${where} · that chat was deleted` }
  }
  return { text: `From “${title}”${where}`, link: true }
}

function MemoriesSection({
  memories, memoryEnabled, setMemoryEnabled, addMemory, editMemory, setMemoryInUse, forgetMemory, forgetAllMemories,
  conversations, openMemorySource, draft, setDraft, draftSource, setDraftSource, draftRef,
}) {
  const [search, setSearch] = useState('')
  const [editingId, setEditingId] = useState(null)
  const [editDraft, setEditDraft] = useState('')
  const [pendingForgetId, setPendingForgetId] = useState(null)
  const [confirmForgetAll, setConfirmForgetAll] = useState(false)

  const given = useMemo(() => new Set(memoriesForPrompt(memories).map((memory) => memory.id)), [memories])
  const inUseCount = memories.filter((memory) => memory.enabled).length
  const visible = useMemo(() => {
    const q = search.trim().toLowerCase()
    return q ? memories.filter((memory) => memory.text.toLowerCase().includes(q)) : memories
  }, [memories, search])

  const add = () => {
    if (!cleanMemoryText(draft)) return
    if (addMemory(draft, draftSource)) {
      setDraft('')
      setDraftSource({ kind: 'manual' })
    }
  }

  const saveEdit = (id) => {
    if (editMemory(id, editDraft)) setEditingId(null)
  }

  return (
    <>
      <div className={`cxv-card memory-status ${memoryEnabled ? 'is-on' : ''}`}>
        <label className="memory-switch">
          <span className="memory-switch__copy">
            <strong>Remember things about me</strong>
            <small>
              {memoryEnabled
                ? 'On. After a reply, the model may suggest something about you to keep, and the memories in use below are given to it in your chats. Any chat can turn memory off in its context.'
                : 'Off. Nothing is suggested, and nothing below is given to the model. Turn it on to get suggestions as you chat.'}
            </small>
          </span>
          <input type="checkbox" role="switch" aria-label="Remember things about me" checked={memoryEnabled} onChange={(event) => setMemoryEnabled(event.target.checked)} />
        </label>
      </div>

      <div className="cxv-card cxv-form memory-add">
        <div className="cxv-card__titles">
          <strong>Add a memory</strong>
          <span className="cxv-card__sub-plain">
            {draftSource.kind === 'note'
              ? 'From a saved note. Shorten it to the fact worth keeping.'
              : 'One fact about you, in a sentence: a preference, your work, how you like answers.'}
          </span>
        </div>
        <div className="memory-add__row">
          <input
            ref={draftRef}
            value={draft}
            maxLength={MAX_MEMORY_CHARS}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); add() } }}
            placeholder="For example: I prefer metric units"
            aria-label="Something to remember"
          />
          <Button variant="primary" icon={<IconPlus size={16} />} onClick={add} disabled={!cleanMemoryText(draft)}>Remember</Button>
          {draftSource.kind === 'note' && (
            <Button variant="ghost" onClick={() => { setDraft(''); setDraftSource({ kind: 'manual' }) }}>Cancel</Button>
          )}
        </div>
      </div>

      {memories.length > 0 && (
        <div className="cxv-toolbar">
          <label className="cxv-search">
            <IconSearch size={16} />
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search memories" aria-label="Search memories" />
          </label>
          <Button variant="ghost" className="cxv-danger" icon={<IconTrash size={15} />} onClick={() => setConfirmForgetAll(true)}>Forget everything</Button>
        </div>
      )}

      {memoryEnabled && inUseCount > given.size && (
        <p className="memory-note">
          Each chat gets the {given.size} most recently updated memories in use (at most {MAX_PROMPT_MEMORIES}, and 6,000 characters).
          The others are kept but not given to the model.
        </p>
      )}

      {visible.length === 0 ? (
        <EmptyState
          icon={<IconMemory size={26} />}
          title={memories.length === 0 ? 'Nothing remembered yet' : 'Nothing matched'}
          description={memories.length === 0
            ? (memoryEnabled
              ? 'As you chat, the model will suggest things worth remembering. You decide what’s kept.'
              : 'Turn memory on to get suggestions as you chat, or add something yourself above.')
            : 'No memory matches that search.'}
        />
      ) : (
        <ul className="memory-list">
          {visible.map((memory) => {
            const source = sourceLabel(memory.source, conversations)
            const editing = editingId === memory.id
            const confirming = pendingForgetId === memory.id
            const givenNow = memoryEnabled && given.has(memory.id)
            return (
              <li key={memory.id} className={`cxv-card memory-item ${memory.enabled ? '' : 'is-off'}`}>
                {editing ? (
                  <div className="memory-item__edit">
                    <input
                      value={editDraft}
                      maxLength={MAX_MEMORY_CHARS}
                      autoFocus
                      aria-label="Edit memory"
                      onChange={(event) => setEditDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') { event.preventDefault(); saveEdit(memory.id) }
                        if (event.key === 'Escape') { event.preventDefault(); setEditingId(null) }
                      }}
                    />
                    <Button variant="primary" size="sm" onClick={() => saveEdit(memory.id)} disabled={!cleanMemoryText(editDraft)}>Save</Button>
                    <Button variant="ghost" size="sm" onClick={() => setEditingId(null)}>Cancel</Button>
                  </div>
                ) : (
                  <p className="memory-item__text">{memory.text}</p>
                )}
                <div className="memory-item__meta">
                  {source.link
                    ? <button type="button" className="memory-item__source" onClick={() => openMemorySource(memory.source.conversation_id, memory.source.message_id)}>
                        <IconChat size={13} /> {source.text}
                      </button>
                    : <span className="memory-item__source is-static">{memory.source.kind === 'note' ? <IconFile size={13} /> : <IconMemory size={13} />} {source.text}</span>}
                  <span>Saved {formatDate(memory.created_at) || 'earlier'}{memory.edited ? ' · edited' : ''}</span>
                  {!memory.enabled && <span className="cxv-tag">Not in use</span>}
                  {memory.enabled && memoryEnabled && !givenNow && <span className="cxv-tag">Not sent: over the limit</span>}
                </div>
                {!editing && (
                  <div className="memory-item__actions">
                    <label className="memory-item__use">
                      <input type="checkbox" role="switch" checked={memory.enabled} onChange={(event) => setMemoryInUse(memory.id, event.target.checked)} aria-label={`Use: ${memory.text}`} />
                      Use
                    </label>
                    <Button variant="ghost" size="sm" icon={<IconEdit size={15} />} onClick={() => { setEditingId(memory.id); setEditDraft(memory.text); setPendingForgetId(null) }}>Edit</Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className={confirming ? 'cxv-danger is-armed' : 'cxv-danger'}
                      icon={<IconTrash size={15} />}
                      aria-label={confirming ? `Confirm: forget “${memory.text}”` : `Forget “${memory.text}”`}
                      onClick={() => {
                        if (!confirming) { setPendingForgetId(memory.id); return }
                        forgetMemory(memory.id)
                        setPendingForgetId(null)
                      }}
                    >
                      {confirming ? 'Forget' : ''}
                    </Button>
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      )}

      <ConfirmDialog
        open={confirmForgetAll}
        title="Forget everything?"
        detail={`This removes all ${memories.length} memor${memories.length === 1 ? 'y' : 'ies'} from this device. Your chats and saved notes stay.`}
        confirmLabel="Forget everything"
        onConfirm={() => { forgetAllMemories(); setConfirmForgetAll(false) }}
        onCancel={() => setConfirmForgetAll(false)}
      />
    </>
  )
}

function NotesSection({
  notes, noteSearch, setNoteSearch, selectedConversation, latestAssistantMessage, saveReplyAsNote,
  createNote, updateNote, deleteNote, setTab, onMakeMemory,
}) {
  const [scopeFilter, setScopeFilter] = useState('all')
  const [showPinnedOnly, setShowPinnedOnly] = useState(false)
  const [newNote, setNewNote] = useState({ title: '', scope: 'General', body: '' })
  const [editingId, setEditingId] = useState(null)
  const [editDraft, setEditDraft] = useState({ title: '', scope: '', body: '' })
  const [pendingDeleteId, setPendingDeleteId] = useState(null)
  const [busyAction, setBusyAction] = useState('')
  const [copyFeedback, setCopyFeedback] = useState(null)
  const copyResetRef = useRef(null)

  useEffect(() => () => { if (copyResetRef.current) window.clearTimeout(copyResetRef.current) }, [])

  const searchableNotes = useMemo(() => {
    if (!noteSearch.trim()) return notes
    const q = noteSearch.toLowerCase()
    return notes.filter((note) =>
      note.title.toLowerCase().includes(q)
      || note.body.toLowerCase().includes(q)
      || note.scope.toLowerCase().includes(q),
    )
  }, [notes, noteSearch])

  const availableScopes = useMemo(
    () => ['all', ...new Set(notes.map((note) => note.scope).filter(Boolean))],
    [notes],
  )

  const visibleNotes = useMemo(() => searchableNotes.filter((note) => {
    if (showPinnedOnly && !note.pinned) return false
    if (scopeFilter !== 'all' && note.scope !== scopeFilter) return false
    return true
  }), [scopeFilter, searchableNotes, showPinnedOnly])

  const latestChatLabel = clampText(selectedConversation?.title?.trim() || 'Current chat', 40)
  const canSaveLatestReply = Boolean(selectedConversation && latestAssistantMessage?.content)

  const handleCreateNote = async () => {
    if (busyAction) return
    setBusyAction('create')
    const saved = await createNote(newNote)
    if (saved) setNewNote({ title: '', scope: newNote.scope || 'General', body: '' })
    setBusyAction('')
  }

  const startEditing = (note) => {
    setEditingId(note.id)
    setEditDraft({ title: note.title, scope: note.scope, body: note.body })
    setPendingDeleteId(null)
  }

  const cancelEditing = () => {
    if (busyAction) return
    setEditingId(null)
    setEditDraft({ title: '', scope: '', body: '' })
  }

  const handleSaveEdit = async (noteId) => {
    if (busyAction) return
    setBusyAction(`edit:${noteId}`)
    const saved = await updateNote(noteId, editDraft)
    if (saved) cancelEditing()
    setBusyAction('')
  }

  const handleTogglePin = async (note) => {
    if (busyAction) return
    setBusyAction(`pin:${note.id}`)
    await updateNote(note.id, { pinned: !note.pinned }, { successMessage: note.pinned ? 'Note unpinned.' : 'Note pinned.' })
    setBusyAction('')
  }

  const handleCopy = async (note) => {
    const ok = await copyText(`${note.title}\n\n${note.body}`)
    setCopyFeedback({ id: note.id, ok })
    if (copyResetRef.current) window.clearTimeout(copyResetRef.current)
    copyResetRef.current = window.setTimeout(() => setCopyFeedback(null), 1600)
  }

  const handleDelete = async (noteId) => {
    if (busyAction) return
    if (pendingDeleteId !== noteId) { setPendingDeleteId(noteId); return }
    setBusyAction(`delete:${noteId}`)
    const deleted = await deleteNote(noteId, { successMessage: 'Note deleted.' })
    if (deleted) {
      if (editingId === noteId) cancelEditing()
      setPendingDeleteId(null)
    }
    setBusyAction('')
  }

  return (
    <>
      <p className="memory-note">Notes are for you to keep. They are never given to the model; to have it know something, add it as a memory.</p>

      <div className="cxv-grid cxv-grid--two">
        <div className="cxv-card cxv-form">
          <div className="cxv-card__titles">
            <strong>Write a note</strong>
            <span className="cxv-card__sub-plain">A decision, a snippet, anything worth keeping</span>
          </div>
          <input value={newNote.title} onChange={(e) => setNewNote((c) => ({ ...c, title: e.target.value }))} placeholder="Short title" />
          <input value={newNote.scope} onChange={(e) => setNewNote((c) => ({ ...c, scope: e.target.value }))} placeholder="Scope (e.g. General, Project)" />
          <textarea value={newNote.body} onChange={(e) => setNewNote((c) => ({ ...c, body: e.target.value }))} placeholder="Note" rows={4} />
          <div className="cxv-form__actions">
            <Button variant="primary" icon={<IconPlus size={16} />} onClick={handleCreateNote} loading={busyAction === 'create'}>Save note</Button>
          </div>
        </div>

        <div className="cxv-card cxv-form">
          <div className="cxv-card__titles">
            <strong>From this chat</strong>
            <span className="cxv-card__sub-plain">Keep a useful reply as a note</span>
          </div>
          <div className="cxv-mem__context">
            <strong title={selectedConversation?.title || 'No chat selected'}>{selectedConversation ? latestChatLabel : 'No chat selected'}</strong>
            <span>{canSaveLatestReply ? 'Latest assistant reply is ready to save.' : 'Open a conversation with an assistant reply to save it here.'}</span>
          </div>
          <div className="cxv-form__actions">
            <Button variant="ghost" icon={<IconChat size={16} />} onClick={() => setTab('chat')}>Open chat</Button>
            <Button variant="primary" onClick={saveReplyAsNote} disabled={!canSaveLatestReply}>Save latest reply</Button>
          </div>
        </div>
      </div>

      <div className="cxv-toolbar">
        <label className="cxv-search">
          <IconSearch size={16} />
          <input value={noteSearch} onChange={(e) => setNoteSearch(e.target.value)} placeholder="Search notes" aria-label="Search notes" />
        </label>
        <select value={scopeFilter} onChange={(e) => setScopeFilter(e.target.value)} aria-label="Note scope">
          {availableScopes.map((scope) => (
            <option key={scope} value={scope}>{scope === 'all' ? 'All scopes' : scope}</option>
          ))}
        </select>
        <Button variant="ghost" icon={<IconPin size={15} />} className={showPinnedOnly ? 'is-active' : ''} onClick={() => setShowPinnedOnly((c) => !c)}>
          Pinned only
        </Button>
      </div>

      {visibleNotes.length === 0 ? (
        <EmptyState
          icon={<IconFile size={26} />}
          title={notes.length === 0 ? 'No notes yet' : 'Nothing matched'}
          description={notes.length === 0
            ? 'Write one above, or save a useful reply from a chat.'
            : 'No notes matched those filters. Try a broader search or clear the pinned / scope filter.'}
        />
      ) : (
        <div className="cxv-grid">
          {visibleNotes.map((note) => {
            const isEditing = editingId === note.id
            const isDeleting = busyAction === `delete:${note.id}`
            const isPinning = busyAction === `pin:${note.id}`
            const isSavingEdit = busyAction === `edit:${note.id}`
            const confirming = pendingDeleteId === note.id
            const copyStatus = copyFeedback?.id === note.id ? copyFeedback : null

            return (
              <article key={note.id} className={`cxv-card cxv-mem ${note.pinned ? 'is-pinned' : ''}`}>
                <div className="cxv-card__head">
                  <div className="cxv-card__titles">
                    <strong>{note.title}</strong>
                    <div className="cxv-mem__meta">
                      <span className="cxv-tag">{note.scope}</span>
                      {note.pinned && <span className="cxv-tag cxv-tag--accent"><IconPin size={11} /> Pinned</span>}
                    </div>
                  </div>
                  <div className="cxv-card__actions">
                    <Button variant="ghost" size="sm" icon={<IconPin size={15} />} aria-label={note.pinned ? 'Unpin' : 'Pin'} className={note.pinned ? 'is-active' : ''} loading={isPinning} onClick={() => handleTogglePin(note)} disabled={Boolean(busyAction)} />
                    <Button
                      variant="ghost"
                      size="sm"
                      icon={copyStatus ? (copyStatus.ok ? <IconCheck size={15} /> : <IconClose size={15} />) : <IconCopy size={15} />}
                      aria-label={copyStatus ? (copyStatus.ok ? 'Copied' : 'Copy failed') : 'Copy'}
                      onClick={() => handleCopy(note)}
                    />
                  </div>
                </div>

                {isEditing ? (
                  <div className="cxv-edit">
                    <input value={editDraft.title} onChange={(e) => setEditDraft((c) => ({ ...c, title: e.target.value }))} placeholder="Title" />
                    <input value={editDraft.scope} onChange={(e) => setEditDraft((c) => ({ ...c, scope: e.target.value }))} placeholder="Scope" />
                    <textarea value={editDraft.body} onChange={(e) => setEditDraft((c) => ({ ...c, body: e.target.value }))} rows={5} placeholder="Note" />
                    <div className="cxv-form__actions">
                      <Button variant="primary" onClick={() => handleSaveEdit(note.id)} loading={isSavingEdit}>Save changes</Button>
                      <Button variant="ghost" onClick={cancelEditing} disabled={isSavingEdit}>Cancel</Button>
                    </div>
                  </div>
                ) : (
                  <>
                    <p className="cxv-mem__body">{note.body}</p>
                    <footer className="cxv-card__foot">
                      <span className="cxv-card__meta">Updated {formatDate(note.updated_at) || 'Unknown'}</span>
                      <div className="cxv-card__actions">
                        <Button variant="ghost" size="sm" icon={<IconMemory size={15} />} onClick={() => onMakeMemory(note)} disabled={Boolean(busyAction)}>Make a memory</Button>
                        <Button variant="ghost" size="sm" icon={<IconEdit size={15} />} onClick={() => startEditing(note)} disabled={Boolean(busyAction)}>Edit</Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className={confirming ? 'cxv-danger is-armed' : 'cxv-danger'}
                          icon={<IconTrash size={15} />}
                          aria-label={confirming ? 'Confirm delete' : 'Delete note'}
                          loading={isDeleting}
                          onClick={() => handleDelete(note.id)}
                          disabled={isDeleting || (Boolean(busyAction) && !confirming)}
                        >
                          {confirming ? 'Confirm' : ''}
                        </Button>
                      </div>
                    </footer>
                  </>
                )}
              </article>
            )
          })}
        </div>
      )}
    </>
  )
}
