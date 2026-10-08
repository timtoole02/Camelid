import { useId, useState } from 'react'
import { MAX_MEMORY_CHARS } from '../../lib/memory.js'
import { IconCheck, IconMemory } from '../ui/icons'
import '../../styles/memory.css'

/* Shown instead of asking on its own when this model is slow to ask: the
   request reads a few hundred tokens that cannot be cut short once started. */
export function MemoryOffer({ looking, onLook, disabled = false }) {
  if (looking) {
    return (
      <div className="memsug memsug--offer" role="status">
        <IconMemory size={14} />
        <span>Looking for things to remember… Sending a message stops this.</span>
      </div>
    )
  }
  return (
    <div className="memsug memsug--offer">
      <IconMemory size={14} />
      <button type="button" className="memsug__offer" onClick={onLook} disabled={disabled}>Look for things to remember</button>
      <span>This model is slow to ask, so it waits for you.</span>
    </div>
  )
}

/* Facts the model noticed in the user's message, offered under the reply.
   Nothing is kept until the user chooses Remember. */
export function MemorySuggestions({ suggestions, memories, onAccept, onDismiss, onUndo, onOpenMemory = null, disabled = false }) {
  const savedIds = new Set((memories || []).map((memory) => memory.id))
  // A saved suggestion whose memory was since forgotten has nothing left to show.
  const visible = (suggestions || []).filter((suggestion) => suggestion.status === 'pending'
    || (suggestion.status === 'saved' && savedIds.has(suggestion.memory_id)))
  if (!visible.length) return null
  const pending = visible.some((suggestion) => suggestion.status === 'pending')
  return (
    <section className="memsug" aria-label="Memory suggestions">
      <header className="memsug__head">
        <IconMemory size={14} />
        <strong>{pending ? 'Remember this?' : 'Remembered'}</strong>
        <span>{pending ? 'Kept only if you choose.' : 'You can change or forget it any time.'}</span>
        {onOpenMemory && <button type="button" className="memsug__link" onClick={onOpenMemory}>Memory</button>}
      </header>
      <ul className="memsug__list">
        {visible.map((suggestion) => (
          <SuggestionRow
            key={suggestion.id}
            suggestion={suggestion}
            disabled={disabled}
            onAccept={(text) => onAccept(suggestion.id, text)}
            onDismiss={() => onDismiss(suggestion.id)}
            onUndo={() => onUndo(suggestion.id)}
          />
        ))}
      </ul>
    </section>
  )
}

function SuggestionRow({ suggestion, disabled, onAccept, onDismiss, onUndo }) {
  const textId = useId()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(suggestion.text)

  if (suggestion.status === 'saved') {
    return (
      <li className="memsug__item is-saved">
        <IconCheck size={14} className="memsug__check" />
        <p id={textId}>{suggestion.text}</p>
        <div className="memsug__actions">
          <button type="button" className="cxturn__action" aria-describedby={textId} onClick={onUndo} disabled={disabled}>Undo</button>
        </div>
      </li>
    )
  }

  if (editing) {
    const cancel = () => { setEditing(false); setDraft(suggestion.text) }
    const save = () => { if (draft.trim()) { onAccept(draft); setEditing(false) } }
    return (
      <li className="memsug__item is-editing">
        <input
          className="memsug__input"
          aria-label="Edit what to remember"
          value={draft}
          maxLength={MAX_MEMORY_CHARS}
          autoFocus
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') { event.preventDefault(); save() }
            if (event.key === 'Escape') { event.preventDefault(); cancel() }
          }}
        />
        <div className="memsug__actions">
          <button type="button" className="memsug__primary" onClick={save} disabled={disabled || !draft.trim()}>Remember</button>
          <button type="button" className="cxturn__action" onClick={cancel}>Cancel</button>
        </div>
      </li>
    )
  }

  return (
    <li className="memsug__item">
      <p id={textId}>{suggestion.text}</p>
      <div className="memsug__actions">
        <button type="button" className="memsug__primary" aria-describedby={textId} onClick={() => onAccept()} disabled={disabled}>Remember</button>
        <button type="button" className="cxturn__action" aria-describedby={textId} onClick={() => setEditing(true)} disabled={disabled}>Edit</button>
        <button type="button" className="cxturn__action" aria-describedby={textId} onClick={onDismiss} disabled={disabled}>Not now</button>
      </div>
    </li>
  )
}
