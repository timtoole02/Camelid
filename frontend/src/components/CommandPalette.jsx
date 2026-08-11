import { useEffect, useMemo, useRef, useState } from 'react'

/* Command palette (Phase 7). Cmd/Ctrl+K. Actions are built from live app
   state: views, conversations, theme, models (labels stay gate-honest — a
   model that cannot chat says so), and compatibility rows (jump = the same
   camelid:open-ledger event Evidence Chips use). Plain substring filter,
   arrow/enter keyboard model, dialog semantics. */

const VIEW_LABELS = [
  ['chat', 'Chat'],
  ['workspace', 'Workspace'],
  ['library', 'Models'],
  ['history', 'Chat history'],
  ['analytics', 'Analytics'],
  ['telemetry', 'Telemetry'],
  ['memory', 'Memory'],
  ['system', 'System'],
  ['api', 'API'],
  ['compatibility', 'Compatibility'],
  ['cluster', 'Cluster'],
  ['observatory', 'Observatory'],
  ['settings', 'Settings'],
]

export function buildPaletteActions({ setTab, showNewChatLanding, cyclePreference, models = [], capabilities = [], setSelectedModelId, close }) {
  const actions = []
  for (const [tab, label] of VIEW_LABELS) {
    actions.push({ id: `nav-${tab}`, group: 'Navigate', label, hint: `#${tab}`, run: () => { setTab(tab); close() } })
  }
  actions.push({ id: 'new-chat', group: 'Chat', label: 'New conversation', hint: 'fresh thread', run: () => { showNewChatLanding(); close() } })
  actions.push({ id: 'toggle-theme', group: 'Appearance', label: 'Toggle theme', hint: 'dark → light → system', run: () => { cyclePreference(); close() } })
  for (const model of models) {
    actions.push({
      id: `model-${model.id}`,
      group: 'Switch model',
      label: model.name || model.id,
      hint: 'select for next chat — readiness still gates send',
      run: () => { setSelectedModelId(model.id); setTab('chat'); close() },
    })
  }
  for (const row of capabilities) {
    actions.push({
      id: `row-${row.id}`,
      group: 'Compatibility',
      label: row.id,
      hint: `${row.family} · ${row.quantization}`,
      run: () => {
        close()
        window.dispatchEvent(new CustomEvent('camelid:open-ledger', { detail: { rowId: row.id } }))
      },
    })
  }
  return actions
}

export function CommandPalette({ open, onClose, ...sources }) {
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const inputRef = useRef(null)
  const listRef = useRef(null)

  const actions = useMemo(
    () => buildPaletteActions({ ...sources, close: onClose }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [open, sources.models, sources.capabilities],
  )
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return actions.slice(0, 40)
    return actions.filter((action) => `${action.group} ${action.label} ${action.hint}`.toLowerCase().includes(needle)).slice(0, 40)
  }, [actions, query])

  useEffect(() => {
    if (open) {
      setQuery('')
      setCursor(0)
      window.requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  /* Closing returns focus to whatever had it before Cmd/Ctrl+K — same contract
     as Modal.jsx. Depends on `open` alone so the cleanup (and the focus jump)
     only fires when the palette actually closes, not on parent re-renders. */
  useEffect(() => {
    if (!open) return undefined
    const previouslyFocused = document.activeElement
    return () => {
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') previouslyFocused.focus()
    }
  }, [open])

  /* Escape must close the palette no matter where focus sits (frame, footer). */
  useEffect(() => {
    if (!open) return undefined
    const onKey = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  useEffect(() => {
    setCursor((current) => Math.min(current, Math.max(filtered.length - 1, 0)))
  }, [filtered.length])

  useEffect(() => {
    const node = listRef.current?.querySelector('[aria-selected="true"]')
    node?.scrollIntoView({ block: 'nearest' })
  }, [cursor])

  if (!open) return null

  const onKeyDown = (event) => {
    if (event.key === 'ArrowDown') { event.preventDefault(); setCursor((c) => Math.min(c + 1, filtered.length - 1)) }
    if (event.key === 'ArrowUp') { event.preventDefault(); setCursor((c) => Math.max(c - 1, 0)) }
    if (event.key === 'Enter') { event.preventDefault(); filtered[cursor]?.run() }
  }

  return (
    <div className="palette-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Command palette">
        <input
          ref={inputRef}
          className="palette__input"
          aria-label="Search commands"
          placeholder="Navigate, switch model, jump to a compatibility row…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          role="combobox"
          aria-expanded="true"
          aria-controls="palette-results"
          aria-activedescendant={filtered[cursor] ? `palette-item-${filtered[cursor].id}` : undefined}
        />
        {filtered.length === 0 && <div className="palette__empty">Nothing matches “{query}”.</div>}
        <ul className="palette__list" id="palette-results" role="listbox" ref={listRef}>
          {filtered.map((action, index) => (
            <li
              key={action.id}
              id={`palette-item-${action.id}`}
              role="option"
              aria-selected={index === cursor}
              className={`palette__item ${index === cursor ? 'is-active' : ''}`}
              onMouseEnter={() => setCursor(index)}
              onMouseDown={(event) => { event.preventDefault(); action.run() }}
            >
              <span className="palette__group">{action.group}</span>
              <span className="palette__label">{action.label}</span>
              <span className="palette__hint">{action.hint}</span>
            </li>
          ))}
        </ul>
        <footer className="palette__foot">↑↓ navigate · Enter run · Esc close</footer>
      </div>
    </div>
  )
}

export default CommandPalette
