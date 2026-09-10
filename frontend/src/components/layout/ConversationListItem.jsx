import { memo, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { clampText } from '../../lib/formatters'
import { IconClose, IconDots, IconEdit, IconFile, IconPin, IconTrash } from '../ui/icons'
import { MAX_TAG_LENGTH, isArchived, isPinned, tagsOf } from '../../lib/conversationOrganization.js'

function ConversationListItemInner({
  conversation,
  selected,
  collapsed,
  onSelect,
  onRename,
  onDelete,
  onTogglePin,
  onToggleArchive,
  onAddTag,
  onRemoveTag,
}) {
  const pinned = isPinned(conversation)
  const archived = isArchived(conversation)
  const tags = tagsOf(conversation)
  /* Non-null while the menu is open: viewport coordinates for the portalled
     popover. Portalling (instead of position:absolute in the row) lets the
     menu escape the sidebar's overflow-y:auto scroll container. */
  const [menuPos, setMenuPos] = useState(null)
  const [editing, setEditing] = useState(false)
  const [draftTitle, setDraftTitle] = useState('')
  const rootRef = useRef(null)
  const triggerRef = useRef(null)
  const menuRef = useRef(null)
  const menuOpen = Boolean(menuPos)

  const rawTitle = conversation.title || 'Untitled conversation'
  const title = clampText(rawTitle, 52) || 'Untitled conversation'

  const openMenu = () => {
    const rect = triggerRef.current?.getBoundingClientRect()
    if (!rect) return
    setMenuPos({ top: rect.bottom + 4, right: window.innerWidth - rect.right })
  }
  const closeMenu = ({ refocus = false } = {}) => {
    setMenuPos(null)
    if (refocus) triggerRef.current?.focus()
  }

  useEffect(() => {
    if (!menuOpen) return undefined
    const onDocDown = (event) => {
      if (rootRef.current?.contains(event.target) || menuRef.current?.contains(event.target)) return
      setMenuPos(null)
    }
    const onKey = (event) => {
      if (event.key === 'Escape') {
        setMenuPos(null)
        triggerRef.current?.focus()
      }
    }
    /* The fixed-position menu goes stale the moment the list scrolls. */
    const onMove = () => setMenuPos(null)
    window.addEventListener('pointerdown', onDocDown)
    window.addEventListener('keydown', onKey)
    window.addEventListener('scroll', onMove, true)
    window.addEventListener('resize', onMove)
    return () => {
      window.removeEventListener('pointerdown', onDocDown)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', onMove, true)
      window.removeEventListener('resize', onMove)
    }
  }, [menuOpen])

  /* role='menu' contract: focus moves into the menu on open. */
  useEffect(() => {
    if (!menuOpen) return undefined
    const frame = window.requestAnimationFrame(() => {
      menuRef.current?.querySelector('[role="menuitem"]')?.focus()
    })
    return () => window.cancelAnimationFrame(frame)
  }, [menuOpen])

  /* role='menu' contract: ArrowUp/Down cycle items, Home/End jump, Tab closes. */
  const onMenuKeyDown = (event) => {
    const items = Array.from(menuRef.current?.querySelectorAll('[role="menuitem"]') || [])
    if (!items.length) return
    const index = items.indexOf(document.activeElement)
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      items[(index + 1) % items.length].focus()
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      items[(index - 1 + items.length) % items.length].focus()
    } else if (event.key === 'Home') {
      event.preventDefault()
      items[0].focus()
    } else if (event.key === 'End') {
      event.preventDefault()
      items[items.length - 1].focus()
    } else if (event.key === 'Tab') {
      closeMenu({ refocus: true })
    }
  }

  const beginRename = () => {
    closeMenu()
    setDraftTitle(rawTitle)
    setEditing(true)
  }
  const commitRename = async () => {
    const ok = await onRename(conversation.id, draftTitle)
    if (ok !== false) setEditing(false)
  }

  if (editing) {
    return (
      <div className="rail-convo rail-convo--editing" ref={rootRef}>
        <input
          className="rail-convo__rename"
          value={draftTitle}
          autoFocus
          aria-label={`Rename ${rawTitle}`}
          onChange={(e) => setDraftTitle(e.target.value)}
          onBlur={commitRename}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); void commitRename() }
            if (e.key === 'Escape') { e.preventDefault(); setEditing(false) }
          }}
        />
      </div>
    )
  }

  return (
    <div className={`rail-convo ${selected ? 'is-selected' : ''} ${menuOpen ? 'has-menu' : ''}`} ref={rootRef}>
      <button
        type="button"
        className="rail-convo__main"
        aria-current={selected ? 'true' : undefined}
        title={rawTitle}
        onClick={() => onSelect(conversation.id)}
      >
        {pinned && !collapsed && <IconPin size={12} className="rail-convo__pin" aria-label="Pinned" />}
        <span className="rail-convo__title">{collapsed ? rawTitle.slice(0, 1).toUpperCase() : title}</span>
        {/* Tags are shown on the row, not only in the menu: a label nobody can
            see is a label nobody applies twice. */}
        {!collapsed && tags.length > 0 && (
          <span className="rail-convo__tags">
            {tags.slice(0, 2).map((tag) => <span key={tag} className="rail-convo__tag">{tag}</span>)}
            {tags.length > 2 && <span className="rail-convo__tag rail-convo__tag--more">+{tags.length - 2}</span>}
          </span>
        )}
      </button>
      {!collapsed && (
        <div className="rail-convo__actions">
          <button
            type="button"
            ref={triggerRef}
            className="rail-convo__menu-btn"
            aria-label={`Options for ${title}`}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            onClick={(e) => { e.stopPropagation(); if (menuOpen) closeMenu(); else openMenu() }}
          >
            <IconDots size={18} />
          </button>
          {menuOpen && createPortal(
            <div
              className="rail-menu"
              role="menu"
              aria-label={`Options for ${title}`}
              ref={menuRef}
              style={{ top: menuPos.top, right: menuPos.right }}
              onKeyDown={onMenuKeyDown}
            >
              <button type="button" role="menuitem" className="rail-menu__item" onClick={beginRename}>
                <IconEdit size={16} /> <span>Rename</span>
              </button>
              {onTogglePin && (
                <button
                  type="button"
                  role="menuitem"
                  className="rail-menu__item"
                  onClick={() => { closeMenu(); onTogglePin(conversation.id, !pinned) }}
                >
                  <IconPin size={16} /> <span>{pinned ? 'Unpin' : 'Pin to top'}</span>
                </button>
              )}
              {onToggleArchive && (
                <button
                  type="button"
                  role="menuitem"
                  className="rail-menu__item"
                  onClick={() => { closeMenu(); onToggleArchive(conversation.id, !archived) }}
                >
                  <IconFile size={16} /> <span>{archived ? 'Unarchive' : 'Archive'}</span>
                </button>
              )}
              {onAddTag && (
                <div className="rail-menu__tags">
                  {tags.length > 0 && (
                    <div className="rail-menu__tag-list">
                      {tags.map((tag) => (
                        <button
                          key={tag}
                          type="button"
                          className="rail-menu__tag"
                          onClick={() => onRemoveTag?.(conversation.id, tag)}
                          title={`Remove tag “${tag}”`}
                          aria-label={`Remove tag ${tag}`}
                        >
                          {tag} <IconClose size={11} />
                        </button>
                      ))}
                    </div>
                  )}
                  <input
                    className="rail-menu__tag-input"
                    placeholder="Add tag…"
                    maxLength={MAX_TAG_LENGTH}
                    aria-label={`Add a tag to ${title}`}
                    /* Enter commits and keeps the menu open: tagging is
                       usually more than one tag, and reopening the menu per
                       tag is the difference between using this and not. */
                    onKeyDown={(event) => {
                      if (event.key !== 'Enter') return
                      event.preventDefault()
                      const value = event.currentTarget.value
                      if (!value.trim()) return
                      onAddTag(conversation.id, value)
                      event.currentTarget.value = ''
                    }}
                  />
                </div>
              )}
              <button
                type="button"
                role="menuitem"
                className="rail-menu__item rail-menu__item--danger"
                onClick={() => { closeMenu(); onDelete(conversation.id) }}
              >
                <IconTrash size={16} /> <span>Delete</span>
              </button>
            </div>,
            document.body,
          )}
        </div>
      )}
    </div>
  )
}

export const ConversationListItem = memo(ConversationListItemInner)
export default ConversationListItem
