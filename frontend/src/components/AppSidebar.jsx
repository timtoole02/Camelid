import { memo, useEffect, useMemo, useState } from 'react'
import { clampText, formatPreview, formatSidebarDate } from '../lib/formatters'

const GeminiSparkle = ({ className = '', size = 18 }) => (
  <svg
    className={`gemini-sparkle-icon ${className}`}
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M12 3C12 3 12.3 8.3 15.5 11.5C18.7 14.7 24 15 24 15C24 15 18.7 15.3 15.5 18.5C12.3 21.7 12 27 12 27C12 27 11.7 21.7 8.5 18.5C5.3 15.3 0 15 0 15C0 15 5.3 14.7 8.5 11.5C11.7 8.3 12 3 12 3Z"
      fill="url(#gemini-sparkle-grad)"
    />
    <defs>
      <linearGradient id="gemini-sparkle-grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#4285f4" />
        <stop offset="35%" stopColor="#9b51e0" />
        <stop offset="70%" stopColor="#e289f2" />
        <stop offset="100%" stopColor="#fa9085" />
      </linearGradient>
    </defs>
  </svg>
)


const tabs = [
  { id: 'chat', label: 'Chat' },
  { id: 'library', label: 'Models' },
  { id: 'api', label: 'API' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'history', label: 'History' },
  { id: 'memory', label: 'Memory' },
  { id: 'system', label: 'System' },
]

const primaryTabs = tabs.slice(0, 3)

const recencyBuckets = ['Today', 'Yesterday', 'Previous 7 days', 'Earlier']

const isMeaningfulConversationMessage = (message) =>
  typeof message?.content === 'string' && message.content.trim() && !message.content.startsWith('Conversation created.')

function startOfDay(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate())
}

function getConversationBucket(value) {
  if (!value) return 'Earlier'

  const date = new Date(value)
  const now = new Date()
  const diffDays = Math.floor((startOfDay(now) - startOfDay(date)) / 86400000)

  if (diffDays <= 0) return 'Today'
  if (diffDays === 1) return 'Yesterday'
  if (diffDays <= 7) return 'Previous 7 days'
  return 'Earlier'
}

function MenuDotsIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">
      <circle cx="10" cy="4" r="1.5" fill="currentColor" />
      <circle cx="10" cy="10" r="1.5" fill="currentColor" />
      <circle cx="10" cy="16" r="1.5" fill="currentColor" />
    </svg>
  )
}

function RenameIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">
      <path d="M4 13.75V16h2.25l6.64-6.64-2.25-2.25L4 13.75Zm10.71-6.46a.996.996 0 0 0 0-1.41l-.59-.59a.996.996 0 1 0-1.41 1.41l.59.59a.996.996 0 0 0 1.41 0Z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function DeleteIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">
      <path d="M6.25 5.5h7.5m-6.75 0 .45 9h5.1l.45-9M8.5 5.5V4.4c0-.22.18-.4.4-.4h2.2c.22 0 .4.18.4.4v1.1m-6.1 0h8.2" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function AppSidebar({
  collapsed,
  dragging,
  onResizeStart,
  onResizeKeyDown,
  onToggleCollapsed,
  width,
  createConversation,
  showNewChatLanding,
  search,
  setSearch,
  tab,
  setTab,
  filteredConversations,
  selectedConversationId,
  setSelectedConversationId,
  deleteConversation,
  renameConversation,
}) {
  const [openMenuId, setOpenMenuId] = useState(null)
  const [editingConversationId, setEditingConversationId] = useState(null)
  const [editingTitle, setEditingTitle] = useState('')
  const utilityTabs = useMemo(() => tabs.filter((item) => !primaryTabs.some((primary) => primary.id === item.id)), [])

  const titleCounts = useMemo(() => {
    const counts = new Map()
    filteredConversations.forEach((conversation) => {
      const key = (conversation.title || 'Untitled conversation').trim().toLowerCase()
      counts.set(key, (counts.get(key) || 0) + 1)
    })
    return counts
  }, [filteredConversations])

  const groupedConversations = useMemo(() => {
    const groups = new Map(recencyBuckets.map((label) => [label, []]))
    filteredConversations.forEach((conversation) => {
      groups.get(getConversationBucket(conversation.updated_at))?.push(conversation)
    })
    return recencyBuckets
      .map((label) => ({ label, items: groups.get(label) || [] }))
      .filter((group) => group.items.length > 0)
  }, [filteredConversations])

  useEffect(() => {
    if (!openMenuId) return undefined

    const handlePointerDown = (event) => {
      if (!event.target.closest('[data-conversation-menu-root="true"]')) {
        setOpenMenuId(null)
      }
    }

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        if (editingConversationId) {
          setEditingConversationId(null)
          setEditingTitle('')
        }
        setOpenMenuId(null)
      }
    }

    window.addEventListener('pointerdown', handlePointerDown)
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('pointerdown', handlePointerDown)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [editingConversationId, openMenuId])

  useEffect(() => {
    setOpenMenuId(null)
    setEditingConversationId(null)
    setEditingTitle('')
  }, [collapsed, selectedConversationId, tab])

  const startRename = (conversation) => {
    setSelectedConversationId(conversation.id)
    setTab('chat')
    setOpenMenuId(null)
    setEditingConversationId(conversation.id)
    setEditingTitle(conversation.title || '')
  }

  const cancelRename = () => {
    setEditingConversationId(null)
    setEditingTitle('')
  }

  const submitRename = async () => {
    if (!editingConversationId) return
    const ok = await renameConversation(editingConversationId, editingTitle)
    if (ok) {
      setEditingConversationId(null)
      setEditingTitle('')
    }
  }

  return (
    <>
      <aside id="camelid-sidebar" className={`sidebar ${collapsed ? 'is-collapsed sidebar-rail' : ''}`} aria-hidden={false}>
        {collapsed ? (
          <>
            <div className="sidebar-rail-top">
              <button
                type="button"
                className="sidebar-rail-button"
                aria-label="Show sidebar"
                aria-controls="camelid-sidebar"
                aria-expanded="false"
                onClick={onToggleCollapsed}
              >
                <span aria-hidden="true">☰</span>
              </button>
              <button type="button" className="sidebar-rail-button" aria-label="Start a new chat" onClick={showNewChatLanding}>
                <span aria-hidden="true">＋</span>
              </button>
            </div>

            <nav className="sidebar-rail-nav" aria-label="Primary navigation">
              {[
                { id: 'chat', label: 'Chat', glyph: '✦' },
                { id: 'library', label: 'Models', glyph: '⌘' },
                { id: 'api', label: 'API', glyph: '⌁' },
                { id: 'analytics', label: 'Analytics', glyph: '◫' },
                { id: 'history', label: 'History', glyph: '◷' },
                { id: 'memory', label: 'Memory', glyph: '☷' },
                { id: 'system', label: 'System', glyph: '⚙' },
              ].map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`sidebar-rail-button ${tab === item.id ? 'active' : ''}`}
                  aria-label={item.label}
                  aria-current={tab === item.id ? 'page' : undefined}
                  onClick={() => setTab(item.id)}
                >
                  <span aria-hidden="true">{item.glyph}</span>
                </button>
              ))}
            </nav>

            <div className="sidebar-rail-bottom">
              <button type="button" className="sidebar-rail-button" aria-label="Open current conversation history" onClick={() => setTab('history')}>
                <span aria-hidden="true">…</span>
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="sidebar-top sidebar-top-flat">
              <div className="sidebar-utility-row">
                <button
                  type="button"
                  className="sidebar-brand-link sidebar-brand-minimal sidebar-brand-minimal-quiet"
                  onClick={() => {
                    showNewChatLanding()
                    window.scrollTo({ top: 0, behavior: 'auto' })
                  }}
                  aria-label="Go to the front page"
                  title="Go to the front page"
                >
                  <strong className="sidebar-brand-with-sparkle"><GeminiSparkle size={18} className="sidebar-brand-sparkle-icon" /> Camelid</strong>
                </button>
                <div className="sidebar-utility-actions">
                  <button type="button" className="sidebar-utility-button" aria-label="Start a new chat" onClick={showNewChatLanding}>
                    <span aria-hidden="true">✎</span>
                  </button>
                  <button
                    type="button"
                    className="sidebar-utility-button"
                    aria-label="Collapse sidebar"
                    aria-controls="camelid-sidebar"
                    aria-expanded={!collapsed}
                    onClick={onToggleCollapsed}
                  >
                    <span aria-hidden="true">☰</span>
                  </button>
                </div>
              </div>

              <button className="sidebar-quick-action sidebar-quick-action-assistant" onClick={showNewChatLanding}>
                <span className="sidebar-quick-action-icon" aria-hidden="true">＋</span>
                <strong>New chat</strong>
              </button>

              <div className="sidebar-search-inline sidebar-search-inline-assistant">
                <span className="sidebar-search-icon" aria-hidden="true">⌕</span>
                <input
                  className="sidebar-input sidebar-input-search"
                  aria-label="Search conversations"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search chats"
                />
              </div>

              <div className="sidebar-flat-section sidebar-flat-section-assistant">
                <div className="sidebar-flat-label">Workspace</div>
                <nav className="nav-stack nav-stack-flat" aria-label="Primary navigation">
                  {primaryTabs.map((item) => (
                    <button key={item.id} className={`nav-item nav-item-flat ${tab === item.id ? 'active' : ''}`} aria-current={tab === item.id ? 'page' : undefined} onClick={() => setTab(item.id)}>
                      <strong>{item.label}</strong>
                    </button>
                  ))}
                </nav>
              </div>

              <details className="sidebar-tools-disclosure" open={tab !== 'chat' && tab !== 'library' && tab !== 'api'}>
                <summary className="sidebar-tools-summary">
                  <span className="sidebar-flat-label sidebar-flat-label-inline">More tools</span>
                  <small>{utilityTabs.map((item) => item.label).join(' · ')}</small>
                </summary>
                <div className="sidebar-quick-link-row sidebar-quick-link-row-tools" aria-label="Secondary workspace tools">
                  {utilityTabs.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className={`sidebar-mini-link ${tab === item.id ? 'active' : ''}`}
                      onClick={() => setTab(item.id)}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </details>
            </div>

            <div className="sidebar-bottom sidebar-bottom-flat sidebar-bottom-assistant">
              <div className="sidebar-list-header sidebar-list-header-flat">
                <div>
                  <p className="panel-kicker">Chats</p>
                </div>
                <small>{filteredConversations.length}</small>
              </div>

              <div className="conversation-list conversation-list-flat conversation-list-assistant">
                {groupedConversations.map((group) => (
                  <section key={group.label} className="conversation-group">
                    <div className="conversation-group-label">{group.label}</div>
                    <div className="conversation-group-items">
                      {group.items.map((conversation) => {
                        const rawTitle = conversation.title || 'Untitled conversation'
                        const normalizedTitle = rawTitle.trim().toLowerCase()
                        return (
                          <ConversationListItem
                            key={conversation.id}
                            conversation={conversation}
                            hasDuplicateTitle={(titleCounts.get(normalizedTitle) || 0) > 1}
                            selected={conversation.id === selectedConversationId}
                            menuOpen={openMenuId === conversation.id}
                            editing={editingConversationId === conversation.id}
                            editingTitle={editingTitle}
                            setEditingTitle={setEditingTitle}
                            submitRename={submitRename}
                            cancelRename={cancelRename}
                            setSelectedConversationId={setSelectedConversationId}
                            setTab={setTab}
                            setOpenMenuId={setOpenMenuId}
                            startRename={startRename}
                            deleteConversation={deleteConversation}
                          />
                        )
                      })}
                    </div>
                  </section>
                ))}
              </div>
            </div>
          </>
        )}
      </aside>

      {collapsed ? (
        <div className="sidebar-grid-spacer" aria-hidden="true" />
      ) : (
        <button
          type="button"
          className={`sidebar-resizer ${dragging ? 'is-dragging' : ''}`}
          role="separator"
          aria-label={`Resize sidebar, current width ${Math.round(width)} pixels`}
          aria-controls="camelid-sidebar"
          aria-valuemin={240}
          aria-valuemax={420}
          aria-valuenow={Math.round(width)}
          aria-orientation="vertical"
          onPointerDown={onResizeStart}
          onKeyDown={onResizeKeyDown}
        >
          <span className="sidebar-resizer-line" />
        </button>
      )}
    </>
  )
}

const ConversationListItem = memo(function ConversationListItem({
  conversation,
  hasDuplicateTitle,
  selected,
  menuOpen,
  editing,
  editingTitle,
  setEditingTitle,
  submitRename,
  cancelRename,
  setSelectedConversationId,
  setTab,
  setOpenMenuId,
  startRename,
  deleteConversation,
}) {
  const rawTitle = conversation.title || 'Untitled conversation'
  const conversationTitle = clampText(rawTitle, 58) || 'Untitled conversation'
  const normalizedTitle = rawTitle.trim().toLowerCase()
  const latestMessage = [...(conversation.messages || [])].reverse().find((message) => isMeaningfulConversationMessage(message))
  const preview = formatPreview(latestMessage?.content, 42) || 'New chat'
  const timeLabel = formatSidebarDate(conversation.updated_at)
  const subtitle = hasDuplicateTitle || normalizedTitle === 'untitled conversation'
    ? `${preview} · ${timeLabel}`
    : preview

  return (
    <div className={`conversation-item-row conversation-item-row-flat ${selected ? 'selected' : ''} ${menuOpen ? 'has-open-menu' : ''}`}>
      <button
        className={`conversation-item conversation-item-flat ${selected ? 'selected' : ''}`}
        aria-current={selected ? 'true' : undefined}
        onClick={() => {
          if (editing) return
          setSelectedConversationId(conversation.id)
          setTab('chat')
        }}
      >
        <div className="conversation-item-flat-main">
          {editing ? (
            <input
              className="conversation-rename-input"
              value={editingTitle}
              onChange={(event) => setEditingTitle(event.target.value)}
              onClick={(event) => event.stopPropagation()}
              onBlur={submitRename}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault()
                  void submitRename()
                }
                if (event.key === 'Escape') {
                  event.preventDefault()
                  cancelRename()
                }
              }}
              autoFocus
              aria-label={`Rename ${rawTitle}`}
            />
          ) : (
            <>
              <strong title={rawTitle}>{conversationTitle}</strong>
              <span className="conversation-item-subtitle" title={subtitle}>{subtitle}</span>
            </>
          )}
        </div>
      </button>
      {!editing && (
        <div className={`conversation-row-actions ${menuOpen ? 'is-open' : ''}`} data-conversation-menu-root="true">
          <button
            type="button"
            className="conversation-menu-button"
            aria-label={`Open chat menu for ${conversationTitle}`}
            aria-haspopup="menu"
            aria-expanded={menuOpen ? 'true' : 'false'}
            onClick={(event) => {
              event.stopPropagation()
              setOpenMenuId((current) => (current === conversation.id ? null : conversation.id))
            }}
          >
            <MenuDotsIcon />
          </button>
          {menuOpen && (
            <div className="conversation-menu-popover" role="menu" aria-label={`Actions for ${conversationTitle}`}>
              <button type="button" role="menuitem" className="conversation-menu-item" onClick={() => startRename(conversation)}>
                <span className="conversation-menu-item-icon"><RenameIcon /></span>
                <span>Rename</span>
              </button>
              <button
                type="button"
                role="menuitem"
                className="conversation-menu-item conversation-menu-item-danger"
                onClick={(event) => {
                  event.stopPropagation()
                  setOpenMenuId(null)
                  deleteConversation(conversation.id)
                }}
              >
                <span className="conversation-menu-item-icon"><DeleteIcon /></span>
                <span>Delete</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
})

export default memo(AppSidebar)
