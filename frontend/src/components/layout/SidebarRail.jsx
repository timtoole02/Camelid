import { useMemo } from 'react'
import { CamelidMark } from '../ui/CamelidMark'
import { StatusDot } from '../ui/StatusDot'
import { ThemeToggle } from '../ui/ThemeToggle'
import { Tooltip } from '../ui/Tooltip'
import { ConversationListItem } from './ConversationListItem'
import {
  IconAnalytics, IconApi, IconBolt, IconChart, IconChat, IconHistory, IconMemory, IconModels,
  IconNetwork, IconNewChat, IconObservatory, IconReceipt, IconSearch, IconSettings, IconSidebar, IconSystem,
} from '../ui/icons'

const NAV = [
  { tab: 'code', label: 'Code', Icon: IconBolt },
  { tab: 'workspace', label: 'Workspace', Icon: IconBolt },
  { tab: 'library', label: 'Models', Icon: IconModels },
  { tab: 'history', label: 'Chat history', Icon: IconHistory },
  { tab: 'analytics', label: 'Analytics', Icon: IconAnalytics },
  { tab: 'telemetry', label: 'Telemetry', Icon: IconChart },
  { tab: 'memory', label: 'Memory', Icon: IconMemory },
  { tab: 'system', label: 'System', Icon: IconSystem },
  { tab: 'api', label: 'API', Icon: IconApi },
  { tab: 'compatibility', label: 'Compatibility', Icon: IconReceipt },
  { tab: 'cluster', label: 'Cluster', Icon: IconNetwork },
  { tab: 'observatory', label: 'Inference Observatory', Icon: IconObservatory },
  { tab: 'settings', label: 'Settings', Icon: IconSettings },
]

const BUCKETS = ['Today', 'Yesterday', 'Previous 7 days', 'Earlier']
function startOfDay(d) { return new Date(d.getFullYear(), d.getMonth(), d.getDate()) }
function bucketFor(value) {
  if (!value) return 'Earlier'
  const normalized = typeof value === 'number' && value < 10_000_000_000 ? value * 1000 : value
  const diff = Math.floor((startOfDay(new Date()) - startOfDay(new Date(normalized))) / 86400000)
  if (diff <= 0) return 'Today'
  if (diff === 1) return 'Yesterday'
  if (diff <= 7) return 'Previous 7 days'
  return 'Earlier'
}

export function SidebarRail({
  collapsed,
  onToggleCollapsed,
  showNewChatLanding,
  search,
  setSearch,
  tab,
  setTab,
  filteredConversations,
  selectedConversationId,
  onSelectConversation,
  renameConversation,
  requestDeleteConversation,
  codeThreads = [],
  selectedCodeThreadId = '',
  onSelectCodeThread,
  onNewCodeSession,
  runtime,
  themePreference,
  themeResolved,
  onCycleTheme,
}) {
  const grouped = useMemo(() => {
    const groups = new Map(BUCKETS.map((b) => [b, []]))
    filteredConversations.forEach((c) => groups.get(bucketFor(c.updated_at))?.push(c))
    return BUCKETS.map((label) => ({ label, items: groups.get(label) || [] })).filter((g) => g.items.length)
  }, [filteredConversations])
  const groupedCodeThreads = useMemo(() => {
    const needle = String(search || '').trim().toLowerCase()
    const groups = new Map(BUCKETS.map((b) => [b, []]))
    codeThreads
      .filter((thread) => !needle || `${thread.title} ${thread.canonical_root}`.toLowerCase().includes(needle))
      .forEach((thread) => groups.get(bucketFor(thread.updated_at))?.push(thread))
    return BUCKETS.map((label) => ({ label, items: groups.get(label) || [] })).filter((group) => group.items.length)
  }, [codeThreads, search])
  const showingCode = tab === 'code'
  const visibleGroups = showingCode ? groupedCodeThreads : grouped

  const online = runtime?.status === 'online'
  const statusTone = online ? 'ready' : 'offline'
  const statusLabel = online ? 'Camelid online' : 'Camelid offline'

  if (collapsed) {
    return (
      <aside className="rail rail--collapsed" id="camelid-sidebar" aria-label="Navigation rail">
        <div className="rail__rail-top">
          <Tooltip content="Expand sidebar" placement="right">
            <button type="button" className="rail__icon-btn" aria-label="Expand sidebar" onClick={onToggleCollapsed}>
              <IconSidebar size={20} />
            </button>
          </Tooltip>
          <Tooltip content={showingCode ? 'New coding session' : 'New chat'} placement="right">
            <button type="button" className="rail__icon-btn rail__icon-btn--accent" aria-label={showingCode ? 'New coding session' : 'New chat'} onClick={showingCode ? onNewCodeSession : showNewChatLanding}>
              <IconNewChat size={20} />
            </button>
          </Tooltip>
        </div>
        <nav className="rail__rail-nav" aria-label="Primary">
          <Tooltip content="Chat" placement="right">
            <button type="button" className={`rail__icon-btn ${tab === 'chat' ? 'is-active' : ''}`} aria-label="Chat" aria-current={tab === 'chat' ? 'page' : undefined} onClick={() => setTab('chat')}>
              <IconChat size={20} />
            </button>
          </Tooltip>
          {NAV.map(({ tab: t, label, Icon }) => (
            <Tooltip key={t} content={label} placement="right">
              <button type="button" className={`rail__icon-btn ${tab === t ? 'is-active' : ''}`} aria-label={label} aria-current={tab === t ? 'page' : undefined} onClick={() => setTab(t)}>
                <Icon size={20} />
              </button>
            </Tooltip>
          ))}
        </nav>
        <div className="rail__rail-bottom">
          <ThemeToggle preference={themePreference} resolved={themeResolved} onCycle={onCycleTheme} compact />
          <Tooltip content={statusLabel} placement="right">
            <span className="rail__status-icon"><StatusDot tone={statusTone} pulse={online} /></span>
          </Tooltip>
        </div>
      </aside>
    )
  }

  return (
    <aside className="rail" id="camelid-sidebar" aria-label="Navigation sidebar">
      <div className="rail__header">
        <button type="button" className="rail__brand" onClick={showNewChatLanding} aria-label="Camelid home">
          <CamelidMark size={24} />
          <span className="rail__brand-name">Camelid</span>
        </button>
        <button type="button" className="rail__icon-btn" aria-label="Collapse sidebar" onClick={onToggleCollapsed}>
          <IconSidebar size={20} />
        </button>
      </div>

      <button type="button" className="rail__new-chat" onClick={showingCode ? onNewCodeSession : showNewChatLanding}>
        <IconNewChat size={18} />
        <span>{showingCode ? 'New coding session' : 'New chat'}</span>
      </button>

      <div className="rail__search">
        <IconSearch size={16} />
        <input
          className="rail__search-input"
          aria-label={showingCode ? 'Search coding sessions' : 'Search conversations'}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={showingCode ? 'Search code history' : 'Search chats'}
        />
      </div>

      <div className="rail__scroll">
        <div className="rail__section">
          <div className="rail__section-label">{showingCode ? 'Code history' : 'Recent'}</div>
          {visibleGroups.length === 0 && <p className="rail__empty">{showingCode ? 'No coding sessions yet' : 'No conversations yet'}</p>}
          {visibleGroups.map((group) => (
            <div key={group.label} className="rail__group">
              <div className="rail__group-label">{group.label}</div>
              {showingCode ? group.items.map((thread) => (
                <button
                  type="button"
                  key={thread.id}
                  className={`rail-code-thread ${thread.id === selectedCodeThreadId ? 'is-selected' : ''}`}
                  onClick={() => onSelectCodeThread?.(thread)}
                  title={thread.canonical_root}
                >
                  <IconBolt size={15} />
                  <span><strong>{thread.title || 'Coding session'}</strong><small>{thread.canonical_root}</small></span>
                </button>
              )) : group.items.map((conversation) => (
                <ConversationListItem
                  key={conversation.id}
                  conversation={conversation}
                  collapsed={false}
                  selected={tab === 'chat' && conversation.id === selectedConversationId}
                  onSelect={onSelectConversation}
                  onRename={renameConversation}
                  onDelete={requestDeleteConversation}
                />
              ))}
            </div>
          ))}
        </div>

        <nav className="rail__section rail__nav" aria-label="Workspace">
          <div className="rail__section-label">Workspace</div>
          {NAV.map(({ tab: t, label, Icon }) => (
            <button
              key={t}
              type="button"
              className={`rail__nav-item ${tab === t ? 'is-active' : ''}`}
              aria-current={tab === t ? 'page' : undefined}
              onClick={() => setTab(t)}
            >
              <Icon size={20} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </div>

      <div className="rail__footer">
        <ThemeToggle preference={themePreference} resolved={themeResolved} onCycle={onCycleTheme} />
        <span className="rail__status"><StatusDot tone={statusTone} pulse={online} label={statusLabel} /></span>
      </div>
    </aside>
  )
}

export default SidebarRail
