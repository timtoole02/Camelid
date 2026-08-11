import { useMemo } from 'react'
import { CamelidMark } from '../ui/CamelidMark'
import { StatusDot } from '../ui/StatusDot'
import { ThemeToggle } from '../ui/ThemeToggle'
import { Tooltip } from '../ui/Tooltip'
import { ConversationListItem } from './ConversationListItem'
import {
  IconAnalytics, IconApi, IconBolt, IconChart, IconChat, IconClose, IconHistory, IconMemory, IconModels,
  IconDownload, IconNetwork, IconNewChat, IconObservatory, IconReceipt, IconSearch, IconSettings, IconSidebar, IconSystem,
} from '../ui/icons'

const NAV_SECTIONS = [
  {
    label: 'Workspace',
    items: [
      { tab: 'chat', label: 'Chat', Icon: IconChat },
      { tab: 'workspace', label: 'Workspace', Icon: IconBolt },
      { tab: 'history', label: 'Chat history', Icon: IconHistory },
      { tab: 'memory', label: 'Memory', Icon: IconMemory },
    ],
  },
  {
    label: 'Models',
    items: [
      { tab: 'library', label: 'Models', Icon: IconModels },
      { tab: 'downloads', label: 'Downloaded models', Icon: IconDownload },
    ],
  },
  {
    label: 'Activity',
    items: [
      { tab: 'analytics', label: 'Analytics', Icon: IconAnalytics },
      { tab: 'telemetry', label: 'Telemetry', Icon: IconChart },
      { tab: 'observatory', label: 'Observatory', Icon: IconObservatory },
    ],
  },
  {
    label: 'System',
    items: [
      { tab: 'system', label: 'System', Icon: IconSystem },
      { tab: 'api', label: 'API', Icon: IconApi },
      { tab: 'compatibility', label: 'Compatibility', Icon: IconReceipt },
      { tab: 'cluster', label: 'Cluster', Icon: IconNetwork },
    ],
  },
]
const NAV_ITEMS = NAV_SECTIONS.flatMap((section) => section.items)

const RECENT_LIMIT = 6

const BUCKETS = ['Today', 'Yesterday', 'Previous 7 days', 'Earlier']
function startOfDay(d) { return new Date(d.getFullYear(), d.getMonth(), d.getDate()) }
function bucketFor(value) {
  if (!value) return 'Earlier'
  const diff = Math.floor((startOfDay(new Date()) - startOfDay(new Date(value))) / 86400000)
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
  runtime,
  themePreference,
  themeResolved,
  onCycleTheme,
}) {
  const grouped = useMemo(() => {
    const groups = new Map(BUCKETS.map((b) => [b, []]))
    filteredConversations.slice(0, RECENT_LIMIT).forEach((c) => groups.get(bucketFor(c.updated_at))?.push(c))
    return BUCKETS.map((label) => ({ label, items: groups.get(label) || [] })).filter((g) => g.items.length)
  }, [filteredConversations])

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
          <Tooltip content="New chat" placement="right">
            <button type="button" className="rail__icon-btn rail__icon-btn--accent" aria-label="New chat" onClick={showNewChatLanding}>
              <IconNewChat size={20} />
            </button>
          </Tooltip>
        </div>
        <nav className="rail__rail-nav" aria-label="Primary">
          {NAV_ITEMS.map(({ tab: t, label, Icon }) => (
            <Tooltip key={t} content={label} placement="right">
              <button type="button" className={`rail__icon-btn ${tab === t ? 'is-active' : ''}`} aria-label={label} aria-current={tab === t ? 'page' : undefined} onClick={() => setTab(t)}>
                <Icon size={20} />
              </button>
            </Tooltip>
          ))}
        </nav>
        <div className="rail__rail-bottom">
          <Tooltip content="Settings" placement="right">
            <button type="button" className={`rail__icon-btn ${tab === 'settings' ? 'is-active' : ''}`} aria-label="Settings" aria-current={tab === 'settings' ? 'page' : undefined} onClick={() => setTab('settings')}>
              <IconSettings size={20} />
            </button>
          </Tooltip>
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

      <button type="button" className="rail__new-chat" onClick={showNewChatLanding}>
        <IconNewChat size={18} />
        <span>New chat</span>
      </button>

      <div className="rail__search">
        <IconSearch size={16} />
        <input
          className="rail__search-input"
          aria-label="Search chats"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search chats"
        />
        {search.trim() !== '' && (
          <button type="button" className="rail__search-clear" aria-label="Clear search" onClick={() => setSearch('')}>
            <IconClose size={14} />
          </button>
        )}
      </div>

      <div className="rail__scroll">
        <div className="rail__section">
          <div className="rail__section-label">Recent</div>
          {grouped.length === 0 && (
            <p className="rail__empty">
              {search.trim() ? `No chats match “${search.trim()}”` : 'No conversations yet'}
            </p>
          )}
          {grouped.map((group) => (
            <div key={group.label} className="rail__group">
              <div className="rail__group-label">{group.label}</div>
              {group.items.map((conversation) => (
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
          {filteredConversations.length > 0 && (
            <button type="button" className="rail__nav-item" onClick={() => setTab('history')}>
              <IconHistory size={20} />
              <span>All chats</span>
            </button>
          )}
        </div>

        <nav className="rail__nav" aria-label="Primary">
          {NAV_SECTIONS.map(({ label: sectionLabel, items }) => (
            <div key={sectionLabel} className="rail__section">
              <div className="rail__section-label">{sectionLabel}</div>
              {items.map(({ tab: t, label, Icon }) => (
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
            </div>
          ))}
        </nav>
      </div>

      <div className="rail__footer">
        <ThemeToggle preference={themePreference} resolved={themeResolved} onCycle={onCycleTheme} />
        <button
          type="button"
          className={`rail__icon-btn ${tab === 'settings' ? 'is-active' : ''}`}
          aria-label="Settings"
          aria-current={tab === 'settings' ? 'page' : undefined}
          onClick={() => setTab('settings')}
        >
          <IconSettings size={20} />
        </button>
        <span className="rail__status"><StatusDot tone={statusTone} pulse={online} label={statusLabel} /></span>
      </div>
    </aside>
  )
}

export default SidebarRail
