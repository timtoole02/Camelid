import { memo } from 'react'
import { clampText, formatPreview, formatSidebarDate } from '../lib/formatters'
import { compatibilityHintLabel, formatCapabilityStatus, frontendSupportContractCopy, getCurrentCompatibilityTarget } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { describeModelState, getModelStatusLabel, modelRuntimeIdMatches } from '../lib/modelState'

const titles = {
  chat: 'Chat',
  library: 'Models',
  api: 'API',
  analytics: 'Analytics',
  history: 'History',
  memory: 'Memory',
  system: 'System',
}

const navItems = [
  { id: 'chat', label: 'Chat' },
  { id: 'library', label: 'Models' },
  { id: 'api', label: 'API' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'history', label: 'History' },
  { id: 'memory', label: 'Memory' },
  { id: 'system', label: 'System' },
]

function exactTargetFromHint(hint) {
  return hint?.exact === true && hint.target?.id ? hint.target : null
}

function exactHintDetail(hint) {
  return exactTargetFromHint(hint) ? compatibilityHintLabel(hint) : ''
}

function TopBar({ tab, setTab, selectedConversationTitle, selectedConversationUpdatedAt, selectedConversationPreview, runtime, capabilities, selectedModelId, setSelectedModelId, models, demoMode = false }) {
  const rawConversationTitle = selectedConversationTitle?.trim()
  const hasCustomConversationTitle = Boolean(rawConversationTitle && rawConversationTitle.toLowerCase() !== 'new conversation')
  const activeModel = models.find((model) => modelRuntimeIdMatches(model, runtime))
  const selectedModel = models.find((model) => model.id === selectedModelId)
  const activeChatGate = getChatGateState(capabilities, activeModel, runtime)
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const runtimeChatReady = activeChatGate.chatUnlocked
  const untitledConversationLabel = selectedConversationTitle
    ? `${formatPreview(selectedConversationPreview, 42)} · ${formatSidebarDate(selectedConversationUpdatedAt) || 'New chat'}`
    : runtimeChatReady
      ? 'Ready when you are'
      : 'Waiting on model readiness'
  const heading = tab === 'chat'
    ? (hasCustomConversationTitle ? clampText(rawConversationTitle, 72) : '')
    : titles[tab] || 'Camelid'
  const activeModelLabel = activeModel?.name || 'Nothing loaded now'
  const selectedModelLabel = selectedModel?.name || 'Nothing chosen for next chat'
  const selectedModelSummary = selectedModel ? describeModelState(selectedModel) : 'Choose the model you want Camelid to use next.'
  const exactCompatibilityDetail = exactHintDetail(activeChatGate.hint) || exactHintDetail(selectedChatGate.hint)
  const currentCompatibilityTarget = exactTargetFromHint(activeChatGate.hint)
    || exactTargetFromHint(selectedChatGate.hint)
    || getCurrentCompatibilityTarget(capabilities)
  const supportGateLabel = capabilities ? frontendSupportContractCopy(capabilities) : 'No /api/capabilities contract'
  const supportGateDetail = exactCompatibilityDetail
    || (currentCompatibilityTarget
      ? `${currentCompatibilityTarget.id}: ${formatCapabilityStatus(currentCompatibilityTarget.status)}`
      : 'Open the API contract before treating any model family or quant as supported.')
  const runtimeGateDetail = `loaded_now=${runtime?.loaded_now ? 'true' : 'false'} · generation_ready=${runtime?.generation_ready ? 'true' : 'false'} · exact_compatibility_row=${activeChatGate.contractSupported ? 'true' : 'false'}`

  if (tab === 'chat') {
    return (
      <header className={`topbar topbar-chat ${demoMode ? 'topbar-demo' : ''}`}>
        <div className="topbar-chat-row">
          <div className="topbar-chat-brand">Camelid</div>
          <div className="topbar-chat-center" title={hasCustomConversationTitle ? rawConversationTitle : untitledConversationLabel}>
            {hasCustomConversationTitle ? clampText(rawConversationTitle, 64) : untitledConversationLabel}
          </div>
          <div className="topbar-chat-actions" />
        </div>
        {!demoMode && (
          <div className="mobile-nav" aria-label="Primary navigation">
            {navItems.map((item) => (
              <button key={item.id} className={`mobile-nav-item ${tab === item.id ? 'active' : ''}`} aria-current={tab === item.id ? 'page' : undefined} onClick={() => setTab(item.id)}>
                {item.label}
              </button>
            ))}
          </div>
        )}
      </header>
    )
  }

  return (
    <header className={`topbar topbar-page ${demoMode ? 'topbar-demo' : ''}`}>
      <div className="topbar-page-row">
        <div className="topbar-chat-brand">Camelid</div>
        <div className="topbar-chat-center topbar-page-center" title={heading}>{heading}</div>
        <div className="topbar-chat-actions">
          {demoMode ? (
            <button type="button" className="ghost-button ghost-button-quiet" onClick={() => setTab('chat')}>Back to chat</button>
          ) : (
            <label className="topbar-chat-picker" title={selectedModel ? getModelStatusLabel(selectedModel) : 'Choose what new chats should use next.'}>
              <select className="topbar-select topbar-select-chat" aria-label="Use for next chat" value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
                {models.map((model) => {
                  const runnable = getChatGateState(capabilities, model, runtime).chatUnlocked
                  return (
                    <option key={model.id} value={model.id} disabled={!runnable}>
                      {model.name}
                    </option>
                  )
                })}
              </select>
            </label>
          )}
        </div>
      </div>
      {!demoMode && tab !== 'library' && (
        <div className="topbar-status-strip" aria-label="Model status">
          <div className={`status-pill topbar-status-pill ${runtimeChatReady ? 'ready' : runtime?.loaded_now ? 'warm' : ''}`} title={`${activeModelLabel} · ${runtimeGateDetail}`}>
            <span className="topbar-status-label">Runtime chat gate</span>
            <strong>{clampText(activeModelLabel, 32)}</strong>
            <small>{runtimeGateDetail}</small>
          </div>
          <div className="status-pill topbar-status-pill topbar-status-pill-wide" title={selectedModelSummary}>
            <span className="topbar-status-label">Next chat</span>
            <strong>{clampText(selectedModelLabel, 36)}</strong>
          </div>
          <button type="button" className={`status-pill topbar-status-pill topbar-status-pill-wide topbar-status-button ${capabilities ? 'ready' : 'warm'}`} onClick={() => setTab('api')} title={`${supportGateLabel}. ${supportGateDetail}`}>
            <span className="topbar-status-label">Support contract</span>
            <strong>{clampText(supportGateLabel, 42)}</strong>
            <small>{clampText(supportGateDetail, 58)}</small>
          </button>
        </div>
      )}
      {!demoMode && (
        <div className="mobile-nav" aria-label="Primary navigation">
          {navItems.map((item) => (
            <button key={item.id} className={`mobile-nav-item ${tab === item.id ? 'active' : ''}`} aria-current={tab === item.id ? 'page' : undefined} onClick={() => setTab(item.id)}>
              {item.label}
            </button>
          ))}
        </div>
      )}
    </header>
  )
}

export default memo(TopBar)
