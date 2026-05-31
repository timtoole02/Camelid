import { memo } from 'react'
import { clampText, formatPreview, formatSidebarDate } from '../lib/formatters'
import { compatibilityHintLabel, formatCapabilityStatus, frontendSupportContractCopy, getCurrentCompatibilityTarget } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { describeModelState, getModelStatusLabel, modelRuntimeIdMatches } from '../lib/modelState'

const GeminiSparkle = ({ className = '', size = 20 }) => (
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

const compactNavItems = navItems.slice(0, 3)

function exactTargetFromHint(hint) {
  return hint?.exact === true && hint.target?.id ? hint.target : null
}

function exactHintDetail(hint) {
  return exactTargetFromHint(hint) ? compatibilityHintLabel(hint) : ''
}

function TopBar({ tab, setTab, selectedConversationTitle, selectedConversationUpdatedAt, selectedConversationPreview, runtime, capabilities, selectedModelId, setSelectedModelId, models, demoMode = false, showNewChatLanding = null }) {
  const rawConversationTitle = selectedConversationTitle?.trim()
  const hasCustomConversationTitle = Boolean(rawConversationTitle && rawConversationTitle.toLowerCase() !== 'new conversation')
  const activeModel = models.find((model) => modelRuntimeIdMatches(model, runtime))
  const selectedModel = models.find((model) => model.id === selectedModelId)
  const activeChatGate = getChatGateState(capabilities, activeModel, runtime)
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const runtimeChatReady = activeChatGate.chatUnlocked
  const selectedModelRunnable = selectedChatGate.chatUnlocked
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
  const apiUnavailable = runtime?.status === 'offline'
  const chatReadinessTone = selectedModelRunnable ? 'ready' : apiUnavailable ? 'offline' : runtime?.loaded_now ? 'warm' : 'idle'
  const chatReadinessLabel = selectedModelRunnable
    ? 'Ready'
    : apiUnavailable
      ? 'Offline'
    : runtime?.loaded_now
      ? 'Checking'
      : 'Not ready'
  const chatCenterLabel = hasCustomConversationTitle ? clampText(rawConversationTitle, 64) : untitledConversationLabel
  const chatSupportLabel = selectedModelRunnable
    ? 'Local assistant UI'
    : apiUnavailable
      ? 'API connection needed'
      : 'Waiting on exact-row readiness'
  const modelOptionLabel = (model) => {
    const gate = getChatGateState(capabilities, model, runtime)
    if (gate.chatUnlocked) return `${model.name} · Ready`
    if (apiUnavailable) return `${model.name} · API offline`
    if (gate.runtimeReady) return `${model.name} · Support gated`
    if (gate.runtimeLoaded) return `${model.name} · Loading`
    return `${model.name} · Not loaded`
  }
  const hasSelectedModel = Boolean(selectedModel?.id)

  if (tab === 'chat') {
    return (
      <header className={`topbar topbar-chat ${demoMode ? 'topbar-demo' : ''}`}>
        <div className="topbar-chat-row">
          <div className="topbar-chat-brand topbar-chat-brand-stack topbar-chat-brand-elevated">
            <span className="topbar-chat-brand-kicker">Camelid chat</span>
            <strong className="topbar-brand-with-sparkle"><GeminiSparkle size={18} className="topbar-brand-sparkle-icon" /> Camelid</strong>
            <span>{chatSupportLabel}</span>
          </div>
          <div className="topbar-chat-center topbar-chat-center-stack topbar-chat-center-elevated" title={hasCustomConversationTitle ? rawConversationTitle : untitledConversationLabel}>
            <strong>{chatCenterLabel}</strong>
            <span>{selectedModelLabel}</span>
          </div>
          <div className="topbar-chat-actions topbar-chat-actions-elevated">
            {!demoMode && (
              <>
                {showNewChatLanding && (
                  <button type="button" className="ghost-button ghost-button-quiet topbar-new-chat-button" onClick={showNewChatLanding}>
                    New chat
                  </button>
                )}
                <div className={`topbar-chat-readiness ${chatReadinessTone}`} title={`${selectedModelSummary} ${runtimeGateDetail}`}>
                  <span className="topbar-chat-readiness-dot" aria-hidden="true" />
                  <span className="topbar-chat-readiness-caption">Model</span>
                  <select
                    className="topbar-select topbar-select-chat"
                    aria-label="Model for chat"
                    value={selectedModel?.id || selectedModelId || ''}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    disabled={!models.length}
                  >
                    {!hasSelectedModel && <option value="">Choose model</option>}
                    {models.length ? models.map((model) => (
                      <option key={model.id} value={model.id}>
                        {modelOptionLabel(model)}
                      </option>
                    )) : (
                      <option value="">No models</option>
                    )}
                  </select>
                  <span className="topbar-chat-readiness-label">{chatReadinessLabel}</span>
                </div>
              </>
            )}
          </div>
        </div>
        {!demoMode && (
          <div className="topbar-chat-support-strip" aria-label="Chat support summary">
            <div className={`topbar-chat-support-card ${selectedModelRunnable ? 'ready' : apiUnavailable ? 'offline' : runtime?.loaded_now ? 'warm' : ''}`}>
              <span>Runtime</span>
              <strong>{chatReadinessLabel}</strong>
              <small>{activeModel ? activeModelLabel : 'No model loaded'}</small>
            </div>
            <button type="button" className="topbar-chat-support-card topbar-chat-support-card-button" onClick={() => setTab('api')} title={`${supportGateLabel}. ${supportGateDetail}`}>
              <span>Support contract</span>
              <strong>{supportGateLabel}</strong>
              <small>{selectedModelRunnable ? 'Exact row unlocked for chat.' : supportGateDetail}</small>
            </button>
          </div>
        )}
        {!demoMode && (
          <div className="mobile-nav" aria-label="Primary navigation">
            {compactNavItems.map((item) => (
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
        <div className="topbar-chat-brand topbar-brand-with-sparkle"><GeminiSparkle size={18} className="topbar-brand-sparkle-icon" /> Camelid</div>
        <div className="topbar-chat-center topbar-page-center" title={heading}>{heading}</div>
        <div className="topbar-chat-actions">
          {demoMode ? (
            <button type="button" className="ghost-button ghost-button-quiet" onClick={() => setTab('chat')}>Back to chat</button>
          ) : (
            <label className="topbar-chat-picker" title={selectedModel ? getModelStatusLabel(selectedModel) : 'Choose what new chats should use next.'}>
              <select className="topbar-select topbar-select-chat" aria-label="Use for next chat" value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
                {!hasSelectedModel && <option value="">Choose model</option>}
                {models.map((model) => {
                  const runnable = getChatGateState(capabilities, model, runtime).chatUnlocked
                  return (
                    <option key={model.id} value={model.id} disabled={!runnable}>
                      {modelOptionLabel(model)}
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
          <div className={`status-pill topbar-status-pill topbar-status-pill-compact ${runtimeChatReady ? 'ready' : runtime?.loaded_now ? 'warm' : ''}`} title={`${activeModelLabel} · ${runtimeGateDetail}`}>
            <span className="topbar-status-label">Runtime chat gate</span>
            <strong>{clampText(activeModelLabel, 32)}</strong>
          </div>
          <div className="status-pill topbar-status-pill topbar-status-pill-compact topbar-status-pill-wide" title={selectedModelSummary}>
            <span className="topbar-status-label">Model</span>
            <strong>{clampText(selectedModelLabel, 32)}</strong>
          </div>
          <button type="button" className={`status-pill topbar-status-pill topbar-status-pill-compact topbar-status-pill-wide topbar-status-button ${capabilities ? 'ready' : 'warm'}`} onClick={() => setTab('api')} title={`${supportGateLabel}. ${supportGateDetail}`}>
            <span className="topbar-status-label">Support contract</span>
            <strong>{clampText(supportGateLabel, 34)}</strong>
          </button>
        </div>
      )}
      {!demoMode && (
        <div className="mobile-nav" aria-label="Primary navigation">
          {compactNavItems.map((item) => (
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
