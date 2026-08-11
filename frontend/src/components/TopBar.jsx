import { memo } from 'react'
import { clampText } from '../lib/formatters'
import { getChatGateState } from '../lib/chatGate'
import { modelRuntimeIdMatches } from '../lib/modelState'
import { IconMenu } from './ui/icons'
import { StatusDot } from './ui/StatusDot'
import { CamelidMark } from './ui/CamelidMark'

const TITLES = {
  chat: 'Chat',
  workspace: 'Workspace',
  library: 'Models',
  downloads: 'Downloaded models',
  api: 'API',
  compatibility: 'Compatibility',
  analytics: 'Analytics',
  telemetry: 'Telemetry',
  history: 'Chat history',
  memory: 'Memory',
  system: 'System',
  settings: 'Settings',
  cluster: 'Cluster',
  observatory: 'Observatory',
}

/* Slim top bar. Chat tab shows the conversation title + a compact model status
   chip; other tabs show the view title. The full model picker and support detail
   live in the chat composer's ModelStatusChip and the System/API views. */
function TopBar({
  tab,
  setTab,
  selectedConversationTitle,
  runtime,
  capabilities,
  selectedModelId,
  models = [],
  onToggleSidebar = null,
  mobileNavOpen = false,
  menuButtonRef = null,
  demoMode = false,
}) {
  const rawTitle = selectedConversationTitle?.trim()
  const hasCustomTitle = Boolean(rawTitle && rawTitle.toLowerCase() !== 'new conversation')
  const heading = tab === 'chat'
    ? (hasCustomTitle ? clampText(rawTitle, 64) : 'New chat')
    : (TITLES[tab] || 'Camelid')

  const selectedModel = models.find((m) => m.id === selectedModelId)
    || models.find((m) => modelRuntimeIdMatches(m, runtime))
  const gate = getChatGateState(capabilities, selectedModel, runtime)
  const apiUnavailable = runtime?.status === 'offline'
  const tone = gate.chatUnlocked || gate.embeddingReady ? 'ready' : apiUnavailable ? 'offline' : runtime?.loaded_now ? 'warn' : 'neutral'
  const modelName = selectedModel?.name || 'No model selected'
  /* Narrow screens swap the full name for this short form (dot + first word)
     so the view title keeps the room; see the 480px rules in shell.css. */
  const modelShortName = selectedModel ? clampText(modelName.split(/\s+/)[0], 14) : 'No model'

  return (
    <header className={`topbar ${demoMode ? 'topbar--demo' : ''}`}>
      {onToggleSidebar && (
        <button
          type="button"
          ref={menuButtonRef}
          className="topbar__menu"
          aria-label="Toggle sidebar"
          aria-expanded={mobileNavOpen}
          aria-controls="camelid-sidebar"
          onClick={onToggleSidebar}
        >
          <IconMenu size={22} />
        </button>
      )}
      <CamelidMark size={18} className="topbar__mark" />
      <h1 className="topbar__title" title={tab === 'chat' && hasCustomTitle ? rawTitle : heading}>{heading}</h1>
      <div className="topbar__spacer" />
      {!demoMode && (
        <div className="topbar__gate">
          {/* Model chip only. Support detail lives in the Models and System
              views; the header stays free of internal status strings. */}
          <button
            type="button"
            className="topbar__model"
            onClick={() => setTab('library')}
            title={gate.chatUnlocked
              ? `${modelName} is ready`
              : gate.embeddingReady
                ? `${modelName} is ready for embeddings, not Chat`
                : gate.embeddingOnly
                  ? `${modelName} is an embedding model — load it from Models`
                  : 'Open Models to load or switch models'}
          >
            <StatusDot tone={tone} pulse={gate.chatUnlocked} />
            <span className="topbar__model-name">{clampText(modelName, 32)}</span>
            <span className="topbar__model-short">{modelShortName}</span>
            {selectedModel && !gate.contractSupported && !apiUnavailable && (
              <span className="topbar__model-hint">· {gate.embeddingOnly ? 'Embedding only' : 'Unverified model'}</span>
            )}
          </button>
        </div>
      )}
    </header>
  )
}

export default memo(TopBar)
