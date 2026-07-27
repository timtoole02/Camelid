import { memo } from 'react'
import { clampText } from '../lib/formatters'
import { getChatGateState } from '../lib/chatGate'
import { modelRuntimeIdMatches } from '../lib/modelState'
import { IconMenu } from './ui/icons'
import { StatusDot } from './ui/StatusDot'
import { EvidenceChip } from './ui/EvidenceChip'
import { CamelidMark } from './ui/CamelidMark'

const TITLES = {
  chat: 'Chat',
  code: 'Code',
  workspace: 'Workspace',
  library: 'Models',
  api: 'API',
  compatibility: 'Compatibility ledger',
  analytics: 'Analytics',
  telemetry: 'Session telemetry',
  history: 'Chat history',
  memory: 'Memory',
  system: 'System',
  settings: 'Settings',
  cluster: 'Cluster Topology',
  observatory: 'Inference Observatory',
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
  const tone = gate.chatUnlocked ? 'ready' : apiUnavailable ? 'offline' : runtime?.loaded_now ? 'warn' : 'neutral'
  const modelName = selectedModel?.name || 'No model selected'

  return (
    <header className={`topbar ${demoMode ? 'topbar--demo' : ''}`}>
      {onToggleSidebar && (
        <button type="button" className="topbar__menu" aria-label="Toggle sidebar" onClick={onToggleSidebar}>
          <IconMenu size={22} />
        </button>
      )}
      <CamelidMark size={18} className="topbar__mark" />
      <h1 className="topbar__title" title={tab === 'chat' && hasCustomTitle ? rawTitle : heading}>{heading}</h1>
      {(tab === 'chat' || tab === 'code') && !demoMode ? (
        <div className="topbar__mode-switch" role="group" aria-label="Interaction mode">
          <button type="button" className={tab === 'chat' ? 'is-active' : ''} aria-pressed={tab === 'chat'} onClick={() => setTab('chat')}>Chat</button>
          <button type="button" className={tab === 'code' ? 'is-active' : ''} aria-pressed={tab === 'code'} onClick={() => setTab('code')}>Code</button>
        </div>
      ) : null}
      <div className="topbar__spacer" />
      {!demoMode && (
        <div className="topbar__gate">
          {/* Support-gate claim: rendered by the EvidenceChip, sourced from the
              shared chat gate. Runtime + contract stay visible on every tab. */}
          <EvidenceChip
            status={gate.hint?.target?.status || ''}
            state={gate.contractSupported ? 'supported' : gate.hint?.target?.status ? null : 'unsupported'}
            label={gate.label}
            source={{ rowId: gate.hint?.target?.id, note: gate.copy }}
            size="sm"
            className="topbar__gate-chip"
          />
          <button
            type="button"
            className="topbar__model"
            onClick={() => setTab('library')}
            title={gate.chatUnlocked ? `${modelName} is ready` : 'Open Models to load or switch models'}
          >
            <StatusDot tone={tone} pulse={gate.chatUnlocked} />
            <span className="topbar__model-name">{clampText(modelName, 32)}</span>
          </button>
        </div>
      )}
    </header>
  )
}

export default memo(TopBar)
