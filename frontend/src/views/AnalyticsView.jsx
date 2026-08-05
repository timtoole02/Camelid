import { displayCapabilityCopy, displayCapabilityId, formatCapabilityStatus, frontendSupportContractCopy, getCurrentCompatibilityTarget, guardedCapabilityCopy, isGuardedCapabilityStatus } from '../lib/capabilities'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { formatCompactNumber, formatDate, formatRate } from '../lib/formatters'
import { isRunnableModel } from '../lib/modelState'
import { EmptyState } from '../components/ui/EmptyState'
import { StatusDot } from '../components/ui/StatusDot'
import { IconAnalytics, IconChart } from '../components/ui/icons'
import { RuntimeMemoryPanel } from '../components/analytics/RuntimeMemoryPanel'

function startOfDay(date) { const next = new Date(date); next.setHours(0, 0, 0, 0); return next }
function dayKey(date) { return startOfDay(date).toISOString().slice(0, 10) }
function labelDay(date) { return new Intl.DateTimeFormat(undefined, { weekday: 'short' }).format(date) }
function safeAverage(values) { return values.length ? values.reduce((s, v) => s + v, 0) / values.length : null }
function conversationLabel(conversation) {
  const raw = conversation?.title?.trim()
  return raw && raw.toLowerCase() !== 'new conversation' ? raw : 'Untitled chat'
}
function summarizeGuardedTargets(targets = []) {
  const guarded = targets.filter((target) => isGuardedCapabilityStatus(target.status))
  if (!guarded.length) return 'No planned or guarded compatibility rows advertised.'
  return guarded.slice(0, 3).map((target) => `${target.id}: ${formatCapabilityStatus(target.status)}`).join(' · ')
}

export default function AnalyticsView({ apiBase, conversations, models, runtime, capabilities }) {
  const now = new Date()
  const sevenDays = Array.from({ length: 7 }, (_, index) => {
    const date = new Date(now)
    date.setDate(now.getDate() - (6 - index))
    return { key: dayKey(date), label: labelDay(date), date, prompts: 0, replies: 0 }
  })
  const dayMap = new Map(sevenDays.map((day) => [day.key, day]))

  const modelStats = new Map()
  let totalMessages = 0
  let totalAssistantReplies = 0
  let totalUserPrompts = 0
  let activeToday = 0
  let latestActivityAt = null

  for (const conversation of conversations) {
    const updatedAt = conversation?.updated_at ? new Date(conversation.updated_at) : null
    if (updatedAt && startOfDay(updatedAt).getTime() === startOfDay(now).getTime()) activeToday += 1
    if (updatedAt && (!latestActivityAt || updatedAt > latestActivityAt)) latestActivityAt = updatedAt

    const modelId = conversation.model_id || 'unknown'
    const base = modelStats.get(modelId) || {
      id: modelId,
      name: models.find((model) => model.id === modelId)?.name || modelId,
      prompts: 0, replies: 0, conversations: new Set(), lastUsedAt: null, outRates: [], inRates: [],
    }
    base.conversations.add(conversation.id)

    for (const message of conversation.messages || []) {
      totalMessages += 1
      const createdAt = message?.created_at ? new Date(message.created_at) : updatedAt
      if (createdAt && (!base.lastUsedAt || createdAt > base.lastUsedAt)) base.lastUsedAt = createdAt
      if (createdAt && (!latestActivityAt || createdAt > latestActivityAt)) latestActivityAt = createdAt

      const bucket = createdAt ? dayMap.get(dayKey(createdAt)) : null
      if (message.role === 'user') {
        totalUserPrompts += 1; base.prompts += 1
        if (bucket) bucket.prompts += 1
      }
      if (message.role === 'assistant') {
        totalAssistantReplies += 1; base.replies += 1
        if (bucket) bucket.replies += 1
        if (message.tokens_out_per_sec !== null && message.tokens_out_per_sec !== undefined) base.outRates.push(Number(message.tokens_out_per_sec))
        if (message.tokens_in_per_sec !== null && message.tokens_in_per_sec !== undefined) base.inRates.push(Number(message.tokens_in_per_sec))
      }
    }
    modelStats.set(modelId, base)
  }

  const modelRows = [...modelStats.values()]
    .map((row) => ({ ...row, conversationCount: row.conversations.size, avgOutRate: safeAverage(row.outRates), avgInRate: safeAverage(row.inRates) }))
    .sort((left, right) => (right.replies !== left.replies ? right.replies - left.replies : (right.prompts + right.replies) - (left.prompts + left.replies)))

  const totalTrackedEvents = Math.max(1, totalAssistantReplies + totalUserPrompts)
  const topModels = modelRows.slice(0, 5)
  const chatReadyModels = models.filter(isRunnableModel).length
  const averageReplyRate = safeAverage(modelRows.flatMap((row) => row.outRates))
  const busiestDay = [...sevenDays].sort((left, right) => (right.prompts + right.replies) - (left.prompts + left.replies))[0]
  const recentThreads = [...conversations]
    .sort((left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime())
    .slice(0, 4)
  const supportContractCurrentGate = frontendSupportContractCopy(capabilities)
  const currentTarget = getCurrentCompatibilityTarget(capabilities)
  const compatibilityTargets = capabilities?.model_compatibility || []
  const guardedTargets = compatibilityTargets.filter((target) => isGuardedCapabilityStatus(target.status))
  const guardedFeatures = (capabilities?.api_features || []).filter((feature) => isGuardedCapabilityStatus(feature.status))
  const maxWeekTotal = Math.max(...sevenDays.map((d) => d.prompts + d.replies), 1)

  return (
    <section className="analytics-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconAnalytics size={14} /> Analytics</p>
          <h1>Usage</h1>
          <p className="cxv-sub">A calm internal view of prompts, replies, and which local models do the work. Telemetry shows usage — not support; family, quant, and API readiness still come from /api/capabilities.</p>
        </div>
        <div className="cxv-head__actions">
          <StatusDot tone={runtime?.generation_ready ? 'ready' : 'warn'} pulse={runtime?.generation_ready} label={runtime?.generation_ready ? 'Generation-ready' : runtime?.loaded_now ? 'Loaded, not ready' : 'No ready model'} />
        </div>
      </header>

      <div className="cxv-stat-grid">
        <div className="cxv-stat"><span>Conversations</span><strong>{formatCompactNumber(conversations.length)}</strong><small>{activeToday} active today</small></div>
        <div className="cxv-stat"><span>Prompts</span><strong>{formatCompactNumber(totalUserPrompts)}</strong><small>{formatCompactNumber(totalMessages)} messages</small></div>
        <div className="cxv-stat"><span>Replies</span><strong>{formatCompactNumber(totalAssistantReplies)}</strong><small>{totalTrackedEvents ? `${Math.round((totalAssistantReplies / totalTrackedEvents) * 100)}% of activity` : '—'}</small></div>
        <div className="cxv-stat"><span>Avg speed</span><strong>{averageReplyRate ? formatRate(averageReplyRate).replace(' tokens/sec', '') : '—'}</strong><small>{averageReplyRate ? 'tokens/sec out' : 'after first reply'}</small></div>
        <div className="cxv-stat"><span>Chat-ready</span><strong>{chatReadyModels}</strong><small>{runtime?.generation_ready ? 'runtime is green' : 'load a local GGUF'}</small></div>
      </div>

      <RuntimeMemoryPanel apiBase={apiBase} />

      <div className="cxv-grid cxv-grid--two">
        <section className="cxv-card cxv-panel">
          <div className="cxv-section__head"><h2>Model leaderboard</h2><span className="cxv-section__count">by replies</span></div>
          {topModels.length === 0 ? (
            <EmptyState icon={<IconChart size={24} />} title="No usage yet" description="Send a few local chats and this board fills in." />
          ) : (
            <div className="a-lead">
              {topModels.map((model) => {
                const share = ((model.prompts + model.replies) / totalTrackedEvents) * 100
                return (
                  <article key={model.id} className="a-lead__row">
                    <div className="a-lead__top">
                      <div className="a-lead__name"><strong>{model.name}</strong><span>{model.id}</span></div>
                      <div className="a-lead__metrics"><strong>{model.replies}</strong> replies<span className="cxv-dot">·</span>{model.avgOutRate ? formatRate(model.avgOutRate) : 'no speed'}</div>
                    </div>
                    <div className="a-bar"><div className="a-bar__fill" style={{ width: `${Math.max(6, Math.min(100, share))}%` }} /></div>
                    <div className="a-lead__foot"><small>{model.conversationCount} conversations</small><small>{model.lastUsedAt ? `Last ${formatDate(model.lastUsedAt.toISOString())}` : 'No recent activity'}</small></div>
                  </article>
                )
              })}
            </div>
          )}
        </section>

        <section className="cxv-card cxv-panel">
          <div className="cxv-section__head"><h2>Weekly flow</h2><span className="cxv-section__count">7 days</span></div>
          <div className="a-week">
            {sevenDays.map((day) => {
              const total = day.prompts + day.replies
              return (
                <div key={day.key} className="a-week__day">
                  <div className="a-week__bars" title={`${day.prompts} prompts · ${day.replies} replies`}>
                    <div className="a-week__bar a-week__bar--prompt" style={{ height: `${Math.max(4, (day.prompts / maxWeekTotal) * 100)}%` }} />
                    <div className="a-week__bar a-week__bar--reply" style={{ height: `${Math.max(4, (day.replies / maxWeekTotal) * 100)}%` }} />
                  </div>
                  <strong>{day.label}</strong>
                  <span>{total}</span>
                </div>
              )
            })}
          </div>
          <div className="a-week__legend">
            <span><i className="a-swatch a-swatch--prompt" /> Prompts</span>
            <span><i className="a-swatch a-swatch--reply" /> Replies</span>
            <span className="a-week__busiest">Busiest: {busiestDay && (busiestDay.prompts + busiestDay.replies) ? `${busiestDay.label} · ${busiestDay.prompts + busiestDay.replies}` : 'none yet'}</span>
          </div>
        </section>
      </div>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head"><h2>By model</h2><span className="cxv-section__count">usage · speed · recency</span></div>
        {modelRows.length === 0 ? (
          <EmptyState icon={<IconChart size={24} />} title="Nothing to compare yet" description="Model usage will show here once you start chatting locally." />
        ) : (
          <div className="a-table">
            <div className="a-table__row a-table__head">
              <span>Model</span><span>Chats</span><span>Prompts</span><span>Replies</span><span>Avg out</span><span>Last used</span>
            </div>
            {modelRows.map((model) => (
              <div key={model.id} className="a-table__row">
                <strong title={model.id}>{model.name}</strong>
                <span>{model.conversationCount}</span>
                <span>{model.prompts}</span>
                <span>{model.replies}</span>
                <span>{model.avgOutRate ? formatRate(model.avgOutRate) : '—'}</span>
                <span>{model.lastUsedAt ? formatDate(model.lastUsedAt.toISOString()) : '—'}</span>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head"><h2>Recent threads</h2><span className="cxv-section__count">latest activity</span></div>
        {recentThreads.length === 0 ? (
          <EmptyState icon={<IconAnalytics size={24} />} title="No conversations yet" description="Recent chats will appear here." />
        ) : (
          <div className="a-threads">
            {recentThreads.map((conversation) => {
              const messageCount = conversation.messages?.length || 0
              const assistantReplies = conversation.messages?.filter((m) => m.role === 'assistant').length || 0
              const modelName = models.find((m) => m.id === conversation.model_id)?.name || conversation.model_id || 'No model recorded'
              return (
                <article key={conversation.id} className="a-thread">
                  <div className="a-thread__copy"><strong>{conversationLabel(conversation)}</strong><span>{modelName}</span></div>
                  <div className="a-thread__meta"><small>{messageCount} msgs</small><small>{assistantReplies} replies</small><small>{formatDate(conversation.updated_at)}</small></div>
                </article>
              )
            })}
          </div>
        )}
      </section>

      <details className="cxv-disclosure">
        <summary>Support-contract boundary — usage analytics never expand model support</summary>
        <div className="cxv-disclosure__body">
          <p className="cxv-sub">Aligned with the same /api/capabilities + COMPATIBILITY.md contract used by Chat, Models, API, and System.</p>
          <div className="cxv-grid cxv-grid--two">
            <div className="cxv-card cxv-card--flat">
              <strong>Current validated gate</strong>
              {currentTarget ? (
                <>
                  <code className="a-code">{currentTarget.id}</code>
                  <p className="cxv-sub">{formatCapabilityStatus(currentTarget.status)} · {currentTarget.family} · {currentTarget.quantization}</p>
                  <p className="cxv-sub">{currentTarget.frontend_load_path_verified ? `Frontend load: ${formatCapabilityStatus(currentTarget.frontend_load_path_verified)}` : 'Frontend load evidence not advertised.'}</p>
                </>
              ) : <p className="cxv-sub">/api/capabilities is unavailable, so analytics will not infer a supported model target.</p>}
            </div>
            <div className="cxv-card cxv-card--flat">
              <strong>Runtime gate</strong>
              <p className="cxv-sub"><b>active_model_id:</b> {runtime?.active_model_id || 'none'}</p>
              <p className="cxv-sub"><b>loaded_now:</b> {runtime?.loaded_now ? 'true' : 'false'} · <b>generation_ready:</b> {runtime?.generation_ready ? 'true' : 'false'}</p>
              <p className="cxv-sub">{supportContractCurrentGate}</p>
            </div>
            <div className="cxv-card cxv-card--flat">
              <strong>Guarded compatibility rows</strong>
              <p className="cxv-sub">{summarizeGuardedTargets(compatibilityTargets)}</p>
              {guardedTargets.slice(0, 4).map((target) => (
                <p className="cxv-sub" key={target.id}><EvidenceChip status={target.status} source={{ rowId: target.id }} size="sm" /> <b>{target.id}</b> — {displayCapabilityCopy(target.next_step || 'Guarded until evidence lands.')}</p>
              ))}
            </div>
            <div className="cxv-card cxv-card--flat">
              <strong>Unsupported / partial API rows</strong>
              {guardedFeatures.length ? guardedFeatures.slice(0, 4).map((feature) => (
                <p className="cxv-sub" key={feature.id}><EvidenceChip status={feature.status} source={{ rowId: feature.id }} size="sm" /> <b>{displayCapabilityId(feature.id)}</b> — {displayCapabilityCopy(guardedCapabilityCopy(feature, 'Analytics-driven shortcuts and UI controls'))}</p>
              )) : <p className="cxv-sub">No unsupported or partial API rows advertised.</p>}
            </div>
          </div>
        </div>
      </details>
    </section>
  )
}
