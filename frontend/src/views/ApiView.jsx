import { displayCapabilityCopy, displayCapabilityId, exactRowSupportLanes, findCompatibilityHint, formatCapabilityStatus, isExactCompatibilityHint, isSupportedCapabilityStatus, rowSupportBoundaryCopy } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { getRuntimeRequestModelId, modelRuntimeIdMatches } from '../lib/modelState'
import { describeRuntimeStatus } from '../lib/runtimeStatus'
import { StatusDot } from '../components/ui/StatusDot'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { SupportContractSummary } from '../components/api/SupportContractSummary'
import { ApiWorkbench } from '../components/api/ApiWorkbench'
import { EmptyState } from '../components/ui/EmptyState'
import { IconApi } from '../components/ui/icons'

function supportLaneTitle(lane) {
  if (lane.key === 'template') return 'Template/Jinja readiness'
  if (lane.key === 'context') return 'Checked context readiness'
  return 'Throughput readiness'
}

export default function ApiView({ runtime, selectedModel, capabilities }) {
  const apiBase = runtime?.api_base || ''
  const modelId = getRuntimeRequestModelId(selectedModel, runtime, '<loaded-model-id>') || '<loaded-model-id>'
  const apiFeatures = capabilities?.api_features || []
  const supportedFeatures = apiFeatures.filter((feature) => isSupportedCapabilityStatus(feature.status))
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const selectedCompatibilityHint = selectedChatGate.hint || findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityTarget = isExactCompatibilityHint(selectedCompatibilityHint) ? selectedCompatibilityHint.target : null
  const selectedCompatibilitySupported = selectedChatGate.contractSupported
  const selectedSupportLanes = exactRowSupportLanes(selectedCompatibilityTarget, apiFeatures)
  const generationReady = Boolean(runtime?.generation_ready)
  const loadedNow = Boolean(runtime?.loaded_now)
  const selectedRuntimeMatches = modelRuntimeIdMatches(selectedModel, runtime)
  const q8Runtime = runtime?.q8_runtime
  const selectedExactRowReady = selectedChatGate.chatUnlocked
  const readinessPillCopy = selectedExactRowReady
    ? 'Ready for the selected model'
    : generationReady && selectedModel && !selectedRuntimeMatches
      ? 'A different model is loaded'
      : generationReady
        ? 'Engine ready — model not verified'
        : 'Load a supported model'
  const chatCompletionsCopy = selectedExactRowReady
    ? 'Ready — the selected model is loaded and verified.'
    : selectedCompatibilityTarget
      ? 'Chat stays locked until this model finishes loading and is verified.'
      : 'Chat stays locked until a supported model is selected and loaded.'
  const curlExample = selectedExactRowReady
    ? `# The selected model is loaded and ready\ncurl ${apiBase}/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '{\n    "model": "${modelId}",\n    "messages": [{"role": "user", "content": "Hello from Camelid"}],\n    "temperature": 0\n  }'`
    : `# Locked until the selected model is loaded and verified\n# loaded_now=${loadedNow ? 'true' : 'false'} generation_ready=${generationReady ? 'true' : 'false'} active_model_id=${runtime?.active_model_id || 'none'}\n# selected_model=${selectedCompatibilityTarget?.id || 'none'}`

  const runtimeStatus = describeRuntimeStatus(runtime)
  const headerStatus = generationReady ? 'Ready to generate' : loadedNow ? 'Model loaded, still preparing' : 'No model loaded'
  const selectedRowStat = selectedExactRowReady ? 'Ready' : selectedCompatibilityTarget ? 'Locked' : 'None'
  const selectedRowSub = selectedCompatibilitySupported ? 'verified' : selectedCompatibilityTarget ? 'known, not verified' : 'not verified'

  return (
    <section className="api-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconApi size={14} /> API</p>
          <h1>API</h1>
          <p className="cxv-sub">The local <code>/v1</code>-compatible API: <code>/api/capabilities</code> reports what has been verified on this machine, and <code>/v1/health</code> reports what is loaded right now.</p>
        </div>
        <div className="cxv-head__actions">
          <StatusDot tone={runtimeStatus.tone} pulse={generationReady} label={headerStatus} />
        </div>
      </header>

      {runtime?.status === 'offline' && (
        <EmptyState
          className="cx-empty--inline"
          icon={<IconApi size={22} />}
          title="Backend unreachable"
          description={`Nothing answered at ${apiBase || 'the configured API base'}. Start the engine from Settings (or fix the API base there); the sections below stay empty until the backend responds.`}
        />
      )}

      <div className="cxv-stat-grid">
        <div className="cxv-stat"><span>Runtime</span><strong>{runtimeStatus.label}</strong><small>{generationReady ? 'generation_ready=true' : loadedNow ? 'loaded_now=true' : runtime?.status === 'offline' ? 'backend unreachable' : 'no model loaded'}</small></div>
        <div className="cxv-stat"><span>Loaded model</span><strong>{loadedNow ? 'Active' : 'None'}</strong><small title={runtime?.active_model_id || 'nothing loaded'}>{runtime?.active_model_id || 'nothing loaded'}</small></div>
        <div className="cxv-stat"><span>Selected model</span><strong>{selectedRowStat}</strong><small>{selectedRowSub}</small></div>
        <div className="cxv-stat"><span>Local API</span><strong>{runtime?.status === 'offline' ? 'Unreachable' : apiBase ? 'Online' : 'Offline'}</strong><small>{apiBase || 'unavailable'}</small></div>
      </div>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head">
          <h2>Standard /v1-compatible surface</h2>
          <StatusDot tone={selectedExactRowReady ? 'ready' : 'warn'} label={readinessPillCopy} />
        </div>
        <p className="cxv-sub">Generation endpoints work once the engine is ready and the selected model is verified for this machine. <code>/api/capabilities</code> reports what has been verified; <code>/v1/health</code> reports what is loaded right now.</p>
        {/* The chat-completions gate sentence stays the single source for the
            generation-endpoint posture shown in the workbench cards below. */}
        <p className="cxv-sub">{chatCompletionsCopy}</p>
        <div className="sys-curl">
          <div className="sys-curl__head"><strong>Readiness-gated curl</strong><span className="cxv-tag">curl</span></div>
          <pre>{apiBase ? curlExample : 'Start the local runtime to see a readiness check for the selected model build.'}</pre>
        </div>
      </section>

      <ApiWorkbench
        apiBase={apiBase}
        modelId={modelId}
        backendOnline={runtime?.status !== 'offline' && Boolean(apiBase)}
        chatUnlocked={selectedExactRowReady}
        tokenizerAvailable={Boolean(runtime?.loaded_now)}
      />

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head"><h2>/api/capabilities summary</h2><span className="cxv-section__count">what’s verified</span></div>
        <p className="cxv-sub">These entries reflect what has actually been verified on this machine. Per-model detail lives on the Compatibility page.</p>

        <div className="cxv-grid cxv-grid--two">
          <SupportContractSummary capabilities={capabilities} />

          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Runtime readiness</strong>
            <p><b>loaded_now:</b> {loadedNow ? 'true' : 'false'}</p>
            <p><b>generation_ready:</b> {generationReady ? 'true' : 'false'}</p>
            <p><b>active_model_id:</b> {runtime?.active_model_id || 'none'}</p>
            <p><b>q8_policy:</b> {q8Runtime?.policy || 'unavailable'}</p>
            <p>{q8Runtime?.note || 'Q8 storage policy is reported by /v1/health when the runtime is online.'}</p>
          </div>
        </div>

        <div className="cxv-grid cxv-grid--two">
          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Selected model evidence</strong>
            {selectedCompatibilityTarget ? (
              <>
                <code className="a-code">{selectedCompatibilityTarget.id}</code>
                <p>{formatCapabilityStatus(selectedCompatibilityTarget.status)} · {selectedCompatibilityTarget.family} · {selectedCompatibilityTarget.quantization}</p>
                <p><b>Scope:</b> {displayCapabilityCopy(selectedCompatibilityTarget.support_scope || 'not advertised')}</p>
                <p><b>Readiness gate:</b> {displayCapabilityCopy(selectedCompatibilityTarget.frontend_readiness_gate)}</p>
                <p><b>Latest checked:</b> {formatCapabilityStatus(selectedCompatibilityTarget.latest_checked_bucket)} · {formatCapabilityStatus(selectedCompatibilityTarget.latest_checked_result)}</p>
                <p><b>Latest output:</b> {displayCapabilityCopy(selectedCompatibilityTarget.latest_checked_output || 'not advertised')}</p>
                <p><b>Full-support status:</b> {formatCapabilityStatus(selectedCompatibilityTarget.full_support_status || 'not advertised')}</p>
                {selectedSupportLanes.map((lane) => (
                  <p key={lane.key}><b>{supportLaneTitle(lane)}:</b> {lane.label}. {displayCapabilityCopy(lane.copy)}</p>
                ))}
                <p><b>Remaining support boundary:</b> {displayCapabilityCopy(rowSupportBoundaryCopy(selectedCompatibilityTarget, apiFeatures))}</p>
                <p>{displayCapabilityCopy(selectedCompatibilityTarget.evidence)}</p>
              </>
            ) : (
              <p>The selected model has no verified compatibility entry. Similar names or files aren’t treated as verification.</p>
            )}
          </div>

          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Selected model contract</strong>
            {selectedModel ? (
              <>
                <code className="a-code">{selectedModel.id}</code>
                {selectedCompatibilityTarget ? (
                  <>
                    <p>
                      <EvidenceChip
                        status={selectedCompatibilityTarget.status}
                        source={{ rowId: selectedCompatibilityTarget.id, detail: selectedCompatibilityTarget.support_scope ? displayCapabilityCopy(selectedCompatibilityTarget.support_scope) : undefined }}
                        size="sm"
                      />{' '}
                      <b>{selectedCompatibilityTarget.id}</b>
                    </p>
                    <p>{selectedCompatibilitySupported ? 'This model is verified for chat; it still needs to finish loading before chat unlocks.' : 'This model is recognized, but this configuration isn’t verified for chat.'}</p>
                  </>
                ) : (
                  <p>No exact compatibility row matched this selected model, so the API UI will not display family, quant-list, filename, or saved-path guesses as support evidence.</p>
                )}
              </>
            ) : (
              <p>No model selected. This list shows what has been verified, not everything on disk.</p>
            )}
          </div>
        </div>

        <div className="cxv-card cxv-card--flat sys-evidence">
          <strong>Supported API feature rows</strong>
          {supportedFeatures.length ? (
            <div className="sys-rows">
              {supportedFeatures.map((feature) => (
                <div key={feature.id} className="sys-row">
                  <div className="sys-row__head">
                    <span>{displayCapabilityId(feature.id)}</span>
                    <EvidenceChip status={feature.status} source={{ rowId: feature.id }} size="sm" />
                  </div>
                  <small>{displayCapabilityCopy(feature.notes || 'Advertised by /api/capabilities. These feature rows do not widen model support; chat still follows the selected model build and runtime readiness gate above.')}</small>
                </div>
              ))}
            </div>
          ) : (
            <p>No supported API feature rows advertised.</p>
          )}
        </div>
      </section>
    </section>
  )
}

