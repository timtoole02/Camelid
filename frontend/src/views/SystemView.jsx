import { EngineMetricsPanel } from '../components/analytics/EngineMetricsPanel'
import { displayCapabilityCopy, displayCapabilityId, exactRowSupportLanes, findCompatibilityHint, formatCapabilityStatus, isExactCompatibilityHint, isSupportedCapabilityStatus } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { describeModelState } from '../lib/modelState'
import { describeExecutionPlan } from '../lib/executionPlan'
import { describeRuntimeStatus } from '../lib/runtimeStatus'
import { formatBytes } from '../lib/formatters'
import { StatusDot } from '../components/ui/StatusDot'
import { EvidenceChip } from '../components/ui/EvidenceChip'
import { SupportContractSummary } from '../components/api/SupportContractSummary'
import { EmptyState } from '../components/ui/EmptyState'
import { IconSystem } from '../components/ui/icons'

function runtimeReadinessLabel(runtime) {
  if (runtime?.generation_ready) return 'Loaded for local generation'
  if (runtime?.loaded_now) return 'Loaded, checking generation readiness'
  return 'Waiting for a generation-ready model'
}

function supportLaneTitle(lane) {
  if (lane.key === 'template') return 'Template/Jinja readiness'
  if (lane.key === 'context') return 'Checked context readiness'
  return 'Throughput readiness'
}

export default function SystemView({ runtime, selectedModel, capabilities, metricsApiBase = '' }) {
  const runtimePill = runtimeReadinessLabel(runtime)
  const selectedModelName = selectedModel?.name || 'No next-chat model selected'
  const apiBase = runtime?.api_base || 'Local API unavailable'
  // Version comes from /v1/health, so it names the engine actually answering rather than
  // whatever this bundle was built alongside. Unknown until health responds.
  const engineVersion = runtime?.version || null
  const engineBuild = runtime?.build || null
  const engineVersionLabel = engineVersion ? `v${engineVersion}` : 'Unknown until /v1/health responds'
  // A build identity that is not just the version restated means this engine is not a
  // released binary — worth showing rather than hiding behind a matching version number.
  const engineBuildDetail = engineBuild && engineBuild !== engineVersion && engineBuild !== `v${engineVersion}`
    ? engineBuild
    : null
  const engineLabel = runtime?.engine
    ? (engineVersion ? `${runtime.engine} v${engineVersion}` : runtime.engine)
    : 'engine unknown'
  const execution = describeExecutionPlan(runtime)
  const ghostExecution = runtime?.backend === 'gemma4-runtime' && runtime?.gemma4_serve_lane === 'ghost_moe'
  const apiFeatures = capabilities?.api_features || []
  const supportedFeatures = apiFeatures.filter((feature) => isSupportedCapabilityStatus(feature.status))
  const selectedChatGate = getChatGateState(capabilities, selectedModel, runtime)
  const selectedCompatibilityHint = selectedChatGate.hint || findCompatibilityHint(capabilities, selectedModel)
  const selectedCompatibilityTarget = isExactCompatibilityHint(selectedCompatibilityHint) ? selectedCompatibilityHint.target : null
  const selectedSupportLanes = exactRowSupportLanes(selectedCompatibilityTarget, apiFeatures)
  const selectedExactRowReady = selectedChatGate.chatUnlocked
  const endpointReadinessLabel = selectedExactRowReady
    ? 'Local API ready for the selected model'
    : selectedChatGate.runtimeReady
      ? 'Engine ready — model not verified yet'
      : runtime?.generation_ready
        ? 'A different model is loaded'
        : runtime?.loaded_now
          ? 'Model loaded, still preparing'
          : 'Load a supported model'
  const chatCompletionsCopy = selectedExactRowReady
    ? 'Ready — the selected model is loaded and verified for chat.'
    : selectedCompatibilityTarget
      ? 'Chat unlocks once this model finishes loading and is verified.'
      : 'Chat unlocks once a supported model is selected and loaded.'
  const q8Runtime = runtime?.q8_runtime
  const q8RuntimeLabel = q8Runtime?.retain_q8_blocks
    ? 'Retained Q8 blocks'
    : q8Runtime?.lazy_q8_linear
      ? 'Lazy Q8 policy'
      : q8Runtime
        ? 'Eager CPU materialization'
        : 'Q8 policy unavailable'
  const q8RuntimeDetail = q8Runtime
    ? `${q8Runtime.policy}${Number.isFinite(q8Runtime.file_cache_bytes) ? ` · cache ${formatBytes(q8Runtime.file_cache_bytes)}` : ''}`
    : 'Start the local runtime to inspect Q8 storage policy.'

  const runtimeStatus = describeRuntimeStatus(runtime)
  const gateStat = selectedExactRowReady ? 'Ready' : selectedChatGate.runtimeReady ? 'Gated' : 'Blocked'
  const gateStatSub = selectedExactRowReady ? 'chat + API unlocked' : selectedChatGate.runtimeReady ? 'model not verified yet' : 'supported model required'

  return (
    <section className="system-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconSystem size={14} /> System</p>
          <h1>System</h1>
          <p className="cxv-sub">Runtime health, model readiness, and local API connection details.</p>
        </div>
        <div className="cxv-head__actions">
          <StatusDot tone={runtimeStatus.tone} pulse={runtime?.generation_ready} label={runtimePill} />
        </div>
      </header>

      {runtime?.status === 'offline' && (
        <EmptyState
          className="cx-empty--inline"
          icon={<IconSystem size={22} />}
          title="Backend unreachable"
          description={`Nothing answered at ${runtime?.api_base || 'the configured API base'}. Start the local runtime (cargo run -- serve) or fix the API base in Settings; runtime health and the support gate below stay unknown until /v1/health responds.`}
        />
      )}

      <div className="cxv-stat-grid cxv-stat-grid--five">
        <div className="cxv-stat"><span>Runtime</span><strong>{runtimeStatus.label}</strong><small title={engineBuildDetail || undefined}>{engineLabel}</small></div>
        <div className="cxv-stat"><span>Ready to generate</span><strong>{runtime?.generation_ready ? 'Yes' : 'No'}</strong><small>{runtimePill}</small></div>
        <div className="cxv-stat"><span>Loaded model</span><strong>{runtime?.loaded_now ? 'Active' : 'None'}</strong><small title={runtime?.loaded_now ? runtime?.active_model_id : 'Nothing loaded'}>{runtime?.loaded_now ? runtime?.active_model_id : 'Nothing loaded'}</small></div>
        <div className="cxv-stat"><span>Chat readiness</span><strong>{gateStat}</strong><small>{gateStatSub}</small></div>
        <div className="cxv-stat"><span>Local API</span><strong>{runtime?.api_base ? 'Online' : 'Offline'}</strong><small>{apiBase}</small></div>
      </div>

      <div className="cxv-grid cxv-grid--two">
        <section className="cxv-card cxv-panel">
          <div className="cxv-section__head"><h2>Runtime</h2><span className="cxv-section__count">local engine</span></div>
          <div className="sys-defs">
            <div><span>Runtime state</span><strong>{runtime?.generation_ready ? 'Generation-ready' : runtime?.loaded_now ? 'Loaded, not generation-ready' : runtime?.status === 'offline' ? 'Backend offline' : 'Online, no model loaded'}</strong></div>
            <div><span>Local engine</span><strong>{runtime?.engine || 'Unknown'}</strong></div>
            <div><span>Engine version</span><strong title={engineBuildDetail || undefined}>{engineVersionLabel}</strong></div>
            {engineBuildDetail && (
              <div><span>Build</span><strong>{engineBuildDetail}</strong></div>
            )}
            <div><span>Loaded model</span><strong>{runtime?.loaded_now ? runtime?.active_model_id : 'Nothing loaded'}</strong></div>
            <div><span>Generation ready</span><strong>{runtime?.generation_ready ? 'Yes' : 'No'}</strong></div>
            <div><span>{ghostExecution ? 'Available Ghost acceleration' : 'Selected device at load'}</span><strong>{execution.device}</strong></div>
            <div><span>{ghostExecution ? 'Ghost serving lane' : 'Selected backend at load'}</span><strong>{execution.backend}</strong></div>
            <div><span>Selected model</span><strong>{selectedExactRowReady ? 'Ready for chat/API' : selectedChatGate.runtimeReady ? 'Engine ready; not verified' : selectedChatGate.label}</strong></div>
            <div><span>Q8 storage</span><strong>{q8RuntimeLabel}</strong></div>
            <div><span>Next chat selection</span><strong>{selectedModelName}</strong></div>
            <div><span>API base</span><strong>{apiBase}</strong></div>
          </div>
        </section>

        <section className="cxv-card cxv-panel">
          <div className="cxv-section__head"><h2>Handling locally</h2><span className="cxv-section__count">on-device</span></div>
          <ul className="sys-feed">
            <li>Persistent conversations are already available from local storage.</li>
            <li>Saved memory remains on-device and can be recalled in later chats.</li>
            <li>{execution.summary}</li>
            <li>Q8 runtime policy: {q8RuntimeDetail}. {q8Runtime?.note || ''}</li>
            <li>Current next-chat model state: {describeModelState(selectedModel)}</li>
            <li>Chat stays blocked until the engine reports the selected model as loaded and ready to generate, and that exact model build is listed as supported.</li>
            <li>The standard /v1-compatible local API is exposed at {apiBase}.</li>
          </ul>
        </section>
      </div>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head">
          <h2>Local API access</h2>
          <StatusDot tone={selectedExactRowReady ? 'ready' : 'warn'} label={endpointReadinessLabel} />
        </div>
        <p className="cxv-sub">Use the same local runtime through standard /v1-compatible endpoints for apps, scripts, and quick terminal checks.</p>
        <div className="sys-endpoints">
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Chat completions</strong><span className="cxv-tag">POST</span></div>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/chat/completions` : 'Unavailable until the local API is running'}</code>
            <p>{chatCompletionsCopy}</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Models</strong><span className="cxv-tag">GET</span></div>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/models` : 'Unavailable until the local API is running'}</code>
            <p>Lists the currently loaded runtime model; this is not a broad model catalog.</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Health</strong><span className="cxv-tag">GET</span></div>
            <code>{runtime?.api_base ? `${runtime.api_base}/v1/health` : 'Unavailable until the local API is running'}</code>
            <p>Source of truth for active_model_id, generation_ready, and the selected load-time execution plan.</p>
          </div>
          <div className="sys-endpoint">
            <div className="sys-endpoint__head"><strong>Capabilities</strong><span className="cxv-tag">GET</span></div>
            <code>{runtime?.api_base ? `${runtime.api_base}/api/capabilities` : 'Unavailable until the local API is running'}</code>
            <p>Support policy for model families, quantization, API features, and compatibility rows.</p>
          </div>
        </div>
      </section>

      <section className="cxv-card cxv-panel">
        <div className="cxv-section__head"><h2>Supported models</h2><span className="cxv-section__count">evidence-based</span></div>
        <p className="cxv-sub">This mirrors /api/capabilities so the UI never implies unvalidated model families, quantization formats, or API features. Row-level evidence lives in the Compatibility ledger.</p>

        <div className="cxv-grid cxv-grid--two">
          <SupportContractSummary capabilities={capabilities} />

          <div className="cxv-card cxv-card--flat sys-evidence">
            <strong>Selected model verification</strong>
            {selectedCompatibilityTarget ? (
              <>
                <code className="a-code">{selectedCompatibilityTarget.id}</code>
                <p>{formatCapabilityStatus(selectedCompatibilityTarget.status)} · {selectedCompatibilityTarget.family} · {selectedCompatibilityTarget.quantization}</p>
                <p><b>Readiness gate:</b> {displayCapabilityCopy(selectedCompatibilityTarget.frontend_readiness_gate || 'not advertised')}</p>
                <p><b>Chat readiness:</b> {selectedExactRowReady ? 'Ready — the model is loaded and verified.' : `${selectedChatGate.label}; loaded=${selectedChatGate.runtimeLoaded ? 'yes' : 'no'}, ready=${selectedChatGate.runtimeGenerationReady ? 'yes' : 'no'}, verified=${selectedChatGate.contractSupported ? 'yes' : 'no'}.`}</p>
                {selectedSupportLanes.map((lane) => (
                  <p key={lane.key}><b>{supportLaneTitle(lane)}:</b> {lane.label}. {displayCapabilityCopy(lane.copy)}</p>
                ))}
                <p>{displayCapabilityCopy(selectedCompatibilityTarget.evidence || selectedCompatibilityTarget.next_step || 'No row evidence advertised.')}</p>
              </>
            ) : (
              <p>The selected model has no verified compatibility entry yet. A running engine alone isn’t treated as verification.</p>
            )}
          </div>
        </div>

        <div className="cxv-card cxv-card--flat sys-evidence">
          <strong>Validated API features</strong>
          {supportedFeatures.length ? (
            <div className="sys-rows">
              {supportedFeatures.map((feature) => (
                <div key={feature.id} className="sys-row">
                  <div className="sys-row__head">
                    <span>{displayCapabilityId(feature.id)}</span>
                    <EvidenceChip status={feature.status} source={{ rowId: feature.id }} size="sm" />
                  </div>
                  <small>{displayCapabilityCopy(feature.notes)}</small>
                </div>
              ))}
            </div>
          ) : (
            <p>No supported API feature rows advertised yet.</p>
          )}
        </div>
      </section>

      <EngineMetricsPanel apiBase={metricsApiBase} runtime={runtime} />
    </section>
  )
}
