import { useEffect, useMemo, useState } from 'react'
import { LLAMA32_3B_ACCEPTANCE_AVAILABILITY, LLAMA32_3B_ACCEPTANCE_GATING_NOTE, LLAMA32_3B_ACCEPTANCE_SUMMARY, LLAMA32_3B_ACCEPTANCE_TARGET } from '../lib/acceptanceTargets'
import { capabilityStatusTone, compatibilityHintCopy, compatibilityHintLabel, compatibilityHintMatchesExactTarget, exactRowSupportLanes, findCompatibilityHint, formatCapabilityStatus, frontendSupportContractCopy, getCurrentCompatibilityTarget, getTrackedCompatibilityTargets, isExactCompatibilityHint, isSupportedCapabilityStatus, rowSupportBoundaryCopy, rowSupportNextStepCopy } from '../lib/capabilities'
import { getChatGateState } from '../lib/chatGate'
import { formatBytes, formatCompactNumber } from '../lib/formatters'
import { canLoadIntoRuntime, describeModelState, getModelStatusLabel, hasLocalModelPath, isExternalModel, isHostedRoutingAvailable, isModelGenerationReady, isModelLoadedNow, isRunnableModel, modelRuntimeIdMatches } from '../lib/modelState'

const FILTERS = [
  { key: 'all', label: 'Everything' },
  { key: 'installed', label: 'Local / loaded' },
  { key: 'external', label: 'Hosted links' },
  { key: 'imported', label: 'Imported' },
  { key: 'downloading', label: 'Downloading' },
  { key: 'attention', label: 'Needs attention' },
]

const CATALOG_PAGE_SIZE = 18

function getGroupKey(model, runtime) {
  if (modelRuntimeIdMatches(model, runtime)) return 'installed'
  if (model.load_error || model.install_error) return 'attention'
  if (isExternalModel(model)) return model.status === 'ready' ? 'external' : 'attention'
  if (model.status === 'failed' || ((model.status === 'ready' || model.status === 'registered') && !model.model_path)) return 'attention'
  if (model.status === 'ready' && model.model_path) return 'installed'
  if (model.status === 'registered') return 'imported'
  if (model.status === 'downloading' || model.status === 'canceling') return 'downloading'
  if (model.status === 'not_installed') return 'catalog'
  return 'attention'
}

function formatModelMeta(model) {
  if (isExternalModel(model)) {
    return [model.source || 'External API', model.runtime_model_name || 'Remote model'].filter(Boolean).join(' · ')
  }
  return [model.size_gb ? `${model.size_gb} GB` : null, model.quant || null, model.engine || null].filter(Boolean).join(' · ')
}

function formatModelOrigin(model) {
  if (isExternalModel(model)) return 'Connected through an external /v1-compatible API.'
  if (model.status === 'registered') return 'Imported from a local GGUF file and waiting for its first successful load.'
  if (model.status === 'downloading') return 'Downloading into Camelid-managed storage.'
  if (model.status === 'canceling') return 'Stopping the download and cleaning up the partial file.'
  if (model.hf_repo) return 'Tracked from the public Hugging Face GGUF catalog.'
  return 'Stored locally on this device.'
}

function formatDownloadCopy(model) {
  if (model.status === 'canceling') return 'Canceling…'
  if (model.status === 'downloading' && model.bytes_downloaded) {
    return `${formatBytes(model.bytes_downloaded)} / ${formatBytes(model.total_bytes)}`
  }
  return `${model.progress || 0}%`
}

function findCatalogMatch(models, item) {
  return models.find((model) => model.hf_repo === item.repo_id && model.hf_filename === item.filename)
}

function matchesLlama32ThreeBTarget(model, capabilities) {
  const target = { id: 'llama32_3b_instruct_q8_0' }
  if (compatibilityHintMatchesExactTarget(capabilities, model, target)) return true

  const subject = [model?.id, model?.name, model?.runtime_model_name, model?.model_path, model?.path, model?.quant].filter(Boolean).join(' ').toLowerCase()
  const hasLlama32ThreeB = /llama[\s._-]*3\.2[\s._-]*3b/.test(subject)
    || /llama[\s._-]*32[\s._-]*3b/.test(subject)
  const hasInstruct = /(?:^|[^a-z0-9])instruct(?:[^a-z0-9]|$)/i.test(subject)
  const hasQ8 = /(?:^|[^a-z0-9])q8[\s._-]*0(?:[^a-z0-9]|$)/i.test(subject)
    || /\bfile[_\s-]*type\s*7\b/i.test(subject)
  const exactAcceptanceIdentity = model?.id === LLAMA32_3B_ACCEPTANCE_TARGET.id
    || model?.model_path === LLAMA32_3B_ACCEPTANCE_TARGET.model_path
  return Boolean((exactAcceptanceIdentity && hasQ8) || (hasLlama32ThreeB && hasInstruct && hasQ8))
}

function findModelMatchingCapabilityRow(models, capabilities, target, runtime, selectedModelId) {
  if (!target) return { model: null, active: false, selected: false }
  const matches = models.filter((model) => compatibilityHintMatchesExactTarget(capabilities, model, target))
  const activeModel = matches.find((model) => modelRuntimeIdMatches(model, runtime)) || null
  const selectedModel = matches.find((model) => model.id === selectedModelId) || null
  return {
    model: activeModel || selectedModel || matches[0] || null,
    active: Boolean(activeModel),
    selected: Boolean(selectedModel),
  }
}

function evidenceTrackTone(value = '') {
  const status = value.toLowerCase()
  if (!status) return ''
  const guardedTone = capabilityStatusTone(status)
  if (guardedTone) return guardedTone
  if (status.includes('validated') || status.includes('measured') || status === 'pass') return 'ready'
  return 'warm'
}

function formatRemoteMeta(item) {
  return [
    item.license ? item.license.toUpperCase() : 'Public repo',
    item.size_bytes ? `${formatBytes(item.size_bytes)} download` : 'Size checks before download',
    `${formatCompactNumber(item.downloads)} downloads`,
    `${formatCompactNumber(item.likes)} likes`,
  ].join(' · ')
}

function formatCatalogTitle(item) {
  const suffix = ` (${item.quant})`
  return item.name.endsWith(suffix) ? item.name.slice(0, -suffix.length) : item.name
}

function CapabilityEvidenceBlock({ capabilities, model, catalogItem }) {
  const quant = model?.quant || catalogItem?.quant || ''
  const compatibilityHint = findCompatibilityHint(capabilities, model, catalogItem)
  const exactTarget = isExactCompatibilityHint(compatibilityHint) ? compatibilityHint.target : null

  return (
    <div className="models-card-copy-stack models-capability-evidence" aria-label="Capability evidence boundary">
      <p className="model-summary"><b>Exact-row quant evidence:</b> {exactTarget ? `${exactTarget.id}: ${exactTarget.quantization} · ${formatCapabilityStatus(exactTarget.status)}` : quant ? `${quant}: no exact compatibility row matched; do not infer support from this quant label.` : 'No quant label and no exact compatibility row matched.'}</p>
      <p className="model-summary"><b>Exact-row support:</b> {compatibilityHintLabel(compatibilityHint, 'No exact compatibility row matched')}. {compatibilityHintCopy(compatibilityHint)}</p>
    </div>
  )
}

function getNextStepCopy(model, { active, selected, runnable, generationReady } = {}) {
  if (!model) return 'Pick a model for the next chat or load one now.'
  if (active && generationReady) return 'Already loaded and ready to answer immediately.'
  if (active) return 'Loaded now, but Camelid still reports generation_ready=false. Check tokenizer/config/tensor readiness and the CPU materialization budget guard before treating this as a UI issue.'
  if (model.load_error || model.install_error) return 'Fix the local path or retry Load now; Camelid will report the exact backend error.'
  if (model.status === 'downloading') return 'Wait for the download to finish before loading it, or cancel it if it is the wrong model.'
  if (model.status === 'canceling') return 'Camelid is stopping this download.'
  if (model.status === 'failed') return isExternalModel(model) ? 'Reconnect the API details to make it usable again.' : 'Retry after checking the local GGUF path.'
  if (model.status === 'registered') return 'Load it once to confirm the file path and move it into the ready set.'
  if (isExternalModel(model) && selected) return 'Chosen for the next chat once API routing is supported by Camelid.'
  if (isExternalModel(model) && runnable) return 'API details are present, but Camelid local chat currently uses local GGUF models.'
  if (model.status === 'ready' && model.model_path && selected) return 'Chosen for your next chat; load it into Camelid before sending a prompt.'
  if (model.status === 'ready' && model.model_path) return 'Saved locally and loadable; choose it or load it into Camelid now.'
  if (runnable && selected) return 'Chosen for your next chat, or load it now for immediate use.'
  if (runnable) return 'Ready locally, choose it for the next chat or load it now.'
  return 'Download or import a local GGUF before you can use it.'
}

function statusTone(model) {
  if (isModelLoadedNow(model) && isModelGenerationReady(model)) return 'ready'
  if (model.status === 'ready' || model.status === 'downloading' || model.status === 'canceling' || model.status === 'registered') return 'warm'
  return ''
}

function modelErrorCopy(model) {
  return model?.load_error || model?.install_error || ''
}

function getReadinessRows(model, runtime) {
  const active = modelRuntimeIdMatches(model, runtime) || isModelLoadedNow(model)
  const readiness = model?.camelid || {}
  const generationReady = isModelGenerationReady(model)
  return [
    { key: 'loaded_now', label: 'Loaded', value: active ? 'true' : 'false', ready: active },
    { key: 'generation_ready', label: 'Generation', value: generationReady ? 'true' : 'false', ready: generationReady },
    { key: 'tokenizer', label: 'Tokenizer', value: readiness.tokenizer_status || (active ? 'unknown' : 'not loaded'), ready: readiness.tokenizer_status === 'available' },
    { key: 'config', label: 'Config', value: readiness.config_ready ? 'ready' : active ? 'missing' : 'not loaded', ready: readiness.config_ready },
    { key: 'tensors', label: 'Tensors', value: readiness.tensor_ready ? 'ready' : active ? 'missing' : 'not loaded', ready: readiness.tensor_ready },
  ]
}

function ReadinessGrid({ model, runtime, includePath = false }) {
  const rows = getReadinessRows(model, runtime)
  const localPath = model?.model_path || model?.path || ''

  return (
    <dl className="models-definition-grid models-readiness-grid">
      {rows.map((row) => (
        <div key={row.key} className={row.ready ? 'ready' : ''}>
          <dt title={row.key}>{row.label}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
      {includePath && localPath && (
        <div className="models-readiness-path">
          <dt>Path</dt>
          <dd title={localPath}>{localPath}</dd>
        </div>
      )}
    </dl>
  )
}

function normalizeSortText(value) {
  return (value || '').toString().trim().toLowerCase()
}

function compareModelsByName(left, right) {
  return normalizeSortText(left.name).localeCompare(normalizeSortText(right.name), undefined, {
    numeric: true,
    sensitivity: 'base',
  }) || normalizeSortText(left.id).localeCompare(normalizeSortText(right.id), undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

function compareCatalogItemsByTitle(left, right) {
  return normalizeSortText(formatCatalogTitle(left)).localeCompare(normalizeSortText(formatCatalogTitle(right)), undefined, {
    numeric: true,
    sensitivity: 'base',
  }) || normalizeSortText(left.repo_id).localeCompare(normalizeSortText(right.repo_id), undefined, {
    numeric: true,
    sensitivity: 'base',
  })
}

export default function ModelsView({
  runtime,
  capabilities,
  refreshDashboard,
  registerForm,
  setRegisterForm,
  externalForm,
  setExternalForm,
  registerModel,
  connectExternalModel,
  models,
  selectedModelId,
  setSelectedModelId,
  loadingModelId,
  activateModel,
  unloadCurrentModel,
  installModel,
  installCatalogModel,
  cancelModelDownload,
}) {
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [showImportAdvanced, setShowImportAdvanced] = useState(false)
  const [catalogItems, setCatalogItems] = useState([])
  const [catalogNextCursor, setCatalogNextCursor] = useState(null)
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [catalogLoadingMore, setCatalogLoadingMore] = useState(false)
  const [catalogError, setCatalogError] = useState('')
  const [catalogAvailable, setCatalogAvailable] = useState(false)
  const [refreshingRuntime, setRefreshingRuntime] = useState(false)

  const catalogApiBase = (runtime?.api_base || '').replace(/\/$/, '')
  const runtimeOnline = runtime?.status === 'online'
  const hostedRoutingAvailable = Boolean(capabilities?.hosted_provider_routing || capabilities?.external_api_routing || capabilities?.openai_compatible_routing)
  const catalogInstallAvailable = Boolean(capabilities?.model_catalog_install || capabilities?.model_downloads || capabilities?.hf_catalog_install)
  const apiFeatures = capabilities?.api_features || []
  const supportContractCurrentGate = frontendSupportContractCopy(capabilities)
  const currentCompatibilityTarget = getCurrentCompatibilityTarget(capabilities)
  const compatibilityRows = capabilities?.model_compatibility || []
  const trackedCompatibilityRows = getTrackedCompatibilityTargets(capabilities)
  const supportedCompatibilityRows = compatibilityRows.filter((target) => isSupportedCapabilityStatus(target.status))
  const plannedCompatibilityRows = compatibilityRows.filter((target) => !isSupportedCapabilityStatus(target.status))
  const supportedCompatibilitySummary = supportedCompatibilityRows.map((target) => target.id).join(' · ') || (currentCompatibilityTarget ? currentCompatibilityTarget.id : '')
  const exactQuantEvidenceSummary = compatibilityRows.map((target) => `${target.id}: ${target.quantization} (${formatCapabilityStatus(target.status)})`).join(' · ') || 'No exact rows advertised'
  const guardedCompatibilitySummary = plannedCompatibilityRows.map((target) => `${target.id}: ${formatCapabilityStatus(target.status)}`).join(' · ') || 'No guarded rows advertised'

  const refreshRuntime = async () => {
    if (!refreshDashboard || refreshingRuntime) return
    setRefreshingRuntime(true)
    try {
      await refreshDashboard()
    } finally {
      setRefreshingRuntime(false)
    }
  }

  useEffect(() => {
    if (!catalogApiBase) {
      setCatalogAvailable(false)
      setCatalogItems([])
      setCatalogNextCursor(null)
      setCatalogError('')
      setCatalogLoading(false)
      return undefined
    }

    const controller = new AbortController()
    const timer = setTimeout(async () => {
      setCatalogLoading(true)
      setCatalogError('')
      try {
        const params = new URLSearchParams({ limit: String(CATALOG_PAGE_SIZE) })
        if (query.trim()) params.set('query', query.trim())
        const res = await fetch(`${catalogApiBase}/api/models/catalog?${params.toString()}`, { signal: controller.signal })
        if (res.status === 404 || res.status === 405) {
          setCatalogAvailable(false)
          setCatalogItems([])
          setCatalogNextCursor(null)
          return
        }
        setCatalogAvailable(true)
        if (!res.ok) throw new Error('Could not load the Hugging Face catalog.')
        const data = await res.json()
        setCatalogItems(data.items || [])
        setCatalogNextCursor(data.next_cursor || null)
      } catch (error) {
        if (error.name === 'AbortError') return
        setCatalogAvailable(false)
        setCatalogError(error.message || 'Could not load the Hugging Face catalog.')
        setCatalogItems([])
        setCatalogNextCursor(null)
      } finally {
        if (!controller.signal.aborted) setCatalogLoading(false)
      }
    }, query.trim() ? 250 : 0)

    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [catalogApiBase, query])

  const loadMoreCatalog = async () => {
    if (!catalogApiBase || !catalogNextCursor || catalogLoadingMore) return
    setCatalogLoadingMore(true)
    setCatalogError('')
    try {
      const params = new URLSearchParams({ limit: String(CATALOG_PAGE_SIZE), cursor: catalogNextCursor })
      if (query.trim()) params.set('query', query.trim())
      const res = await fetch(`${catalogApiBase}/api/models/catalog?${params.toString()}`)
      if (res.status === 404 || res.status === 405) {
        setCatalogAvailable(false)
        setCatalogItems([])
        setCatalogNextCursor(null)
        return
      }
      if (!res.ok) throw new Error('Could not load more catalog results.')
      const data = await res.json()
      setCatalogItems((current) => {
        const existing = new Set(current.map((item) => item.catalog_id))
        return [...current, ...(data.items || []).filter((item) => !existing.has(item.catalog_id))]
      })
      setCatalogNextCursor(data.next_cursor || null)
    } catch (error) {
      setCatalogError(error.message || 'Could not load more catalog results.')
    } finally {
      setCatalogLoadingMore(false)
    }
  }

  const filteredModels = useMemo(() => {
    const q = query.trim().toLowerCase()
    return models.filter((model) => {
      const groupKey = getGroupKey(model, runtime)
      const matchesFilter = statusFilter === 'all' || groupKey === statusFilter
      if (!matchesFilter) return false
      if (!q) return true
      return [
        model.name,
        model.id,
        model.quant,
        model.engine,
        model.source,
        model.hf_repo,
        model.hf_filename,
        model.runtime_model_name,
      ].filter(Boolean).some((value) => value.toLowerCase().includes(q))
    }).sort(compareModelsByName)
  }, [models, query, runtime, statusFilter])

  const groupedModels = useMemo(() => {
    const groups = {
      installed: [],
      external: [],
      imported: [],
      downloading: [],
      attention: [],
      catalog: [],
    }
    filteredModels.forEach((model) => {
      groups[getGroupKey(model, runtime)]?.push(model)
    })
    return groups
  }, [filteredModels, runtime])

  const counts = useMemo(() => ({
    loaded: models.filter((model) => isModelLoadedNow(model) || modelRuntimeIdMatches(model, runtime)).length || (runtime?.loaded_now ? 1 : 0),
    generationReady: models.filter(isModelGenerationReady).length || (runtime?.generation_ready ? 1 : 0),
    localPaths: models.filter((model) => !isExternalModel(model) && hasLocalModelPath(model)).length,
    installed: models.filter((model) => getGroupKey(model, runtime) === 'installed').length,
    external: models.filter((model) => getGroupKey(model, runtime) === 'external').length,
    downloading: models.filter((model) => getGroupKey(model, runtime) === 'downloading').length,
    imported: models.filter((model) => getGroupKey(model, runtime) === 'imported').length,
    attention: models.filter((model) => getGroupKey(model, runtime) === 'attention').length,
  }), [models, runtime])

  const selectedLocalModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) || null,
    [models, selectedModelId],
  )

  const activeLocalModel = useMemo(
    () => models.find((model) => modelRuntimeIdMatches(model, runtime)) || null,
    [models, runtime?.active_model_id],
  )

  const llama32ThreeBModel = useMemo(
    () => models.find((model) => matchesLlama32ThreeBTarget(model, capabilities)) || null,
    [models, capabilities],
  )
  const showLlama32ThreeBAcceptanceTarget = !llama32ThreeBModel
  const fillLlama32ThreeBImport = () => {
    setRegisterForm({
      id: LLAMA32_3B_ACCEPTANCE_TARGET.id,
      name: LLAMA32_3B_ACCEPTANCE_TARGET.name,
      model_path: LLAMA32_3B_ACCEPTANCE_TARGET.model_path,
      runtime_model_name: LLAMA32_3B_ACCEPTANCE_TARGET.runtime_model_name,
    })
    setShowImportAdvanced(true)
  }

  const selectedChatGate = getChatGateState(capabilities, selectedLocalModel, runtime)
  const selectedRuntimeReady = selectedChatGate.runtimeReady
  const selectedRunnable = selectedChatGate.chatUnlocked
  const selectedContractBlocked = selectedRuntimeReady && !selectedChatGate.contractSupported
  const activeGenerationReady = activeLocalModel ? isModelGenerationReady(activeLocalModel) : Boolean(runtime?.generation_ready)
  const readyModels = [...groupedModels.installed].sort(compareModelsByName)
  const apiLinkModels = [...groupedModels.external].sort(compareModelsByName)
  const setupModels = [...groupedModels.imported, ...groupedModels.downloading, ...groupedModels.attention].sort(compareModelsByName)
  const discoverCatalogItems = useMemo(
    () => catalogItems.filter((item) => {
      const localMatch = findCatalogMatch(models, item)
      if (!localMatch) return true
      return localMatch.status === 'failed' || localMatch.status === 'not_installed'
    }).sort(compareCatalogItemsByTitle),
    [catalogItems, models],
  )

  return (
    <section className="view-stack models-view view-shell-wide">
      <div className="panel models-toolbar-panel">
        <div className="models-toolbar-top">
          <label className="models-search-field">
            <span>Search models</span>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search by name, repo, quant, source, or file" />
          </label>
          <label className="models-filter-field">
            <span>Show</span>
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              {FILTERS.map((filter) => (
                <option key={filter.key} value={filter.key}>{filter.label}</option>
              ))}
            </select>
          </label>
          <button type="button" className="ghost-button models-refresh-button" onClick={refreshRuntime} disabled={!refreshDashboard || refreshingRuntime}>
            {refreshingRuntime ? 'Refreshing…' : 'Refresh runtime'}
          </button>
        </div>
        <p className="model-summary">Models are sorted A→Z by name. The backend currently exposes the loaded runtime model through /v1/models; saved local paths come from this browser until Camelid grows a local registry endpoint.</p>
        <div className="models-truth-strip" aria-label="Backend model data sources">
          <div>
            <span>API</span>
            <strong>{runtimeOnline ? 'Online' : 'Offline'}</strong>
            <small>{runtime?.api_base || 'No API base configured'}</small>
          </div>
          <div>
            <span>/v1/models</span>
            <strong>{runtime?.loaded_now ? 'Loaded model only' : 'No loaded model'}</strong>
            <small>{runtime?.loaded_now ? `${runtime?.active_model_id}; saved browser paths: ${counts.localPaths}` : `Camelid reports an empty model list when nothing is loaded. Saved browser paths: ${counts.localPaths}.`}</small>
          </div>
          <div>
            <span>/api/capabilities</span>
            <strong>{capabilities ? 'Contract live' : 'Not available'}</strong>
            <small>{capabilities?.support_contract ? supportContractCurrentGate : 'No support contract returned; non-matching rows stay disabled.'}{supportedCompatibilitySummary ? ` Rows: ${supportedCompatibilitySummary}.` : ''}</small>
          </div>
          <div>
            <span>Catalog</span>
            <strong>{catalogAvailable ? 'Endpoint detected' : 'Hidden'}</strong>
            <small>{catalogAvailable ? 'Showing only because /api/models/catalog responded.' : 'Hidden until the backend exposes /api/models/catalog.'}</small>
          </div>
          <div>
            <span>Hosted APIs</span>
            <strong>{hostedRoutingAvailable ? 'Routing advertised' : 'Routing planned'}</strong>
            <small>Provider links stay disabled until /api/capabilities exposes hosted-provider routing and the frontend has matching evidence.</small>
          </div>
        </div>
        <div className="models-compatibility-strip" aria-label="Camelid compatibility support contract">
          <div>
            <span>Current supported gate</span>
            <strong>{capabilities?.support_contract ? supportContractCurrentGate : 'No /api/capabilities contract'}</strong>
            <small>{supportedCompatibilitySummary ? `Supported rows: ${supportedCompatibilitySummary}. Runtime loaded_now=true and generation_ready=true are still required.` : 'The UI will not infer support beyond loaded/model readiness.'}</small>
          </div>
          <div>
            <span>Exact-row quants</span>
            <strong>{exactQuantEvidenceSummary}</strong>
            <small>Quant evidence is row-scoped; filenames and saved paths do not promote support.</small>
          </div>
          <div>
            <span>Guarded rows</span>
            <strong>{guardedCompatibilitySummary}</strong>
            <small>{plannedCompatibilityRows.length ? `${plannedCompatibilityRows.length} compatibility row${plannedCompatibilityRows.length === 1 ? '' : 's'} remain planned or guarded; backend typed errors are expected until COMPATIBILITY.md records evidence.` : 'No planned compatibility rows advertised.'}</small>
          </div>
        </div>
        <div className="models-summary-strip" aria-label="Model summary">
          <div className="models-summary-pill models-summary-pill-static">
            <span>Loaded now</span>
            <strong>{counts.loaded}</strong>
          </div>
          <div className="models-summary-pill models-summary-pill-static">
            <span>Generation-ready</span>
            <strong>{counts.generationReady}</strong>
          </div>
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'installed' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'installed' ? 'all' : 'installed')}>
            <span>Local / loaded</span>
            <strong>{counts.installed}</strong>
          </button>
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'external' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'external' ? 'all' : 'external')}>
            <span>Hosted links</span>
            <strong>{counts.external}</strong>
          </button>
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'imported' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'imported' ? 'all' : 'imported')}>
            <span>Imported</span>
            <strong>{counts.imported}</strong>
          </button>
          <button type="button" className={`ghost-button models-summary-pill ${statusFilter === 'attention' ? 'active' : ''}`} onClick={() => setStatusFilter((current) => current === 'attention' ? 'all' : 'attention')}>
            <span>Needs attention</span>
            <strong>{counts.attention}</strong>
          </button>
        </div>
      </div>

      <section className="models-status-grid" aria-label="Camelid runtime model readiness">
        <div className="panel models-status-card">
          <p className="panel-kicker">Loaded now</p>
          <h3>{runtime?.loaded_now ? activeLocalModel?.name || runtime?.active_model_id : 'Nothing loaded'}</h3>
          <p className="model-summary">
            {runtime?.loaded_now
              ? activeGenerationReady
                ? 'Camelid reports generation_ready=true for the loaded model.'
                : 'A model is loaded, but generation_ready=false. Chat stays blocked while Camelid checks tokenizer/config/tensors and the CPU materialization budget guard.'
              : 'Load a saved local GGUF to make it available for chat.'}
          </p>
          <div className="models-card-tags">
            <div className={`pin-badge ${runtime?.loaded_now ? 'ready' : ''}`}>loaded_now: {runtime?.loaded_now ? 'true' : 'false'}</div>
            <div className={`pin-badge ${runtime?.generation_ready ? 'ready' : ''}`}>generation_ready: {runtime?.generation_ready ? 'true' : 'false'}</div>
          </div>
          {activeLocalModel && <ReadinessGrid model={activeLocalModel} runtime={runtime} includePath />}
          {runtime?.loaded_now && (
            <div className="models-card-actions">
              <button className="ghost-button" onClick={unloadCurrentModel} disabled={Boolean(loadingModelId)}>{loadingModelId === runtime?.active_model_id ? 'Unloading…' : 'Unload current model'}</button>
            </div>
          )}
        </div>
        <div className="panel models-status-card">
          <p className="panel-kicker">Next chat</p>
          <h3>{selectedLocalModel?.name || 'No model selected'}</h3>
          <p className="model-summary">
            {selectedRunnable
              ? 'The selected model is loaded, generation-ready, and backed by an exact supported /api/capabilities row for the next chat.'
              : selectedContractBlocked
                ? `${selectedChatGate.label}: Camelid reports this exact model loaded and generation-ready, but chat stays blocked until /api/capabilities promotes the matching COMPATIBILITY.md row.`
                : selectedLocalModel
                  ? getNextStepCopy(selectedLocalModel, {
                    active: modelRuntimeIdMatches(selectedLocalModel, runtime),
                    selected: true,
                    runnable: selectedRuntimeReady,
                    generationReady: isModelGenerationReady(selectedLocalModel),
                  })
                  : 'Import or load a local GGUF, then choose it for the next chat.'}
          </p>
          <div className="models-card-tags">
            {selectedLocalModel && <div className="pin-badge">selected: {selectedLocalModel.id}</div>}
            {selectedLocalModel && <div className={`pin-badge ${selectedRunnable ? 'ready' : 'warm'}`}>{selectedRunnable ? 'chat enabled' : selectedContractBlocked ? 'contract blocked' : 'chat blocked'}</div>}
          </div>
          {selectedLocalModel && <ReadinessGrid model={selectedLocalModel} runtime={runtime} includePath />}
        </div>
      </section>

      {showLlama32ThreeBAcceptanceTarget && (
        <section className="panel models-section-panel" aria-label="Llama 3.2 3B Q8 exact supported row">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">Exact supported row</p>
              <h3>Llama 3.2 3B Instruct Q8_0</h3>
            </div>
            <p className="model-summary">{LLAMA32_3B_ACCEPTANCE_SUMMARY}</p>
          </div>

          <div className="models-card-grid">
            <article className="model-card models-model-card">
              <div className="models-card-head">
                <div className="models-card-title">
                  <strong>{LLAMA32_3B_ACCEPTANCE_TARGET.name}</strong>
                  <span>{formatModelMeta(LLAMA32_3B_ACCEPTANCE_TARGET)}</span>
                </div>
                <div className="status-pill ready">Supported exact-row smoke · runtime required</div>
              </div>

              <div className="models-card-tags">
                <div className="pin-badge ready">Exact row only</div>
                <div className="pin-badge">Q8_0</div>
                <div className="pin-badge ready">Compact parity exists</div>
                <div className="pin-badge ready">3B API smoke passed</div>
                <div className="pin-badge ready">3B WebUI smoke passed</div>
              </div>

              <div className="models-card-copy-stack">
                <p className="model-summary">Expected source: {LLAMA32_3B_ACCEPTANCE_TARGET.source}.</p>
                <p className="model-summary">{LLAMA32_3B_ACCEPTANCE_AVAILABILITY}</p>
                <p className="model-summary">{LLAMA32_3B_ACCEPTANCE_GATING_NOTE}</p>
              </div>

              <CapabilityEvidenceBlock capabilities={capabilities} model={LLAMA32_3B_ACCEPTANCE_TARGET} />
              <ReadinessGrid model={LLAMA32_3B_ACCEPTANCE_TARGET} runtime={runtime} includePath />
              <p className="model-summary">Do not infer readiness from the 1B row, the 8B row, or any neighboring quant; this exact 3B row still needs runtime-green loaded_now=true + generation_ready=true before chat unlocks.</p>

              <div className="models-card-actions">
                <button className="ghost-button" onClick={fillLlama32ThreeBImport}>Fill import form with exact path</button>
                <button className="ghost-button" onClick={refreshRuntime} disabled={!refreshDashboard || refreshingRuntime}>{refreshingRuntime ? 'Refreshing…' : 'Refresh runtime'}</button>
              </div>
            </article>
          </div>
        </section>
      )}

      {trackedCompatibilityRows.length > 0 && (
        <section className="panel models-section-panel" aria-label="Tracked exact Q8 compatibility rows">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">Exact-row full-support hardening</p>
              <h3>Current Q8 support rows</h3>
            </div>
            <p className="model-summary">These cards mirror the current exact Q8 rows from /api/capabilities. Each row gets credit only for its own evidence, with template/Jinja and production-throughput shown as row-scoped readiness lanes instead of repeated generic caveats; chat still unlocks only when the active local GGUF is loaded_now=true, generation_ready=true, and matched to that exact supported row.</p>
          </div>

          <div className="models-card-grid">
            {trackedCompatibilityRows.map((target) => {
              const tone = capabilityStatusTone(target.status)
              const supported = isSupportedCapabilityStatus(target.status)
              const match = findModelMatchingCapabilityRow(models, capabilities, target, runtime, selectedModelId)
              const matchedModel = match.model
              const runtimeReady = Boolean(match.active && matchedModel && isModelGenerationReady(matchedModel))
              const chatUnlocked = Boolean(supported && runtimeReady)
              const supportLanes = exactRowSupportLanes(target, apiFeatures)
              const templateLane = supportLanes.find((lane) => lane.key === 'template')
              const throughputLane = supportLanes.find((lane) => lane.key === 'throughput')

              return (
                <article key={target.id} className="model-card models-model-card">
                  <div className="models-card-head">
                    <div className="models-card-title">
                      <strong>{target.id}</strong>
                      <span>{target.family} · {target.quantization}</span>
                    </div>
                    <div className={`status-pill ${tone}`}>{formatCapabilityStatus(target.status)}</div>
                  </div>

                  <div className="models-card-tags">
                    <div className="pin-badge">{target.tested_context || 'Context not advertised'}</div>
                    <div className={`pin-badge ${target.frontend_load_path_verified === 'validated' ? 'ready' : 'warm'}`}>frontend: {formatCapabilityStatus(target.frontend_load_path_verified || 'not_promoted')}</div>
                    <div className={`pin-badge ${evidenceTrackTone(target.chat_template_shape_pack)}`}>template pack: {formatCapabilityStatus(target.chat_template_shape_pack || 'not_started')}</div>
                    {templateLane && <div className={`pin-badge ${templateLane.tone}`}>Template/Jinja: {templateLane.label}</div>}
                    <div className={`pin-badge ${evidenceTrackTone(target.bounded_context_512_pack)}`}>512-context: {formatCapabilityStatus(target.bounded_context_512_pack || 'not_started')}</div>
                    <div className={`pin-badge ${evidenceTrackTone(target.bounded_context_1024_pack)}`}>1024-context: {formatCapabilityStatus(target.bounded_context_1024_pack || 'not_started')}</div>
                    <div className={`pin-badge ${evidenceTrackTone(target.bounded_context_2048_pack)}`}>2048-context: {formatCapabilityStatus(target.bounded_context_2048_pack || 'not_started')}</div>
                    <div className={`pin-badge ${evidenceTrackTone(target.bounded_context_4096_pack)}`}>4096-context: {formatCapabilityStatus(target.bounded_context_4096_pack || 'not_started')}</div>
                    <div className={`pin-badge ${evidenceTrackTone(target.bounded_context_8192_pack)}`}>8192-context: {formatCapabilityStatus(target.bounded_context_8192_pack || 'not_started')}</div>
                    {target.latest_checked_bucket && (
                      <div className={`pin-badge ${target.latest_checked_result === 'pass' ? 'ready' : evidenceTrackTone(target.latest_checked_result)}`}>
                        latest: {formatCapabilityStatus(target.latest_checked_bucket)} → {formatCapabilityStatus(target.latest_checked_result || 'not_started')}{target.latest_checked_output && target.latest_checked_output !== 'not_applicable' ? ` (${target.latest_checked_output})` : ''}
                      </div>
                    )}
                    <div className={`pin-badge ${evidenceTrackTone(target.full_support_status)}`}>full-support: {formatCapabilityStatus(target.full_support_status || 'not_advertised')}</div>
                    <div className={`pin-badge ${evidenceTrackTone(target.performance_measured)}`}>perf: {formatCapabilityStatus(target.performance_measured || 'not_started')}</div>
                    {throughputLane && <div className={`pin-badge ${throughputLane.tone}`}>Throughput: {throughputLane.label}</div>}
                    <div className={`pin-badge ${chatUnlocked ? 'ready' : 'warm'}`}>{chatUnlocked ? 'Chat unlockable' : supported ? 'Runtime still needed' : 'Chat blocked by row status'}</div>
                    {target.id === 'tinyllama_1_1b_chat_q8_0' && <div className="pin-badge ready">TinyLlama current gate</div>}
                    {target.id === 'tinyllama_1_1b_chat_q8_0' && <div className="pin-badge ready">TinyLlama API/WebUI smoke passed</div>}
                    {target.id === 'tinyllama_1_1b_chat_q8_0' && <div className="pin-badge ready">TinyLlama five-prompt parity passed</div>}
                    {target.id === 'tinyllama_1_1b_chat_q8_0' && <div className="pin-badge ready">TinyLlama first 512-context pack passed</div>}
                    {target.id === 'tinyllama_1_1b_chat_q8_0' && <div className="pin-badge warm">TinyLlama 1024/2048 refresh not promoted</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B API/WebUI smoke passed</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B compact + broader parity passed</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B first 512-context pack passed</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B second 1024-context pack passed</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B third 2048-context pack passed</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B fourth 4096-context pack passed</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B fifth 8192-context pack passed</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B compact template-shapes pack passed</div>}
                    {target.id === 'llama32_1b_instruct_q8_0' && <div className="pin-badge ready">1B unique-chat perf/RSS passed</div>}
                    {target.id === 'llama32_3b_instruct_q8_0' && <div className="pin-badge ready">3B API/WebUI smoke passed</div>}
                    {target.id === 'llama32_3b_instruct_q8_0' && <div className="pin-badge ready">3B broader 50-token parity passed</div>}
                    {target.id === 'llama32_3b_instruct_q8_0' && <div className="pin-badge ready">3B five-prompt API smoke passed</div>}
                    {target.id === 'llama32_3b_instruct_q8_0' && <div className="pin-badge ready">3B first 512-context pack passed</div>}
                    {target.id === 'llama32_3b_instruct_q8_0' && <div className="pin-badge ready">3B second 1024-context pack passed</div>}
                    {target.id === 'llama32_3b_instruct_q8_0' && <div className="pin-badge ready">3B third 2048-context pack passed</div>}
                    {target.id === 'llama32_3b_instruct_q8_0' && <div className="pin-badge ready">3B compact template-shapes pack passed</div>}
                    {target.id === 'llama32_3b_instruct_q8_0' && <div className="pin-badge ready">3B unique-chat perf/RSS passed</div>}
                    {target.id === 'llama3_8b_instruct_q8_0' && <div className="pin-badge ready">8B API/WebUI smoke passed</div>}
                    {target.id === 'llama3_8b_instruct_q8_0' && <div className="pin-badge ready">8B clean-main timing/RSS smoke passed</div>}
                    {target.id === 'llama3_8b_instruct_q8_0' && <div className="pin-badge warm">8B lazy-Q8 hot-path costs measured</div>}
                    {target.id === 'llama3_8b_instruct_q8_0' && <div className="pin-badge ready">8B broader 50-token pack passed</div>}
                    {target.id === 'llama3_8b_instruct_q8_0' && <div className="pin-badge ready">8B first 512-context pack passed</div>}
                    {target.id === 'llama3_8b_instruct_q8_0' && <div className="pin-badge ready">8B 1024/2048 bounded packs passed</div>}
                    {target.id === 'llama3_8b_instruct_q8_0' && <div className="pin-badge ready">8B compact template-shapes pack passed</div>}
                    {match.active && <div className="pin-badge ready">Loaded exact-row match</div>}
                    {!match.active && match.selected && <div className="pin-badge">Selected exact-row match</div>}
                  </div>

                  <div className="models-card-copy-stack">
                    <p className="model-summary"><b>Evidence:</b> {target.evidence}</p>
                    {supportLanes.map((lane) => (
                      <p className="model-summary" key={lane.key}><b>{lane.key === 'template' ? 'Template/Jinja readiness' : 'Throughput readiness'}:</b> {lane.copy}</p>
                    ))}
                    <p className="model-summary"><b>Remaining support boundary:</b> {rowSupportBoundaryCopy(target, apiFeatures)}</p>
                    <p className="model-summary"><b>Next step:</b> {rowSupportNextStepCopy(target, apiFeatures)}</p>
                    <p className="model-summary">
                      {chatUnlocked
                        ? 'This exact row is both contract-supported and runtime-green.'
                        : runtimeReady
                          ? `Camelid can load a matching local GGUF right now, but chat must stay blocked because this row is still ${formatCapabilityStatus(target.status)}.`
                          : supported
                            ? 'This row is contract-supported, but chat still requires a matching loaded_now=true + generation_ready=true local GGUF.'
                            : 'This row is not promoted yet; keep readiness and smoke expectations guarded until backend evidence advances this exact row.'}
                    </p>
                  </div>

                  {matchedModel ? (
                    <>
                      <CapabilityEvidenceBlock capabilities={capabilities} model={matchedModel} />
                      <ReadinessGrid model={matchedModel} runtime={runtime} includePath />
                    </>
                  ) : (
                    <p className="model-summary">No local browser entry currently matches this exact row. That is not a green state and does not erase backend evidence or blockers for the row.</p>
                  )}

                  {target.id === 'llama32_3b_instruct_q8_0' && !matchedModel && (
                    <div className="models-card-actions">
                      <button className="ghost-button" onClick={fillLlama32ThreeBImport}>Fill import form with exact path</button>
                    </div>
                  )}
                </article>
              )
            })}
          </div>
        </section>
      )}

      <section className="panel models-section-panel">
        <div className="models-section-heading">
          <div>
            <p className="panel-kicker">Local runtime</p>
            <h3>{readyModels.length === 0 ? 'No local models listed yet' : `${readyModels.length} local ${readyModels.length === 1 ? 'model' : 'models'}`}</h3>
          </div>
          <p className="model-summary">These are loaded now or saved with a local GGUF path. The page keeps loadable-local separate from chat-ready: chat unlocks only when Camelid reports loaded_now=true, generation_ready=true, and an exact supported /api/capabilities row for the active model/quant.</p>
        </div>

        {readyModels.length === 0 ? (
          <div className="empty-state">No loaded or saved local GGUF paths yet. Import a local file below, or load a model from another client and refresh.</div>
        ) : (
          <div className="models-card-grid">
            {readyModels.map((model) => {
              const runtimeReady = isRunnableModel(model)
              const chatGate = getChatGateState(capabilities, model, runtime)
              const chatUnlocked = chatGate.chatUnlocked
              const contractBlocked = runtimeReady && !chatGate.contractSupported
              const canLoad = canLoadIntoRuntime(model)
              const external = isExternalModel(model)
              const active = modelRuntimeIdMatches(model, runtime)
              const loadedNow = isModelLoadedNow(model) || active
              const generationReady = isModelGenerationReady(model)
              const selected = selectedModelId === model.id
              const busy = loadingModelId === model.id
              const errorCopy = modelErrorCopy(model)

              return (
                <article key={model.id} className={`model-card models-model-card ${active ? 'active-model-card' : ''} ${selected ? 'selected-model-card' : ''}`}>
                  <div className="models-card-head">
                    <div className="models-card-title">
                      <strong>{model.name}</strong>
                      <span>{formatModelMeta(model)}</span>
                    </div>
                    <div className={`status-pill ${statusTone(model)}`}>{getModelStatusLabel(model)}</div>
                  </div>

                  <div className="models-card-tags">
                    {loadedNow && <div className={`pin-badge ${generationReady ? 'ready' : 'warm'}`}>{generationReady ? 'Loaded + generation-ready' : 'Loaded, not ready'}</div>}
                    {selected && <div className={`pin-badge ${chatUnlocked ? 'ready' : contractBlocked ? 'warm' : ''}`}>{chatUnlocked ? 'Next chat ready' : contractBlocked ? 'Next chat contract-blocked' : 'Next chat'}</div>}
                    {external && <div className="pin-badge">API planned</div>}
                    {model.model_path && !external && <div className="pin-badge">Saved locally</div>}
                  </div>

                  <div className="models-card-copy-stack">
                    <p className="model-summary">{describeModelState(model)}</p>
                    <p className="model-summary">{contractBlocked ? `${chatGate.label}: runtime is green, but chat remains blocked until this exact COMPATIBILITY.md row is supported by /api/capabilities.` : getNextStepCopy(model, { active: loadedNow, selected, runnable: runtimeReady, generationReady })}</p>
                    <p className="model-summary">{formatModelOrigin(model)}</p>
                  </div>

                  {!external && <CapabilityEvidenceBlock capabilities={capabilities} model={model} />}

                  {!external && <ReadinessGrid model={model} runtime={runtime} includePath />}

                  {errorCopy && <p className="library-error-copy">{errorCopy}</p>}

                  <div className="models-card-actions">
                    <button className="ghost-button" onClick={() => setSelectedModelId(model.id)} disabled={busy}>{selected ? 'Chosen for next chat' : chatUnlocked ? 'Use for next chat' : contractBlocked ? 'Select; chat stays blocked' : generationReady ? 'Select and inspect contract' : 'Select after load'}</button>
                    {canLoad && !external && <button className="primary-button" onClick={() => activateModel(model.id)} disabled={busy || (loadedNow && generationReady)}>{busy ? 'Loading…' : loadedNow ? generationReady ? 'Loaded now' : 'Retry readiness check' : 'Load now'}</button>}
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </section>

      {catalogAvailable && (
        <section className="panel models-section-panel models-catalog-panel-clean">
          <div className="models-section-heading models-section-heading-catalog">
          <div>
            <p className="panel-kicker">Catalog preview</p>
            <h3>Public Hugging Face GGUF picks</h3>
          </div>
          <p className="model-summary">This area stays honest: it appears only if a catalog endpoint responds. Downloads stay disabled unless Camelid advertises a catalog-install capability; local GGUF loading remains the working path today.</p>
        </div>

        {catalogError && <p className="library-error-copy">{catalogError}</p>}

        {catalogLoading && discoverCatalogItems.length === 0 ? (
          <div className="empty-state">Loading open Hugging Face models…</div>
        ) : discoverCatalogItems.length === 0 ? (
          <div className="empty-state">No new public GGUF models matched that search.</div>
        ) : (
          <>
            <div className="models-card-grid models-catalog-grid-clean">
              {discoverCatalogItems.map((item) => {
                const localMatch = findCatalogMatch(models, item)
                const runnable = Boolean(localMatch && (localMatch.status === 'ready' || localMatch.status === 'registered' || localMatch.status === 'failed') && hasLocalModelPath(localMatch))
                const active = modelRuntimeIdMatches(localMatch, runtime)
                const selected = selectedModelId === localMatch?.id
                const busy = localMatch && loadingModelId === localMatch.id
                const errorCopy = modelErrorCopy(localMatch)

                return (
                  <article key={item.catalog_id} className={`model-card models-model-card models-catalog-card-clean ${active ? 'active-model-card' : ''} ${selected ? 'selected-model-card' : ''}`}>
                    <div className="models-card-head">
                      <div className="models-card-title">
                        <strong>{formatCatalogTitle(item)}</strong>
                        <span>{formatRemoteMeta(item)}</span>
                      </div>
                      {localMatch ? <div className={`status-pill ${statusTone(localMatch)}`}>{getModelStatusLabel(localMatch)}</div> : null}
                    </div>

                    <div className="models-card-tags">
                      {active && <div className="pin-badge">Loaded now</div>}
                      {selected && <div className="pin-badge">Next chat</div>}
                      {item.quant && <div className="pin-badge">Catalog quant: {item.quant}</div>}
                      {hasLocalModelPath(localMatch) && <div className="pin-badge">Saved locally</div>}
                    </div>

                    <dl className="models-definition-grid">
                      <div>
                        <dt>Repo</dt>
                        <dd title={item.repo_id}>{item.repo_id}</dd>
                      </div>
                      <div>
                        <dt>File</dt>
                        <dd title={item.filename}>{item.filename}</dd>
                      </div>
                      <div>
                        <dt>Download size</dt>
                        <dd>{item.size_bytes ? formatBytes(item.size_bytes) : 'Checked before download'}</dd>
                      </div>
                    </dl>

                    <CapabilityEvidenceBlock capabilities={capabilities} model={localMatch} catalogItem={item} />

                    {localMatch && <ReadinessGrid model={localMatch} runtime={runtime} />}

                    {errorCopy && <p className="library-error-copy">{errorCopy}</p>}

                    <div className="models-card-actions">
                      {(!localMatch || localMatch.status === 'not_installed' || (localMatch.status === 'failed' && !hasLocalModelPath(localMatch))) && <button className="primary-button" onClick={() => installCatalogModel(item)} disabled={Boolean(busy) || !catalogInstallAvailable}>{catalogInstallAvailable ? 'Download' : 'Download planned'}</button>}
                      {(localMatch?.status === 'downloading' || localMatch?.status === 'canceling') && <button className="ghost-button" onClick={() => cancelModelDownload(localMatch.id)} disabled={localMatch.status === 'canceling'}>{localMatch.status === 'canceling' ? 'Canceling…' : 'Cancel download'}</button>}
                      {runnable && <button className="ghost-button" onClick={() => setSelectedModelId(localMatch.id)} disabled={Boolean(busy)}>{selected ? 'Chosen for next chat' : 'Use for next chat'}</button>}
                      {runnable && <button className="primary-button" onClick={() => activateModel(localMatch.id)} disabled={Boolean(busy) || active}>{busy ? 'Loading…' : active ? 'Loaded now' : localMatch.status === 'registered' ? 'Load now and confirm file' : 'Load now'}</button>}
                    </div>
                  </article>
                )
              })}
            </div>

            {catalogNextCursor && (
              <div className="library-load-more-row">
                <button className="ghost-button" onClick={loadMoreCatalog} disabled={catalogLoadingMore}>
                  {catalogLoadingMore ? 'Loading more models…' : 'Load more models'}
                </button>
              </div>
            )}
          </>
        )}
        </section>
      )}

      <div className="models-setup-grid">
        <div className="panel models-section-panel">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">Import a local GGUF</p>
              <h3>Bring in a model you already downloaded</h3>
            </div>
            <p className="model-summary">Keep the first step simple. Camelid can generate the internal ID, then confirm the file on first load. Support still comes from /api/capabilities, not filename optimism.</p>
          </div>

          <div className="models-form-stack">
            <div className="composer-actions import-grid">
              <input value={registerForm.name} onChange={(e) => setRegisterForm((form) => ({ ...form, name: e.target.value }))} placeholder="Model name" />
              <input value={registerForm.model_path} onChange={(e) => setRegisterForm((form) => ({ ...form, model_path: e.target.value }))} placeholder="/path/to/your-model.gguf" />
            </div>

            <button className="ghost-button subtle-action import-advanced-toggle" onClick={() => setShowImportAdvanced((current) => !current)}>
              {showImportAdvanced ? 'Hide advanced options' : 'Show advanced options'}
            </button>

            {showImportAdvanced && (
              <div className="composer-actions import-grid import-grid-advanced">
                <input value={registerForm.id} onChange={(e) => setRegisterForm((form) => ({ ...form, id: e.target.value }))} placeholder="Internal model ID (optional)" />
                <input value={registerForm.runtime_model_name} onChange={(e) => setRegisterForm((form) => ({ ...form, runtime_model_name: e.target.value }))} placeholder="Runtime name override (optional)" />
              </div>
            )}

            <div className="import-callout-row">
              <p className="model-summary">Import calls Camelid’s load endpoint immediately, saves the path locally, and records tokenizer/config/tensor readiness. Unsupported tokenizers, quants, model families, or oversized CPU materialization stay visible as typed backend errors instead of becoming chat-ready.</p>
              <button className="primary-button" onClick={registerModel} disabled={Boolean(loadingModelId)}>{loadingModelId ? 'Loading local model…' : 'Import and load local model'}</button>
            </div>
          </div>
        </div>

        <div className="panel models-section-panel">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">Hosted API setup</p>
              <h3>Planned hosted /v1-compatible API link</h3>
            </div>
            <p className="model-summary">This form preserves the intended fields, but it stays disabled because the current Camelid API does not expose hosted-provider routing.</p>
          </div>

          <fieldset className="models-form-stack models-planned-fieldset" disabled>
            <div className="composer-actions import-grid">
              <input value={externalForm.name} onChange={(e) => setExternalForm((form) => ({ ...form, name: e.target.value }))} placeholder="Display name" />
              <input value={externalForm.model_name} onChange={(e) => setExternalForm((form) => ({ ...form, model_name: e.target.value }))} placeholder="Remote model name" />
            </div>
            <div className="composer-actions import-grid import-grid-advanced">
              <input value={externalForm.source} onChange={(e) => setExternalForm((form) => ({ ...form, source: e.target.value }))} placeholder="Provider label" />
              <input value={externalForm.api_base} onChange={(e) => setExternalForm((form) => ({ ...form, api_base: e.target.value }))} placeholder="https://api.example/v1" />
            </div>
            <div className="composer-actions import-grid import-grid-advanced">
              <input value={externalForm.id} onChange={(e) => setExternalForm((form) => ({ ...form, id: e.target.value }))} placeholder="Internal model ID (optional)" />
              <input type="password" value={externalForm.api_key} onChange={(e) => setExternalForm((form) => ({ ...form, api_key: e.target.value }))} placeholder="API key" autoComplete="off" />
            </div>
            <div className="import-callout-row">
              <p className="model-summary">Hosted-provider routing is {hostedRoutingAvailable ? 'advertised by capabilities but still waiting for frontend wiring.' : 'planned, so these controls are visible for intent but disabled until Camelid has an endpoint to use them.'}</p>
              <button className="primary-button" onClick={connectExternalModel} disabled>{hostedRoutingAvailable ? 'Hosted API wiring pending' : 'Hosted API routing planned'}</button>
            </div>
          </fieldset>
        </div>
      </div>

      {apiLinkModels.length > 0 && (
        <section className="panel models-section-panel">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">API links</p>
              <h3>{apiLinkModels.length} planned hosted link{apiLinkModels.length === 1 ? '' : 's'}</h3>
            </div>
            <p className="model-summary">Camelid is keeping these visible as planned connections, but hosted-provider chat routing is disabled until an API route exists.</p>
          </div>
          <div className="models-card-grid">
            {apiLinkModels.map((model) => {
              const routingReady = isHostedRoutingAvailable(model) && hostedRoutingAvailable
              return (
                <article key={model.id} className="model-card models-model-card">
                  <div className="models-card-head">
                    <div className="models-card-title">
                      <strong>{model.name}</strong>
                      <span>{formatModelMeta(model)}</span>
                    </div>
                    <div className={`status-pill ${routingReady ? 'ready' : 'warm'}`}>{routingReady ? 'API routing advertised' : 'API routing planned'}</div>
                  </div>
                  <p className="model-summary">Hosted API links stay disabled until Camelid exposes and wires provider routing for chat.</p>
                  <div className="models-card-actions">
                    <button className="ghost-button" disabled>{routingReady ? 'Frontend wiring pending' : 'Planned, not selectable yet'}</button>
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      )}

      {setupModels.length > 0 && (
        <section className="panel models-section-panel">
          <div className="models-section-heading">
            <div>
              <p className="panel-kicker">Still needs setup</p>
              <h3>{setupModels.length} model{setupModels.length === 1 ? '' : 's'} still need attention</h3>
            </div>
            <p className="model-summary">This is the short list of models that are still importing, downloading, or need a fix before they can be used confidently.</p>
          </div>

          <div className="models-card-grid">
            {setupModels.map((model) => {
              const runtimeReady = isRunnableModel(model)
              const chatGate = getChatGateState(capabilities, model, runtime)
              const chatUnlocked = chatGate.chatUnlocked
              const contractBlocked = runtimeReady && !chatGate.contractSupported
              const canLoad = canLoadIntoRuntime(model)
              const external = isExternalModel(model)
              const selected = selectedModelId === model.id
              const active = modelRuntimeIdMatches(model, runtime)
              const loadedNow = isModelLoadedNow(model) || active
              const generationReady = isModelGenerationReady(model)
              const busy = loadingModelId === model.id
              const errorCopy = modelErrorCopy(model)

              return (
                <article key={model.id} className={`model-card models-model-card ${active ? 'active-model-card' : ''} ${selected ? 'selected-model-card' : ''}`}>
                  <div className="models-card-head">
                    <div className="models-card-title">
                      <strong>{model.name}</strong>
                      <span>{formatModelMeta(model)}</span>
                    </div>
                    <div className={`status-pill ${statusTone(model)}`}>{getModelStatusLabel(model)}</div>
                  </div>

                  <div className="models-card-tags">
                    {loadedNow && <div className={`pin-badge ${generationReady ? 'ready' : 'warm'}`}>{generationReady ? 'Loaded + generation-ready' : 'Loaded, not ready'}</div>}
                    {selected && <div className={`pin-badge ${chatUnlocked ? 'ready' : contractBlocked ? 'warm' : ''}`}>{chatUnlocked ? 'Next chat ready' : contractBlocked ? 'Next chat contract-blocked' : 'Next chat'}</div>}
                    {external && <div className="pin-badge">API planned</div>}
                    {model.model_path && !external && <div className="pin-badge">Saved locally</div>}
                  </div>

                  <div className="models-card-copy-stack">
                    <p className="model-summary">{describeModelState(model)}</p>
                    <p className="model-summary">{contractBlocked ? `${chatGate.label}: runtime is green, but chat remains blocked until this exact COMPATIBILITY.md row is supported by /api/capabilities.` : getNextStepCopy(model, { active: loadedNow, selected, runnable: runtimeReady, generationReady })}</p>
                    <p className="model-summary">{formatModelOrigin(model)}</p>
                  </div>

                  {!external && <CapabilityEvidenceBlock capabilities={capabilities} model={model} />}

                  {!external && (model.status === 'downloading' || model.status === 'canceling' || model.progress) && (
                    <div className="progress-wrap">
                      <div className="progress-bar"><div style={{ width: `${model.progress || 0}%` }} /></div>
                      <small>{formatDownloadCopy(model)}</small>
                    </div>
                  )}

                  {!external && <ReadinessGrid model={model} runtime={runtime} includePath />}

                  {errorCopy && <p className="library-error-copy">{errorCopy}</p>}

                  <div className="models-card-actions">
                    {!external && (model.status === 'not_installed' || (model.status === 'failed' && !model.model_path)) && <button className="primary-button" onClick={() => installModel(model.id)} disabled={busy}>{model.status === 'failed' ? 'Retry download' : 'Download'}</button>}
                    {!external && (model.status === 'downloading' || model.status === 'canceling') && <button className="ghost-button" onClick={() => cancelModelDownload(model.id)} disabled={model.status === 'canceling'}>{model.status === 'canceling' ? 'Canceling…' : 'Cancel download'}</button>}
                    {!external && !model.model_path && (model.status === 'ready' || model.status === 'registered') && <button className="ghost-button" disabled>Re-import with a local file path</button>}
                    {(runtimeReady || model.model_path) && <button className="ghost-button" onClick={() => setSelectedModelId(model.id)} disabled={busy}>{selected ? 'Chosen for next chat' : chatUnlocked ? 'Use for next chat' : contractBlocked ? 'Select; chat stays blocked' : runtimeReady ? 'Select and inspect contract' : 'Select, then load'}</button>}
                    {canLoad && !external && <button className="primary-button" onClick={() => activateModel(model.id)} disabled={busy || (loadedNow && generationReady)}>{busy ? 'Loading…' : loadedNow ? generationReady ? 'Loaded now' : 'Retry readiness check' : model.status === 'registered' ? 'Load now and confirm file' : 'Load now'}</button>}
                  </div>
                </article>
              )
            })}
          </div>
        </section>
      )}
    </section>
  )
}
