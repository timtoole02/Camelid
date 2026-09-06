import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ModelInspector } from '../components/models/ModelInspector'
import { TokenizerPlayground } from '../components/models/TokenizerPlayground'
import { ActiveModelBar } from '../components/models/ActiveModelBar'
import { CatalogLaneBrowse } from '../components/models/CatalogLaneBrowse'
import { DownloadsPanel } from '../components/models/DownloadsPanel'
import { ModelFamilyGroups } from '../components/models/ModelFamilyGroups'
import { UnsupportedBlocker } from '../components/models/UnsupportedBlocker'
import { Section, SupportedRow, CompatibleRow, EligibleRow, NotAnchoredRow } from '../components/models/LaneRows'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { Notice } from '../components/ui/Notice'
import { useModelsPageData } from '../hooks/useModelsPageData'
import { formatBytes } from '../lib/formatters'
import { isCompatibilityNumericalVarianceRunnableForModel, isCompatibilityVerifiedRunnableForModel } from '../lib/capabilities'
import { bucketByLane, matchModel } from '../lib/modelLanes'
import { modelSearchText, shouldOpenModelFamily } from '../lib/modelFamilies'
import { loadLocalModelForChat, modelFilenameFromPath, unloadLocalModel } from '../lib/modelActivation'
import { modelDeleteBlockedReason } from '../lib/modelDeletion'
import { IconClose, IconModels, IconRefresh, IconSearch } from '../components/ui/icons'

/* The Models page: one scroll, five zones.
     1. Active model bar — what is loaded now, with Unload.
     2. Supported — local GGUFs matching an exact supported /api/capabilities row.
     3. Other local models — runnable, ready-to-test, and unverified GGUFs with
        their distinct evidence states preserved.
     4. Downloads — one global live-progress area with cancel.
     5. Get models — curated picks + live Hugging Face search, confirmed downloads.
   Membership everywhere is DERIVED at render time from /api/models/local +
   /api/capabilities (lib/modelLanes); no hand-authored arrays place models, no
   localStorage records claim "downloaded". Diagnostics (tokenizer playground,
   metadata inspector, import-by-path) live in a collapsed disclosure at the end. */

export default function ModelsView({
  runtime,
  capabilities,
  refreshDashboard,
  onOpenChat,
  unloadCurrentModel,
  loadingModelId,
  registerForm,
  setRegisterForm,
  registerModel,
  apiBase = '',
}) {
  const catalogApiBase = (runtime?.api_base || '').replace(/\/$/, '')
  const runtimeOnline = runtime?.status === 'online'
  const catalogInstallAvailable = Boolean(
    capabilities?.model_catalog_install || capabilities?.model_downloads || capabilities?.hf_catalog_install,
  )

  /* Single data spine: /api/models/local + /api/models/current + downloads. */
  const spine = useModelsPageData({ apiBase: catalogApiBase || apiBase })
  const [receipts, setReceipts] = useState({})
  const [smokeBusy, setSmokeBusy] = useState({})
  const [usingFilename, setUsingFilename] = useState('')
  const [unloading, setUnloading] = useState(false)
  const [verification, setVerification] = useState(null)
  const [verificationBusy, setVerificationBusy] = useState(false)
  const verificationRequestRef = useRef(0)
  // Typed fail-closed blocker from a pre-load inspect ({ code, message }), shown
  // verbatim instead of attempting a multi-GB load that cannot run.
  const [blocker, setBlocker] = useState(null)
  const [laneError, setLaneError] = useState('')
  const [cancelingDownloads, setCancelingDownloads] = useState(new Set())
  const [canceledCatalogIds, setCanceledCatalogIds] = useState(new Set())
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [importing, setImporting] = useState(false)
  const [pendingDeleteEntry, setPendingDeleteEntry] = useState(null)
  const [deletingFilename, setDeletingFilename] = useState('')
  const [defaultingFilename, setDefaultingFilename] = useState('')
  const [deleteNotice, setDeleteNotice] = useState('')
  const [catalogOperations, setCatalogOperations] = useState(new Set())
  const [modelQuery, setModelQuery] = useState('')
  const [quantizeModalOpen, setQuantizeModalOpen] = useState(false)
  const [quantizeInputModel, setQuantizeInputModel] = useState('')
  const [quantizeOutputPath, setQuantizeOutputPath] = useState('')
  const [quantizeType, setQuantizeType] = useState('q4_k_m')
  const [quantizeBusy, setQuantizeBusy] = useState(false)
  const [quantizeResult, setQuantizeResult] = useState(null)
  const [quantizeError, setQuantizeError] = useState('')

  const handleQuantize = async (e) => {
    e.preventDefault()
    if (!quantizeInputModel.trim()) return
    setQuantizeBusy(true)
    setQuantizeError('')
    setQuantizeResult(null)
    try {
      const res = await fetch(`${catalogApiBase || apiBase}/api/models/quantize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: quantizeInputModel.trim(),
          output_path: quantizeOutputPath.trim() || undefined,
          quant_type: quantizeType,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data?.error?.message || data?.message || `Quantization failed (HTTP ${res.status})`)
      }
      setQuantizeResult(data)
      spine.refreshAll()
    } catch (err) {
      setQuantizeError(err.message || 'Quantization failed')
    } finally {
      setQuantizeBusy(false)
    }
  }
  /* How many curated catalog rows the current term matches, reported up by
     CatalogLaneBrowse so the result line can say whether scrolling is worth it. */
  const [catalogMatchCount, setCatalogMatchCount] = useState(null)
  const loadInFlightRef = useRef('')
  // Catalog downloads are intentionally parallel. Their completion order is
  // nondeterministic, so serialize only the short model-transition step after
  // each download finishes. This preserves every requested start instead of
  // rejecting whichever download happened to complete second.
  const loadQueueRef = useRef(Promise.resolve())

  const laneBuckets = useMemo(
    () => (spine.local ? bucketByLane(spine.local.models, capabilities) : null),
    [spine.local, capabilities],
  )
  const activeEntry = useMemo(
    () => spine.local?.models.find((m) => m.filename === spine.activeFilename) || null,
    [spine.local, spine.activeFilename],
  )
  const ghostMoePreparedFilenames = useMemo(
    () => new Set((spine.local?.models || []).filter((model) => model.ghost_moe_prepared).map((model) => model.filename)),
    [spine.local],
  )
  /* Search the same identity the row exposes: display name, filename,
     architecture, and derived family label. The latter two matter for renamed
     local files whose scan rows do not carry a display name. */
  const matchesModelQuery = (model) => {
    const needle = modelQuery.trim().toLowerCase()
    if (!needle) return true
    return modelSearchText(model).includes(needle)
  }
  const supportedRows = (laneBuckets ? laneBuckets.supported : []).filter(matchesModelQuery)
  /* Keep the experimental sub-lanes separate after filtering. Their row types
     expose different actions, so flattening only for the count and then mapping
     the raw buckets would make the badge say "1" while unrelated rows remained
     visible. Broad family terms such as "qwen" must render every Qwen2/Qwen3/4B
     match and nothing outside that family. */
  const compatibleRows = (laneBuckets ? laneBuckets.compatible : []).filter(matchesModelQuery)
  const eligibleRows = (laneBuckets ? laneBuckets.eligible : []).filter(matchesModelQuery)
  const notAnchoredRows = (laneBuckets ? laneBuckets.not_anchored : []).filter(matchesModelQuery)
  const experimentalRows = [
    ...compatibleRows.map((entry) => ({ ...entry, _familyLane: 'compatible' })),
    ...eligibleRows.map((entry) => ({ ...entry, _familyLane: 'eligible' })),
    ...notAnchoredRows.map((entry) => ({ ...entry, _familyLane: 'not_anchored' })),
  ]
  const filteringModels = Boolean(modelQuery.trim())
  const localMatchCount = supportedRows.length + experimentalRows.length
  const localFamilyInitiallyOpen = (group) => shouldOpenModelFamily(group, {
    filtering: filteringModels,
    activeFilename: spine.activeFilename,
    loadedModelIds: spine.loadedModelIds,
  })
  const deleteBlockedReason = modelDeleteBlockedReason({
    activeFilename: spine.activeFilename,
    residentModelsLoaded: spine.loadedModelIds.size > 0,
    downloads: spine.downloads,
    loading: Boolean(usingFilename || loadingModelId || importing || unloading),
    smoking: Object.values(smokeBusy).some(Boolean) || catalogOperations.size > 0,
  })

  const setCatalogOperationBusy = useCallback((catalogId, busy) => {
    setCatalogOperations((current) => {
      const next = new Set(current)
      if (busy) next.add(catalogId)
      else next.delete(catalogId)
      return next
    })
  }, [])

  // Load a local model into the chat backend. The HTTP sequence itself (header-only
  // inspect -> authoritative load -> identity + readiness confirmation) lives in
  // lib/modelActivation so this page and the first-run card cannot drift; what stays
  // here is the page's own state wiring. The spine's `/api/models/current` refresh
  // answers the identity check, so the confirmation costs no extra request.
  const loadModelForChat = (filename, { onStage, model = null } = {}) => {
    const run = async () => {
      loadInFlightRef.current = filename
      setUsingFilename(filename)
      setLaneError('')
      setBlocker(null)
      try {
        const result = await loadLocalModelForChat({
          apiBase: spine.base,
          filename,
          model: model || spine.local?.models.find((entry) => entry.filename === filename) || null,
          onStage,
          readActiveFilename: async () => modelFilenameFromPath((await spine.refreshCurrent())?.path),
        })
        if (!result.ok) {
          if (result.blocker) setBlocker(result.blocker)
          setLaneError(result.message)
          return result
        }
        await Promise.all([
          spine.refreshLoadedModels(),
          refreshDashboard?.({ silent: true }),
        ])
        return result
      } finally {
        if (loadInFlightRef.current === filename) loadInFlightRef.current = ''
        setUsingFilename('')
      }
    }

    const queued = loadQueueRef.current.then(run, run)
    loadQueueRef.current = queued.catch(() => {})
    return queued
  }

  const unloadEmbeddingModel = async (filename) => {
    if (loadInFlightRef.current) return
    loadInFlightRef.current = filename
    setUsingFilename(filename)
    setLaneError('')
    try {
      const result = await unloadLocalModel({ apiBase: spine.base, modelId: filename })
      if (!result.ok) {
        setLaneError(result.message)
        return
      }
      await Promise.all([
        spine.refreshLoadedModels(),
        spine.refreshCurrent(),
        refreshDashboard?.({ silent: true }),
      ])
    } finally {
      if (loadInFlightRef.current === filename) loadInFlightRef.current = ''
      setUsingFilename('')
    }
  }

  const handleUnload = async () => {
    setUnloading(true)
    try {
      await unloadCurrentModel()
      await spine.refreshCurrent()
    } finally {
      setUnloading(false)
    }
  }

  const refreshVerification = async () => {
    const requestId = ++verificationRequestRef.current
    if (!spine.activeFilename) {
      setVerification(null)
      return
    }
    try {
      const res = await fetch(`${spine.base}/api/models/verify`)
      const next = res.ok ? await res.json() : null
      if (requestId === verificationRequestRef.current) setVerification(next)
    } catch {
      if (requestId === verificationRequestRef.current) setVerification(null)
    }
  }

  useEffect(() => {
    refreshVerification()
  }, [spine.activeFilename, spine.base])

  const runVerification = async () => {
    setVerificationBusy(true)
    setLaneError('')
    try {
      const res = await fetch(`${spine.base}/api/models/verify`, { method: 'POST' })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(body?.error?.message || `verification failed (HTTP ${res.status})`)
      await refreshVerification()
    } catch (err) {
      setLaneError(String(err?.message || err))
    } finally {
      setVerificationBusy(false)
    }
  }

  const cancelDownloadById = async (id) => {
    setCancelingDownloads((s) => new Set([...s, id]))
    try {
      const canceled = await spine.cancelDownload(id)
      if (canceled) {
        setCanceledCatalogIds((current) => new Set([...current, id]))
      }
    } finally {
      setCancelingDownloads((s) => {
        const next = new Set(s)
        next.delete(id)
        return next
      })
    }
  }

  const clearCanceledDownload = (catalogId) => {
    setCanceledCatalogIds((current) => {
      if (!current.has(catalogId)) return current
      const next = new Set(current)
      next.delete(catalogId)
      return next
    })
  }

  const requestDeleteModel = (entry) => {
    if (deleteBlockedReason) {
      setLaneError(deleteBlockedReason)
      return
    }
    setLaneError('')
    setDeleteNotice('')
    setPendingDeleteEntry(entry)
  }

  const makeDefaultModel = async (filename) => {
    setDefaultingFilename(filename)
    setLaneError('')
    setDeleteNotice('')
    try {
      await spine.setDefaultModel(filename)
      setDeleteNotice(`${filename} will load automatically when Camelid starts.`)
    } catch (error) {
      setLaneError(String(error?.message || error))
    } finally {
      setDefaultingFilename('')
    }
  }

  useEffect(() => {
    if (pendingDeleteEntry && deleteBlockedReason && !deletingFilename) {
      setPendingDeleteEntry(null)
      setLaneError(deleteBlockedReason)
    }
  }, [deleteBlockedReason, deletingFilename, pendingDeleteEntry])

  const deleteModelFromDisk = async () => {
    if (!pendingDeleteEntry || deletingFilename) return
    if (deleteBlockedReason) {
      setPendingDeleteEntry(null)
      setLaneError(deleteBlockedReason)
      return
    }
    const entry = pendingDeleteEntry
    setDeletingFilename(entry.filename)
    setLaneError('')
    try {
      const result = await spine.deleteLocalModel(entry)
      setReceipts((current) => {
        const next = { ...current }
        delete next[entry.filename]
        return next
      })
      setPendingDeleteEntry(null)
      setDeleteNotice(result.bytes_freed
        ? `Deleted ${entry.filename} and freed ${formatBytes(result.bytes_freed)}.`
        : `Deleted ${entry.filename}.`)
    } catch (error) {
      setPendingDeleteEntry(null)
      setLaneError(String(error?.message || error))
    } finally {
      setDeletingFilename('')
    }
  }

  const runSmoke = async (filename) => {
    setSmokeBusy((b) => ({ ...b, [filename]: true }))
    setLaneError('')
    try {
      const res = await fetch(`${spine.base}/api/models/runnable-smoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      })
      const body = await res.json()
      if (res.ok && body.passed) {
        setReceipts((r) => ({ ...r, [filename]: body.receipt }))
        await spine.refreshLocal()
      } else {
        setLaneError(body?.error?.message || `The quick test did not pass for ${filename}.`)
      }
    } catch (err) {
      setLaneError(String(err?.message || err))
    } finally {
      setSmokeBusy((b) => ({ ...b, [filename]: false }))
    }
  }

  // Pull the runnable receipt for each Compatible model (those that passed smoke).
  useEffect(() => {
    if (!spine.local) return
    spine.local.models
      .filter((m) => m.runnable_receipt_present && !receipts[m.filename])
      .forEach(async (m) => {
        try {
          const res = await fetch(
            `${spine.base}/api/models/runnable-receipt?filename=${encodeURIComponent(m.filename)}`,
          )
          if (res.ok) {
            const receipt = await res.json()
            setReceipts((r) => ({ ...r, [m.filename]: receipt }))
          }
        } catch {
          /* receipt is best-effort; the row still renders */
        }
      })
  }, [spine.local, spine.base, receipts])

  const importFromPath = async () => {
    setImporting(true)
    try {
      await registerModel()
      await spine.refreshAll()
    } finally {
      setImporting(false)
    }
  }

  return (
    <section className="models-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconModels size={14} /> Model support</p>
          <h1>Models</h1>
          <p className="cxv-sub">
            Load, download, and manage the models on this machine.
          </p>
        </div>
        <div className="cxv-head__actions models-head-actions">
          <label className="cxv-search models-head-search">
            <IconSearch size={17} />
            <input
              value={modelQuery}
              onChange={(event) => setModelQuery(event.target.value)}
              placeholder="Search models by name"
              aria-label="Search models on this machine by name"
              type="search"
            />
            {modelQuery && (
              <button
                type="button"
                className="cxv-search__clear"
                aria-label="Clear model search"
                onClick={() => setModelQuery('')}
              >
                <IconClose size={14} />
              </button>
            )}
          </label>
          <Button
            variant="outline"
            size="sm"
            icon={<IconRefresh size={16} />}
            onClick={() => spine.refreshAll()}
            disabled={spine.localLoading}
          >
            {spine.localLoading ? 'Refreshing…' : 'Refresh'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setQuantizeResult(null)
              setQuantizeError('')
              if (spine.local?.models?.length && !quantizeInputModel) {
                const first = spine.local.models[0]
                setQuantizeInputModel(first.path || first.filename)
              }
              setQuantizeModalOpen(true)
            }}
          >
            Quantize GGUF
          </Button>
        </div>
      </header>

      {/* One result line for the whole page rather than a changed empty state in
          each section: the filter spans Supported and Experimental, so saying it
          once is clearer than saying it twice. A search that matches nothing
          locally is the moment someone is most likely looking for a model they
          do not have yet, so it hands the term to the catalog instead of just
          reporting failure. */}
      {filteringModels && (
        <div className="models-filter-summary" role="status">
          <span>
            {localMatchCount > 0
              ? <>{localMatchCount} on this machine {localMatchCount === 1 ? 'matches' : 'match'} <strong>{modelQuery.trim()}</strong></>
              : <>Nothing installed matches <strong>{modelQuery.trim()}</strong></>}
            {catalogMatchCount !== null && catalogMatchCount > 0 && (
              <> · {catalogMatchCount} to download in Get models</>
            )}
          </span>
          <div className="models-filter-summary__actions">
            {catalogMatchCount !== null && catalogMatchCount > 0 && (
              <Button
                variant="tonal"
                size="sm"
                icon={<IconSearch size={15} />}
                onClick={() => document.querySelector('.catalog-lane-browse')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
              >
                Jump to Get models
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => setModelQuery('')}>Clear</Button>
          </div>
        </div>
      )}

      {/* Zone 1 — active model bar */}
      <ActiveModelBar
        runtime={runtime}
        activeFilename={spine.activeFilename}
        activeEntry={activeEntry}
        capabilities={capabilities}
        busy={unloading || verificationBusy || Boolean(loadingModelId)}
        unloading={unloading}
        verification={verification}
        verificationBusy={verificationBusy}
        onVerify={runVerification}
        onUnload={handleUnload}
      />
      <Notice notice={laneError} tone="error" onDismiss={() => setLaneError('')} />
      <Notice notice={deleteNotice} tone="success" onDismiss={() => setDeleteNotice('')} />
      {deleteBlockedReason ? (
        <p className="lane-delete-guard" id="model-delete-guard">{deleteBlockedReason}</p>
      ) : null}
      {spine.localError && !spine.local ? (
        <Notice notice={`Could not list local models: ${spine.localError}`} tone="error" />
      ) : null}

      {/* Zone 2 — supported local models (derived membership only) */}
      <Section
        title="Supported"
        count={laneBuckets ? supportedRows.length : undefined}
        subtitle="Verified to run correctly here."
      >
        {!laneBuckets ? (
          <p className="lane-empty">
            {spine.localLoading ? 'Scanning local models…' : runtimeOnline ? 'Local model scan unavailable.' : 'Runtime offline — the local scan resumes when the backend is back.'}
          </p>
        ) : supportedRows.length ? (
          <ModelFamilyGroups
            items={supportedRows}
            initiallyOpen={localFamilyInitiallyOpen}
            renderItem={(m) => (
              <SupportedRow
                key={m.filename}
                entry={m}
                active={m.filename === spine.activeFilename}
                resident={m.filename === spine.activeFilename || spine.loadedModelIds.has(m.filename)}
                busy={usingFilename === m.filename}
                deleteBusy={deletingFilename === m.filename}
                defaultBusy={defaultingFilename === m.filename}
                isDefault={spine.defaultFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onUse={() => loadModelForChat(m.filename)}
                onUnload={unloadEmbeddingModel}
                onDelete={requestDeleteModel}
                onMakeDefault={makeDefaultModel}
              />
            )}
          />
        ) : (
          <p className="lane-empty">{filteringModels ? 'No verified model matches this search.' : 'No verified models on this machine yet — download one below in “Get models”.'}</p>
        )}
      </Section>

      {/* Zone 3 — everything else local, honestly labeled by evidence state */}
      <Section
        title="Other local models"
        count={laneBuckets ? experimentalRows.length : undefined}
        subtitle="Verification varies by exact row; each model shows what has actually passed."
      >
        {blocker ? <UnsupportedBlocker blocker={blocker} className="local-lane-blocker" /> : null}
        {!laneBuckets ? (
          <p className="lane-empty">
            {spine.localLoading ? 'Scanning local models…' : runtimeOnline ? 'Local model scan unavailable.' : 'Runtime offline — the local scan resumes when the backend is back.'}
          </p>
        ) : experimentalRows.length ? (
          <ModelFamilyGroups
            items={experimentalRows}
            initiallyOpen={localFamilyInitiallyOpen}
            renderItem={(m) => m._familyLane === 'compatible' ? (
              <CompatibleRow
                key={m.filename}
                entry={m}
                receipt={receipts[m.filename]}
                exactRowVerified={isCompatibilityVerifiedRunnableForModel(capabilities, matchModel(m))}
                numericalVariance={m.lane_class === 'runnable_with_variance' || isCompatibilityNumericalVarianceRunnableForModel(capabilities, matchModel(m))}
                active={m.filename === spine.activeFilename}
                resident={m.filename === spine.activeFilename || spine.loadedModelIds.has(m.filename)}
                busy={usingFilename === m.filename}
                deleteBusy={deletingFilename === m.filename}
                defaultBusy={defaultingFilename === m.filename}
                isDefault={spine.defaultFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onUse={() => loadModelForChat(m.filename)}
                onUnload={unloadEmbeddingModel}
                onDelete={requestDeleteModel}
                onMakeDefault={makeDefaultModel}
              />
            ) : m._familyLane === 'eligible' ? (
              <EligibleRow
                key={m.filename}
                entry={m}
                busy={Boolean(smokeBusy[m.filename])}
                deleteBusy={deletingFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onRun={() => runSmoke(m.filename)}
                onDelete={requestDeleteModel}
              />
            ) : (
              <NotAnchoredRow
                key={m.filename}
                entry={m}
                active={m.filename === spine.activeFilename}
                resident={m.filename === spine.activeFilename || spine.loadedModelIds.has(m.filename)}
                busy={usingFilename === m.filename}
                deleteBusy={deletingFilename === m.filename}
                defaultBusy={defaultingFilename === m.filename}
                isDefault={spine.defaultFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onUse={() => loadModelForChat(m.filename)}
                onUnload={unloadEmbeddingModel}
                onDelete={requestDeleteModel}
                onMakeDefault={makeDefaultModel}
              />
            )}
          />
        ) : (
          <p className="lane-empty">{filteringModels ? 'No other local model matches this search.' : 'Every downloaded model is in the Supported section.'}</p>
        )}
      </Section>

      {/* Zone 4 — downloads in progress (global; hidden while idle) */}
      <DownloadsPanel
        downloads={spine.downloads}
        cancelingIds={cancelingDownloads}
        onCancel={cancelDownloadById}
      />

      {/* Zone 5 — get models: curated picks + live Hugging Face search */}
      <CatalogLaneBrowse
        externalQuery={modelQuery}
        onCuratedMatchCount={setCatalogMatchCount}
        apiBase={catalogApiBase || apiBase}
        capabilities={capabilities}
        localFilenames={spine.localFilenames}
        ghostMoePreparedFilenames={ghostMoePreparedFilenames}
        downloads={spine.downloads}
        installAvailable={runtimeOnline && catalogInstallAvailable}
        installBlockedReason={
          !runtimeOnline
            ? 'The runtime is offline — start the Camelid backend to download models.'
            : 'This backend does not support model downloads.'
        }
        onInstallStarted={spine.kickDownloadsPoll}
        onDownloadAcknowledged={spine.refreshDownloads}
        onAcquired={spine.refreshLocalAndDefault}
        canceledCatalogIds={canceledCatalogIds}
        onDownloadRetry={clearCanceledDownload}
        onStartModel={loadModelForChat}
        onModelStarted={onOpenChat}
        onOperationBusy={setCatalogOperationBusy}
      />

      {/* Diagnostics — operator tools, collapsed by default. Import-by-path lives
          here because it is the only way to load a GGUF stored outside models/. */}
      <details className="models-diagnostics">
        <summary>Diagnostics</summary>
        <div className="models-diagnostics__body">
          <div className="models-diagnostics__tools">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setInspectorOpen(true)}
              title="View the loaded model's metadata, tokenizer, and tensors"
            >
              Inspect loaded model metadata
            </Button>
          </div>

          <div className="models-diagnostics__import">
            <h3>Import a GGUF by path</h3>
            <p className="lane-empty">
              Models inside the <code>models/</code> folder appear above automatically. Use this only
              for a model file stored elsewhere; it loads immediately and is verified the same way as
              the models above.
            </p>
            <div className="models-diagnostics__import-grid">
              <input
                value={registerForm.name}
                onChange={(e) => setRegisterForm((form) => ({ ...form, name: e.target.value }))}
                placeholder="Model name"
              />
              <input
                value={registerForm.model_path}
                onChange={(e) => setRegisterForm((form) => ({ ...form, model_path: e.target.value }))}
                placeholder="/path/to/your-model.gguf"
              />
              <Button
                variant="tonal"
                size="sm"
                onClick={importFromPath}
                loading={importing || Boolean(loadingModelId)}
                disabled={importing || Boolean(loadingModelId)}
              >
                Import and load
              </Button>
            </div>
          </div>

          <TokenizerPlayground apiBase={catalogApiBase || apiBase} />
        </div>
      </details>

      {inspectorOpen && (
        <ModelInspector apiBase={catalogApiBase || apiBase} onClose={() => setInspectorOpen(false)} />
      )}

      <ConfirmDialog
        open={Boolean(pendingDeleteEntry)}
        title={pendingDeleteEntry ? `Delete ${pendingDeleteEntry.filename}?` : 'Delete model?'}
        detail={pendingDeleteEntry
          ? `This permanently removes ${pendingDeleteEntry.size_bytes ? formatBytes(pendingDeleteEntry.size_bytes) : 'this file'}${pendingDeleteEntry.ghost_moe_prepared ? ' and its Ghost MoE expert pack' : ''} from disk. This cannot be undone.`
          : ''}
        confirmLabel="Delete model"
        busy={Boolean(deletingFilename)}
        onCancel={() => { if (!deletingFilename) setPendingDeleteEntry(null) }}
        onConfirm={deleteModelFromDisk}
      />

      {quantizeModalOpen && (
        <div className="citation-modal-overlay" onClick={() => !quantizeBusy && setQuantizeModalOpen(false)}>
          <div className="citation-modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 640 }}>
            <div className="citation-modal__header">
              <div className="citation-modal__title">
                <strong>Local GGUF Quantizer Utility</strong>
              </div>
              <button
                type="button"
                className="cxturn__action cxturn__action--icon"
                disabled={quantizeBusy}
                onClick={() => setQuantizeModalOpen(false)}
              >
                <IconClose size={16} />
              </button>
            </div>
            <form onSubmit={handleQuantize}>
              <div className="citation-modal__body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <p style={{ margin: 0, color: 'var(--color-text-muted)', fontSize: 13 }}>
                  Quantize uncompressed FP16/BF16/Q8_0 models directly into high-efficiency Q4_K_M or Q8_0 weights with bit-exact parity validation.
                </p>

                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 13 }}>
                  <span>Source GGUF Model Path or Selection:</span>
                  <input
                    className="input"
                    type="text"
                    required
                    value={quantizeInputModel}
                    onChange={(e) => setQuantizeInputModel(e.target.value)}
                    placeholder="C:\path\to\model.gguf or filename"
                  />
                  {spine.local?.models?.length > 0 && (
                    <select
                      className="input"
                      style={{ marginTop: 4 }}
                      onChange={(e) => {
                        if (e.target.value) setQuantizeInputModel(e.target.value)
                      }}
                      defaultValue=""
                    >
                      <option value="">-- Or select from local models --</option>
                      {spine.local.models.map((m) => (
                        <option key={m.filename} value={m.path || m.filename}>
                          {m.filename} ({formatBytes(m.size_bytes)})
                        </option>
                      ))}
                    </select>
                  )}
                </label>

                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 13 }}>
                  <span>Target Quantization:</span>
                  <select
                    className="input"
                    value={quantizeType}
                    onChange={(e) => setQuantizeType(e.target.value)}
                  >
                    <option value="q4_k_m">Q4_K_M (Recommended: 256-elem superblocks, optimal quality/RAM balance)</option>
                    <option value="q8_0">Q8_0 (32-elem blocks, near FP16 precision)</option>
                    <option value="q4_0">Q4_0 (Standard legacy 4-bit)</option>
                  </select>
                </label>

                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 13 }}>
                  <span>Output Destination Path (optional):</span>
                  <input
                    className="input"
                    type="text"
                    value={quantizeOutputPath}
                    onChange={(e) => setQuantizeOutputPath(e.target.value)}
                    placeholder="Defaults to <source>-<QUANT>.gguf in the same directory"
                  />
                </label>

                {quantizeError && (
                  <div style={{ color: 'var(--color-error, #f87171)', fontSize: 13 }}>
                    Error: {quantizeError}
                  </div>
                )}

                {quantizeResult && (
                  <div style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: 8, padding: 12, fontSize: 13 }}>
                    <strong style={{ color: '#4ade80' }}>Quantization Succeeded!</strong>
                    <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
                      <div>Saved: <code>{quantizeResult.output_path}</code></div>
                      <div>Size: {formatBytes(quantizeResult.input_bytes)} → {formatBytes(quantizeResult.output_bytes)} ({((1 - quantizeResult.compression_ratio) * 100).toFixed(1)}% savings)</div>
                      <div>Tensors Quantized: {quantizeResult.quantized_tensors} / {quantizeResult.tensor_count}</div>
                      <div>SHA-256: <code style={{ fontSize: 11 }}>{quantizeResult.sha256}</code></div>
                    </div>
                  </div>
                )}
              </div>
              <div className="citation-modal__footer" style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
                <Button variant="outline" size="sm" type="button" disabled={quantizeBusy} onClick={() => setQuantizeModalOpen(false)}>
                  {quantizeResult ? 'Done' : 'Cancel'}
                </Button>
                <Button variant="primary" size="sm" type="submit" loading={quantizeBusy} disabled={quantizeBusy || !quantizeInputModel.trim()}>
                  {quantizeBusy ? 'Quantizing Model…' : 'Start Quantization'}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </section>
  )
}
