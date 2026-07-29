import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ModelInspector } from '../components/models/ModelInspector'
import { TokenizerPlayground } from '../components/models/TokenizerPlayground'
import { ActiveModelBar } from '../components/models/ActiveModelBar'
import { CatalogLaneBrowse } from '../components/models/CatalogLaneBrowse'
import { DownloadsPanel } from '../components/models/DownloadsPanel'
import { UnsupportedBlocker } from '../components/models/UnsupportedBlocker'
import { Section, SupportedRow, CompatibleRow, EligibleRow, NotAnchoredRow, prettySize } from '../components/models/LaneRows'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { useModelsPageData } from '../hooks/useModelsPageData'
import { bucketByLane } from '../lib/modelLanes'
import { modelDeleteBlockedReason } from '../lib/modelDeletion'
import { IconModels } from '../components/ui/icons'

/* The Models page: one scroll, five zones.
     1. Active model bar — what is loaded now, with Unload.
     2. Supported — local GGUFs matching an exact supported /api/capabilities row.
     3. Experimental — every other local GGUF, honestly labeled by evidence state.
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
  const loadInFlightRef = useRef('')

  const laneBuckets = useMemo(
    () => (spine.local ? bucketByLane(spine.local.models, capabilities) : null),
    [spine.local, capabilities],
  )
  const activeEntry = useMemo(
    () => spine.local?.models.find((m) => m.filename === spine.activeFilename) || null,
    [spine.local, spine.activeFilename],
  )
  const experimentalRows = laneBuckets
    ? [...laneBuckets.compatible, ...laneBuckets.eligible, ...laneBuckets.not_anchored]
    : []
  const deleteBlockedReason = modelDeleteBlockedReason({
    activeFilename: spine.activeFilename,
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

  const filenameFromPath = (value) => String(value || '').split(/[\\/]/).pop() || ''

  // Load a local model into the chat backend. First predict the lane with a
  // header-only inspect (no multi-GB read): if the architecture is not implemented,
  // surface the exact typed blocker and stop — never attempt to run it. Implemented
  // architectures (supported or experimental) load as before.
  const loadModelForChat = async (filename, { onStage } = {}) => {
    if (loadInFlightRef.current) {
      const message = loadInFlightRef.current === filename
        ? `${filename} is already loading.`
        : `Wait for ${loadInFlightRef.current} to finish loading, then retry.`
      setLaneError(message)
      return { ok: false, stage: 'loading', message }
    }
    loadInFlightRef.current = filename
    setUsingFilename(filename)
    setLaneError('')
    setBlocker(null)
    // Send a models-relative path, NOT the engine's absolute models_dir joined
    // with '/'. On Windows the reported models_dir can be a `\\?\` verbatim path,
    // and Win32 does not normalize a '/' inside a verbatim path, so the old
    // concatenation produced an invalid name (os error 123). The backend resolves
    // this relative path against its configured models directory.
    const path = `models/${filename}`
    let activeStage = 'checking'
    try {
      onStage?.('checking')
      const inspectRes = await fetch(`${spine.base}/api/models/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })
      const inspect = await inspectRes.json().catch(() => ({}))
      if (!inspectRes.ok) {
        const message = inspect?.error?.message || `model inspection failed (HTTP ${inspectRes.status})`
        if (inspect?.error?.code) setBlocker({ code: inspect.error.code, message })
        setLaneError(message)
        return { ok: false, stage: 'checking', message }
      }
      if (inspect?.blocker) {
        setBlocker(inspect.blocker)
        setLaneError(inspect.blocker.message)
        return { ok: false, stage: 'checking', message: inspect.blocker.message }
      }
      const embeddingModel = inspect?.architecture === 'nomic-bert'
      // Only an inspected, implemented model reaches the authoritative load.
      activeStage = 'loading'
      onStage?.('loading')
      // replace: picking a model in the app is a SWAP, not a second resident
      // copy. Without it the previous model keeps its RAM and VRAM, and the
      // fit preflight refuses the new one on a host that could hold it alone.
      // The engine releases the old model first, then re-probes live memory.
      const res = await fetch(`${spine.base}/api/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: filename,
          path,
          replace: !embeddingModel,
          set_active: !embeddingModel,
        }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        // A typed fail-closed load error (e.g. invalid metadata) becomes a blocker.
        if (body?.error?.code && body.error.code !== 'invalid_model') {
          setBlocker({ code: body.error.code, message: body.error.message })
          return { ok: false, stage: 'loading', message: body.error.message }
        }
        throw new Error(body?.error?.message || `load failed (HTTP ${res.status})`)
      }
      if (embeddingModel) {
        const probeRes = await fetch(`${spine.base}/v1/embeddings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: filename,
            input: 'search_query: Camelid embedding readiness probe',
            dimensions: 256,
          }),
        })
        const probe = await probeRes.json().catch(() => ({}))
        if (!probeRes.ok || probe?.data?.[0]?.embedding?.length !== 256) {
          throw new Error(
            probe?.error?.message
              || `Camelid registered ${filename}, but its embedding runtime did not pass readiness.`,
          )
        }
        await refreshDashboard?.({ silent: true })
        return { ok: true, embedding: true }
      }
      const current = await spine.refreshCurrent()
      if (filenameFromPath(current?.path) !== filename) {
        throw new Error(`Camelid loaded the request but did not confirm ${filename} as the active model.`)
      }
      const healthRes = await fetch(`${spine.base}/v1/health`)
      const health = await healthRes.json().catch(() => ({}))
      if (!healthRes.ok) {
        throw new Error(health?.error?.message || `readiness check failed (HTTP ${healthRes.status})`)
      }
      if (!health.loaded_now || !health.generation_ready || health.active_model_id !== filename) {
        throw new Error(`Camelid loaded ${filename}, but it is not generation-ready yet.`)
      }
      await refreshDashboard?.({ silent: true })
      return { ok: true }
    } catch (err) {
      const message = String(err?.message || err)
      setLaneError(message)
      return { ok: false, stage: activeStage, message }
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
      setDeleteNotice(`${filename} will load automatically the next time Camelid starts.`)
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
      setDeleteNotice(`Deleted ${entry.filename} and freed ${prettySize(result.bytes_freed)}.`)
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
        setLaneError(body?.error?.message || `Smoke-admission did not pass for ${filename}.`)
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
            Load, download, and manage local GGUF models. Section membership is derived live from the
            disk scan and the /api/capabilities support contract.
          </p>
        </div>
        <div className="cxv-head__actions">
          <button
            type="button"
            className="lane-refresh"
            onClick={() => spine.refreshAll()}
            disabled={spine.localLoading}
          >
            {spine.localLoading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </header>

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
      {laneError ? <p className="lane-error">{laneError}</p> : null}
      {deleteNotice ? <p className="lane-delete-success" role="status">{deleteNotice}</p> : null}
      {deleteBlockedReason ? (
        <p className="lane-delete-guard" id="model-delete-guard">{deleteBlockedReason}</p>
      ) : null}
      {spine.localError && !spine.local ? (
        <p className="lane-empty">Could not list local models: {spine.localError}</p>
      ) : null}

      {/* Zone 2 — supported local models (derived membership only) */}
      <Section
        title="Supported"
        count={laneBuckets ? laneBuckets.supported.length : undefined}
        subtitle="Local models matching an exact supported /api/capabilities row — cross-validated parity."
      >
        {!laneBuckets ? (
          <p className="lane-empty">
            {spine.localLoading ? 'Scanning local models…' : runtimeOnline ? 'Local model scan unavailable.' : 'Runtime offline — the local scan resumes when the backend is back.'}
          </p>
        ) : laneBuckets.supported.length ? (
          laneBuckets.supported.map((m) => (
            <SupportedRow
              key={m.filename}
              entry={m}
              active={m.filename === spine.activeFilename}
              busy={usingFilename === m.filename}
              deleteBusy={deletingFilename === m.filename}
              defaultBusy={defaultingFilename === m.filename}
              isDefault={spine.defaultFilename === m.filename}
              blockedReason={deleteBlockedReason}
              onUse={() => loadModelForChat(m.filename)}
              onDelete={requestDeleteModel}
              onMakeDefault={makeDefaultModel}
            />
          ))
        ) : (
          <p className="lane-empty">No local model matches a supported row yet — download one below in “Get models”.</p>
        )}
      </Section>

      {/* Zone 3 — everything else local, honestly labeled by evidence state */}
      <Section
        title="Experimental"
        count={laneBuckets ? experimentalRows.length : undefined}
        subtitle="These run without parity anchoring — output is not cross-validated against the reference."
      >
        {blocker ? <UnsupportedBlocker blocker={blocker} className="local-lane-blocker" /> : null}
        {!laneBuckets ? (
          <p className="lane-empty">
            {spine.localLoading ? 'Scanning local models…' : runtimeOnline ? 'Local model scan unavailable.' : 'Runtime offline — the local scan resumes when the backend is back.'}
          </p>
        ) : experimentalRows.length ? (
          <>
            {laneBuckets.compatible.map((m) => (
              <CompatibleRow
                key={m.filename}
                entry={m}
                receipt={receipts[m.filename]}
                busy={usingFilename === m.filename}
                deleteBusy={deletingFilename === m.filename}
                defaultBusy={defaultingFilename === m.filename}
                isDefault={spine.defaultFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onUse={() => loadModelForChat(m.filename)}
                onDelete={requestDeleteModel}
                onMakeDefault={makeDefaultModel}
              />
            ))}
            {laneBuckets.eligible.map((m) => (
              <EligibleRow
                key={m.filename}
                entry={m}
                busy={Boolean(smokeBusy[m.filename])}
                deleteBusy={deletingFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onRun={() => runSmoke(m.filename)}
                onDelete={requestDeleteModel}
              />
            ))}
            {laneBuckets.not_anchored.map((m) => (
              <NotAnchoredRow
                key={m.filename}
                entry={m}
                busy={usingFilename === m.filename}
                deleteBusy={deletingFilename === m.filename}
                defaultBusy={defaultingFilename === m.filename}
                isDefault={spine.defaultFilename === m.filename}
                blockedReason={deleteBlockedReason}
                onUse={() => loadModelForChat(m.filename)}
                onDelete={requestDeleteModel}
                onMakeDefault={makeDefaultModel}
              />
            ))}
          </>
        ) : (
          <p className="lane-empty">Nothing experimental on disk — every local model matches a supported row.</p>
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
        apiBase={catalogApiBase || apiBase}
        capabilities={capabilities}
        localFilenames={spine.localFilenames}
        downloads={spine.downloads}
        installAvailable={runtimeOnline && catalogInstallAvailable}
        installBlockedReason={
          !runtimeOnline
            ? 'The runtime is offline — start the Camelid backend to download models.'
            : 'The backend does not advertise a catalog-install capability, so downloads stay disabled.'
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
            <button
              type="button"
              className="lane-refresh"
              onClick={() => setInspectorOpen(true)}
              title="GGUF metadata, tokenizer, tensors — descriptive only, not support evidence"
            >
              Inspect loaded model metadata
            </button>
          </div>

          <div className="models-diagnostics__import">
            <h3>Import a GGUF by path</h3>
            <p className="lane-empty">
              Models inside the <code>models/</code> folder appear above automatically. Use this only
              for a GGUF stored elsewhere; it loads immediately and support still comes from
              /api/capabilities, not filename optimism.
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
              <button
                type="button"
                className="lane-row-action"
                onClick={importFromPath}
                disabled={importing || Boolean(loadingModelId)}
              >
                {importing || loadingModelId ? 'Loading…' : 'Import and load'}
              </button>
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
        title="Delete model from disk?"
        detail={pendingDeleteEntry
          ? `${pendingDeleteEntry.filename} (${prettySize(pendingDeleteEntry.size_bytes)}; ${pendingDeleteEntry.size_bytes.toLocaleString()} bytes) will be permanently removed. This cannot be undone.`
          : ''}
        confirmLabel="Delete from disk"
        busy={Boolean(deletingFilename)}
        onCancel={() => { if (!deletingFilename) setPendingDeleteEntry(null) }}
        onConfirm={deleteModelFromDisk}
      />
    </section>
  )
}
