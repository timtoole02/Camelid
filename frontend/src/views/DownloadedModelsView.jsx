import { useMemo, useRef, useState } from 'react'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import { Notice } from '../components/ui/Notice'
import {
  IconDownload,
  IconModels,
  IconPlay,
  IconRefresh,
  IconSearch,
  IconStop,
  IconTrash,
} from '../components/ui/icons'
import { useModelsPageData } from '../hooks/useModelsPageData'
import { formatBytes } from '../lib/formatters'
import {
  loadLocalModelForChat,
  modelFilenameFromPath,
  unloadLocalModel,
} from '../lib/modelActivation'
import { modelDeleteBlockedReason } from '../lib/modelDeletion'
import { isEmbeddingOnlyModel, isGenerationCapableModel } from '../lib/modelCapabilities.js'
import {
  describeModel,
  metaLine,
} from '../components/models/LaneRows'

function desktopInvoke() {
  if (typeof window === 'undefined') return null
  return window.__TAURI__?.core?.invoke || null
}

export default function DownloadedModelsView({
  runtime,
  apiBase = '',
  refreshDashboard,
  onOpenModels,
}) {
  const runtimeApiBase = (runtime?.api_base || '').replace(/\/$/, '')
  const spine = useModelsPageData({ apiBase: runtimeApiBase || apiBase })
  const [query, setQuery] = useState('')
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deletingFilename, setDeletingFilename] = useState('')
  const [defaultingFilename, setDefaultingFilename] = useState('')
  const [modelAction, setModelAction] = useState({ filename: '', type: '' })
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const [storageBusy, setStorageBusy] = useState(false)
  // { path: string | null, restartRequired: boolean } — null path means the
  // desktop default folder. Kept out of the transient notice state so the
  // pending-restart banner survives clearMessages().
  const [pendingStorage, setPendingStorage] = useState(null)
  const modelActionInFlightRef = useRef('')

  const models = spine.local?.models || []
  const totalBytes = models.reduce((sum, model) => sum + Number(model.size_bytes || 0), 0)
  const normalizedQuery = query.trim().toLowerCase()
  const filteredModels = useMemo(
    () => models.filter((model) => {
      if (!normalizedQuery) return true
      return [
        model.filename,
        model.architecture,
        model.quantization,
        model.tokenizer_kind,
      ].some((value) => String(value || '').toLowerCase().includes(normalizedQuery))
    }),
    [models, normalizedQuery],
  )
  const hasLoadedModels = Boolean(spine.activeFilename) || spine.loadedModelIds.size > 0
  const loadedModelCount = Math.max(spine.loadedModelIds.size, spine.activeFilename ? 1 : 0)
  const deleteBlockedReason = modelDeleteBlockedReason({
    residentModelsLoaded: hasLoadedModels,
    downloads: spine.downloads,
    loading: Boolean(modelAction.filename),
  })
  const invoke = desktopInvoke()

  const clearMessages = () => {
    setNotice('')
    setError('')
  }

  const makeDefault = async (filename) => {
    clearMessages()
    setDefaultingFilename(filename)
    try {
      await spine.setDefaultModel(filename)
      setNotice(`${filename} will load automatically when Camelid starts.`)
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setDefaultingFilename('')
    }
  }

  const load = async (entry) => {
    const filename = entry.filename
    if (modelActionInFlightRef.current) return
    modelActionInFlightRef.current = filename
    clearMessages()
    setModelAction({ filename, type: 'load' })
    try {
      const result = await loadLocalModelForChat({
        apiBase: spine.base,
        filename,
        model: entry,
        readActiveFilename: async () => modelFilenameFromPath((await spine.refreshCurrent())?.path),
      })
      if (!result.ok) throw new Error(result.message)
      await Promise.all([spine.refreshLoadedModels(), refreshDashboard?.({ silent: true })])
      setNotice(result.embedding
        ? `${filename} is loaded as an embedding model.`
        : `${filename} is loaded and ready.`)
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      modelActionInFlightRef.current = ''
      setModelAction({ filename: '', type: '' })
    }
  }

  const unload = async (filename, modelId = filename) => {
    if (!filename || modelActionInFlightRef.current) return
    modelActionInFlightRef.current = filename
    clearMessages()
    setModelAction({ filename, type: 'unload' })
    try {
      const result = await unloadLocalModel({ apiBase: spine.base, modelId })
      if (!result.ok) throw new Error(result.message)
      await Promise.all([
        spine.refreshCurrent(),
        spine.refreshLoadedModels(),
        refreshDashboard?.({ silent: true }),
      ])
      setNotice(`${filename} was unloaded. Its downloaded file is still available.`)
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      modelActionInFlightRef.current = ''
      setModelAction({ filename: '', type: '' })
    }
  }

  const confirmDelete = async () => {
    if (!pendingDelete || deletingFilename) return
    const entry = pendingDelete
    clearMessages()
    setDeletingFilename(entry.filename)
    try {
      const result = await spine.deleteLocalModel(entry)
      setPendingDelete(null)
      setNotice(result.bytes_freed
        ? `Deleted ${entry.filename} and freed ${formatBytes(result.bytes_freed)}.`
        : `Deleted ${entry.filename}.`)
    } catch (err) {
      setPendingDelete(null)
      setError(String(err?.message || err))
    } finally {
      setDeletingFilename('')
    }
  }

  const chooseStorage = async () => {
    if (!invoke) return
    clearMessages()
    setStorageBusy(true)
    try {
      const choice = await invoke('choose_models_directory')
      if (choice?.path) {
        setPendingStorage({ path: choice.path, restartRequired: Boolean(choice.restart_required) })
      }
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setStorageBusy(false)
    }
  }

  const resetStorage = async () => {
    if (!invoke) return
    clearMessages()
    setStorageBusy(true)
    try {
      // The command may return no path (the desktop resolves its default at
      // startup); render that as plain language, never as a fake path string.
      const choice = await invoke('reset_models_directory')
      setPendingStorage({ path: choice?.path || null, restartRequired: Boolean(choice?.restart_required) })
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setStorageBusy(false)
    }
  }

  return (
    <section className="downloaded-models-view cxv">
      <header className="cxv-head">
        <div className="cxv-head__copy">
          <p className="cxv-kicker"><IconDownload size={14} /> Local library</p>
          <h1>Downloaded models</h1>
          <p className="cxv-sub">
            See every GGUF in Camelid&apos;s active model folder, choose the startup default,
            and permanently remove files you no longer need.
          </p>
        </div>
        <div className="cxv-stats">
          <div className="cxv-stat">
            <span>Models</span>
            <strong>{models.length}</strong>
            <small>on disk</small>
          </div>
          <div className="cxv-stat">
            <span>Storage</span>
            <strong>{formatBytes(totalBytes)}</strong>
            <small>GGUF files</small>
          </div>
        </div>
      </header>

      <article className="cxv-card cxv-card--flat downloaded-storage">
        <div className="cxv-card__head">
          <div className="cxv-card__titles">
            <strong>Model storage folder</strong>
            <span className="cxv-card__sub-plain">
              Downloads and local scans use this folder.
            </span>
          </div>
          <span className="cxv-tag cxv-tag--accent">Active</span>
        </div>
        <code className="downloaded-storage__path">
          {spine.local?.models_dir || (spine.localLoading ? 'Loading…' : 'Unavailable')}
        </code>
        {pendingStorage ? (
          <div className="downloaded-storage__next" role="status">
            <span>After restart</span>
            {pendingStorage.path ? (
              <code>{pendingStorage.path}</code>
            ) : (
              <span>The default model folder</span>
            )}
          </div>
        ) : null}
        {pendingStorage?.restartRequired ? (
          <p className="downloaded-storage__note" role="status">
            Quit and reopen Camelid Desktop to start using this folder.
          </p>
        ) : null}
        <p className="downloaded-storage__note">
          Changing folders takes effect after restart. Existing model files stay in the current
          folder; Camelid does not move or copy them automatically.
        </p>
        <div className="downloaded-storage__actions">
          <Button variant="outline" size="sm" onClick={chooseStorage} disabled={!invoke || storageBusy}>
            {storageBusy ? 'Choosing…' : 'Choose folder…'}
          </Button>
          <Button variant="ghost" size="sm" onClick={resetStorage} disabled={!invoke || storageBusy}>
            Use default folder
          </Button>
          {!invoke ? (
            <span className="downloaded-storage__desktop-only">
              Folder selection is available in Camelid Desktop.
            </span>
          ) : null}
        </div>
      </article>

      {hasLoadedModels ? (
        <div className="downloaded-active" id="model-delete-guard">
          <div>
            <strong>{loadedModelCount} {loadedModelCount === 1 ? 'model is' : 'models are'} loaded.</strong>
            <span> Unload every loaded model before deleting any model file.</span>
          </div>
          {spine.activeFilename ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => unload(spine.activeFilename, spine.current?.id)}
              loading={modelAction.filename === spine.activeFilename && modelAction.type === 'unload'}
              disabled={Boolean(modelAction.filename)}
            >
              Unload current model
            </Button>
          ) : null}
        </div>
      ) : null}

      <Notice notice={error} tone="error" onDismiss={() => setError('')} />
      <Notice notice={notice} tone="success" onDismiss={() => setNotice('')} />
      {spine.localError && !spine.local ? (
        <Notice notice={`Could not list downloaded models: ${spine.localError}`} tone="error" />
      ) : null}

      <div className="cxv-toolbar downloaded-toolbar">
        <label className="cxv-search">
          <IconSearch size={17} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search downloaded models"
            aria-label="Search downloaded models"
          />
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
          variant="tonal"
          size="sm"
          icon={<IconModels size={16} />}
          onClick={onOpenModels}
        >
          Get more models
        </Button>
      </div>

      {/* The Load button is disabled while loading, which blurs it, so its name
          change is never announced. This carries the state instead. */}
      <span className="sr-only" role="status" aria-live="polite">
        {modelAction.type === 'load' && modelAction.filename ? `Loading ${modelAction.filename}…` : ''}
      </span>

      {!spine.local && spine.localLoading ? (
        <p className="lane-empty">Scanning the model folder…</p>
      ) : filteredModels.length ? (
        <div className="cxv-grid downloaded-grid">
          {filteredModels.map((entry) => {
            const isActive = entry.filename === spine.activeFilename
            const isLoaded = isActive || spine.loadedModelIds.has(entry.filename)
            const loadedModelId = isActive ? spine.current?.id : entry.filename
            const isDefault = entry.filename === spine.defaultFilename
            const deleteAvailable = Boolean(entry.delete_token)
            const canLoad = entry.lane_class !== 'unsupported'
            const embeddingOnly = isEmbeddingOnlyModel(entry, runtime)
            const generationCapable = isGenerationCapableModel(entry, runtime)
            const standaloneRuntime = embeddingOnly || generationCapable
            const actionBusy = Boolean(modelAction.filename)
            const isModelLoading = modelAction.filename === entry.filename && modelAction.type === 'load'
            return (
              <article
                className={`cxv-card downloaded-model${isActive ? ' downloaded-model--active' : ''}${isModelLoading ? ' downloaded-model--loading' : ''}`}
                key={entry.filename}
              >
                <div className="cxv-card__head">
                  <div className="cxv-card__titles">
                    <strong title={entry.filename}>{entry.filename}</strong>
                    <span className="cxv-card__sub">{metaLine(entry) || (entry.size_bytes ? formatBytes(entry.size_bytes) : '')}</span>
                  </div>
                  <div className="downloaded-model__tags">
                    {isLoaded ? <span className="cxv-tag cxv-tag--ready">Loaded</span> : null}
                    {isDefault ? <span className="cxv-tag cxv-tag--accent">Default</span> : null}
                  </div>
                </div>
                <p className="cxv-card__preview">{describeModel(entry)}</p>
                <div className="cxv-card__foot">
                  <div className="cxv-card__meta">
                    <strong>{entry.size_bytes ? formatBytes(entry.size_bytes) : 'Unknown size'}</strong>
                    <span className="cxv-dot">·</span>
                    <span>{embeddingOnly ? 'Embedding model' : generationCapable ? (entry.chat_capable ? 'Chat model' : 'Text model') : 'Companion asset'}</span>
                  </div>
                  <div className="cxv-card__actions">
                    {!isDefault && canLoad && generationCapable ? (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => makeDefault(entry.filename)}
                        loading={defaultingFilename === entry.filename}
                        disabled={Boolean(deletingFilename) || actionBusy}
                      >
                        Make default
                      </Button>
                    ) : null}
                    {/* Load and Unload are one mutually exclusive control: only
                        one of them is ever actionable, and a row of permanently
                        disabled Unload buttons made the grid look unfinished. */}
                    {isLoaded ? (
                      <Button
                        variant="outline"
                        size="sm"
                        icon={<IconStop size={15} />}
                        onClick={() => unload(entry.filename, loadedModelId)}
                        loading={modelAction.filename === entry.filename && modelAction.type === 'unload'}
                        disabled={actionBusy || Boolean(deletingFilename)}
                        aria-label={`Unload ${entry.filename}`}
                        title="Unload this model from memory"
                      >
                        Unload
                      </Button>
                    ) : standaloneRuntime ? (
                      <Button
                        variant="tonal"
                        size="sm"
                        icon={<IconPlay size={16} />}
                        onClick={() => load(entry)}
                        loading={isModelLoading}
                        loadingVariant="bar"
                        loadingLabel="Loading model…"
                        disabled={!canLoad || actionBusy || Boolean(deletingFilename)}
                        aria-label={`Load ${entry.filename}`}
                        title={canLoad ? 'Load this model into Camelid' : 'Camelid can’t run this model type'}
                      >
                        Load
                      </Button>
                    ) : null}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="cxv-danger"
                      icon={<IconTrash size={17} />}
                      onClick={() => {
                        clearMessages()
                        setPendingDelete(entry)
                      }}
                      disabled={!deleteAvailable || Boolean(deleteBlockedReason) || Boolean(deletingFilename) || actionBusy}
                      aria-label={`Delete ${entry.filename} from disk`}
                      aria-describedby={deleteBlockedReason ? 'model-delete-guard' : undefined}
                      title={
                        deleteBlockedReason
                        || (deleteAvailable ? 'Delete from disk' : 'Deletion is unavailable from this client')
                      }
                    >
                      Delete
                    </Button>
                  </div>
                </div>
              </article>
            )
          })}
        </div>
      ) : models.length ? (
        <p className="lane-empty">No downloaded models match “{query}”.</p>
      ) : (
        <div className="downloaded-empty">
          <IconDownload size={28} />
          <strong>No downloaded models</strong>
          <span>Open Models to choose and download a local GGUF.</span>
          <Button variant="tonal" size="sm" onClick={onOpenModels}>Browse models</Button>
        </div>
      )}

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title={pendingDelete ? `Delete ${pendingDelete.filename}?` : 'Delete model?'}
        detail={pendingDelete
          ? `This permanently removes ${pendingDelete.size_bytes ? formatBytes(pendingDelete.size_bytes) : 'this file'}${pendingDelete.ghost_moe_prepared ? ' and its Ghost MoE expert pack' : ''} from disk. This cannot be undone.`
          : ''}
        confirmLabel="Delete model"
        busy={Boolean(deletingFilename)}
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
      />
    </section>
  )
}
