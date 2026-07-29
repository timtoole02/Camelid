import { useMemo, useState } from 'react'
import { Button } from '../components/ui/Button'
import { ConfirmDialog } from '../components/ui/ConfirmDialog'
import {
  IconDownload,
  IconModels,
  IconRefresh,
  IconSearch,
  IconTrash,
} from '../components/ui/icons'
import { useModelsPageData } from '../hooks/useModelsPageData'
import { modelDeleteBlockedReason } from '../lib/modelDeletion'
import {
  describeModel,
  metaLine,
  prettySize,
} from '../components/models/LaneRows'

function desktopInvoke() {
  if (typeof window === 'undefined') return null
  return window.__TAURI__?.core?.invoke || null
}

export default function DownloadedModelsView({
  runtime,
  apiBase = '',
  unloadCurrentModel,
  onOpenModels,
}) {
  const runtimeApiBase = (runtime?.api_base || '').replace(/\/$/, '')
  const spine = useModelsPageData({ apiBase: runtimeApiBase || apiBase })
  const [query, setQuery] = useState('')
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deletingFilename, setDeletingFilename] = useState('')
  const [defaultingFilename, setDefaultingFilename] = useState('')
  const [unloading, setUnloading] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const [storageBusy, setStorageBusy] = useState(false)
  const [nextStoragePath, setNextStoragePath] = useState('')

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
  const deleteBlockedReason = modelDeleteBlockedReason({
    activeFilename: spine.activeFilename,
    downloads: spine.downloads,
    loading: unloading,
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

  const unload = async () => {
    clearMessages()
    setUnloading(true)
    try {
      await unloadCurrentModel()
      await spine.refreshCurrent()
      setNotice('The active model was unloaded. Downloaded files can now be deleted.')
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setUnloading(false)
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
      setNotice(`Deleted ${entry.filename} and freed ${prettySize(result.bytes_freed)}.`)
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
        setNextStoragePath(choice.path)
        setNotice('The new model folder will be used after Camelid Desktop restarts.')
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
      const choice = await invoke('reset_models_directory')
      setNextStoragePath(choice?.path || 'Camelid Desktop default')
      setNotice('The default model folder will be restored after Camelid Desktop restarts.')
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
            <strong>{prettySize(totalBytes) || '0 MB'}</strong>
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
        {nextStoragePath ? (
          <div className="downloaded-storage__next" role="status">
            <span>After restart</span>
            <code>{nextStoragePath}</code>
          </div>
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

      {spine.activeFilename ? (
        <div className="downloaded-active" id="model-delete-guard">
          <div>
            <strong>{spine.activeFilename}</strong>
            <span> is loaded. Unload it before deleting any model file.</span>
          </div>
          <Button variant="outline" size="sm" onClick={unload} loading={unloading}>
            Unload current model
          </Button>
        </div>
      ) : null}

      {error ? <p className="lane-error" role="alert">{error}</p> : null}
      {notice ? <p className="lane-delete-success" role="status">{notice}</p> : null}
      {spine.localError && !spine.local ? (
        <p className="lane-error">Could not list downloaded models: {spine.localError}</p>
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

      {!spine.local && spine.localLoading ? (
        <p className="lane-empty">Scanning the model folder…</p>
      ) : filteredModels.length ? (
        <div className="cxv-grid downloaded-grid">
          {filteredModels.map((entry) => {
            const isActive = entry.filename === spine.activeFilename
            const isDefault = entry.filename === spine.defaultFilename
            const deleteAvailable = Boolean(entry.delete_token)
            const canStartAutomatically = entry.lane_class !== 'unsupported'
            return (
              <article
                className={`cxv-card downloaded-model${isActive ? ' downloaded-model--active' : ''}`}
                key={entry.filename}
              >
                <div className="cxv-card__head">
                  <div className="cxv-card__titles">
                    <strong title={entry.filename}>{entry.filename}</strong>
                    <span className="cxv-card__sub">{metaLine(entry) || prettySize(entry.size_bytes)}</span>
                  </div>
                  <div className="downloaded-model__tags">
                    {isActive ? <span className="cxv-tag cxv-tag--ready">Loaded</span> : null}
                    {isDefault ? <span className="cxv-tag cxv-tag--accent">Default</span> : null}
                  </div>
                </div>
                <p className="cxv-card__preview">{describeModel(entry)}</p>
                <div className="cxv-card__foot">
                  <div className="cxv-card__meta">
                    <strong>{prettySize(entry.size_bytes) || 'Unknown size'}</strong>
                    <span className="cxv-dot">·</span>
                    <span>{entry.chat_capable ? 'Chat model' : 'Text model'}</span>
                  </div>
                  <div className="cxv-card__actions">
                    {!isDefault && canStartAutomatically ? (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => makeDefault(entry.filename)}
                        loading={defaultingFilename === entry.filename}
                        disabled={Boolean(deletingFilename)}
                      >
                        Make default
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
                      disabled={!deleteAvailable || Boolean(deleteBlockedReason) || Boolean(deletingFilename)}
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
          ? `This permanently removes ${prettySize(pendingDelete.size_bytes) || 'this file'} from disk. This cannot be undone.`
          : ''}
        confirmLabel="Delete model"
        busy={Boolean(deletingFilename)}
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
      />
    </section>
  )
}
