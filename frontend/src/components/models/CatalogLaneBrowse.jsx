import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { isCompatibilitySupportedForModel } from '../../lib/capabilities'
import { beginCatalogSettlement, catalogDownloadSettlement, completeCatalogAcquisition, reserveCatalogAcquisition } from '../../lib/catalogActivation'
import {
  defaultFileIndex,
  fitDetail,
  fitIsRecheckable,
  fitIsSettled,
  fitLabel,
  groupHfFilesByRepo,
  isRefusingFit,
  partitionByArchSupport,
  partitionCuratedByFit,
  quantAdvice,
} from '../../lib/catalogBrowse'
import { SUPPORTED_MODELS } from '../../lib/supportedModels'
import { EvidenceChip } from '../ui/EvidenceChip'

/* Zone 5 — Get models. Curated picks first, then live Hugging Face GGUF search
   (>= 2 chars). Each row shows which lane it WOULD land in (derived: supported
   contract match, oracle-qualified runnable, or not-yet-anchored). Download is
   user-initiated and explicitly confirmed (filename + HF repo + size); no
   background/auto pulls. Live progress renders in the global Downloads zone —
   rows here only reflect their own acquisition state, read from the shared
   downloads poll + the live /api/models/local scan (never localStorage). After a
   download lands, smoke-admission runs for oracle-qualified combos and the model
   appears in its derived local section.

   Hugging Face results are grouped into one card per repo with a quantization
   picker. The Hub returns files, and a single repo routinely ships 20-27 quants of
   the same weights, so rendering them flat turned a 15-repo search into a 250-row
   wall. Grouping is presentation only: a download is still one exact file,
   confirmed by name. */

const GB = 1024 * 1024 * 1024
function prettySize(bytes) {
  if (!bytes) return ''
  if (bytes >= GB) return `${(bytes / GB).toFixed(bytes >= 10 * GB ? 0 : 1)} GB`
  return `${Math.round(bytes / (1024 * 1024))} MB`
}

/* Curated download suggestions (blurbs, "Recommended") may DECORATE catalog rows,
   never place them: lane membership and outcome chips stay derived. */
const CURATED_DECORATION = new Map(SUPPORTED_MODELS.map((item) => [item.catalog_id, item]))

/* Why a live Hugging Face row can never imply support — said once per section and
   in a tooltip on the guessed value, rather than as a paragraph on all 150 rows. */
const HF_GUESS_EXPLANATION =
  'Architecture and quantization are read from the filename, not the model. The real lane is only known after the file loads.'

/* Predicted lane for a catalog entry — derived, never a hand-authored label. */
function predictedLane(item, capabilities) {
  // Experimental (live Hugging Face) rows are advisory only: their architecture/quant
  // are filename guesses, so they can never anchor a lane or imply support — even when
  // the filename happens to coincide with a supported contract row. Always not-anchored.
  if (item.group === 'experimental') return 'not_anchored'
  if (isCompatibilitySupportedForModel(capabilities, null, item)) return 'supported'
  if (item.oracle_qualified) return 'compatible'
  return 'not_anchored'
}

function laneChip(lane) {
  if (lane === 'supported') return <EvidenceChip status="supported" asText>Lands in Supported</EvidenceChip>
  if (lane === 'compatible') return <EvidenceChip state="runnable" asText>Experimental · runnable</EvidenceChip>
  return <EvidenceChip state="unsupported" asText>Experimental · unverified</EvidenceChip>
}

/* A small CPU/chip glyph so the capacity chip reads as "your hardware" — distinct
   from the support/lane chips. A check (fits) or cross (refused) sits in the die. */
function FitIcon({ bad }) {
  const stroke = 'currentColor'
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="4.5" y="4.5" width="7" height="7" rx="1.2" stroke={stroke} strokeWidth="1.3" />
      {bad ? (
        <path d="M6.4 6.4l3.2 3.2M9.6 6.4l-3.2 3.2" stroke={stroke} strokeWidth="1.3" strokeLinecap="round" />
      ) : (
        <path d="M6 8.2l1.4 1.4L10 6.6" stroke={stroke} strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
      )}
      <path
        d="M6.5 2.6v1.9M9.5 2.6v1.9M6.5 11.5v1.9M9.5 11.5v1.9M2.6 6.5h1.9M2.6 9.5h1.9M11.5 6.5h1.9M11.5 9.5h1.9"
        stroke={stroke}
        strokeWidth="1.1"
        strokeLinecap="round"
      />
    </svg>
  )
}

/* Capacity advisory for THIS host (fit axis, NOT a support claim — kept on its own
   line, never merged into the lane/support chip).

   Four shapes, because they mean four different things: a positive fit; a refusal
   that names WHY (too big for the machine vs. the machine is merely busy); a row
   nobody has measured yet, which gets an explicit "check this one" affordance
   instead of silence; and a row whose question is settled with no answer possible.

   The last two both arrive as `fit: 'unknown'`. Rendering them the same put the
   button in a loop — press, get `unknown` back, see the same button — which is why
   the backend reports whether the check settled. */
function FitAdvisory({ item, onCheckFit, checking }) {
  const label = fitLabel(item.fit)
  const detail = fitDetail(item.fit)
  const refused = isRefusingFit(item.fit)
  const transient = item.fit === 'insufficient_free_memory'

  if (!label) {
    if (fitIsSettled(item)) {
      return (
        <div className="catalog-fit-row">
          <span
            className="catalog-fit-chip catalog-fit-chip--unknown"
            title="Camelid looked at this model and will not promise a verdict here — either its dimensions cannot be read, this machine's memory cannot be probed, or the GPU has room while free system memory is too low to stage the weights."
          >
            <FitIcon />
            Fit can’t be determined here
          </span>
          <span className="catalog-fit-detail">
            The download is not blocked — Camelid just will not claim a fit it cannot verify.
          </span>
        </div>
      )
    }
    if (!onCheckFit) return null
    return (
      <div className="catalog-fit-row">
        <span className="catalog-fit-chip catalog-fit-chip--unknown">
          <FitIcon />
          Fit unknown until checked
        </span>
        <button
          type="button"
          className="catalog-fit-check"
          onClick={onCheckFit}
          disabled={checking}
          aria-busy={checking || undefined}
        >
          {checking ? 'Checking…' : item.fit_checked === false ? 'Try again' : 'Check if it fits'}
        </button>
        {item.fit_checked === false ? (
          <span className="catalog-fit-detail">
            Could not reach Hugging Face for this model’s header. Retrying may work.
          </span>
        ) : null}
      </div>
    )
  }

  return (
    <div className="catalog-fit-row">
      <span
        className={`catalog-fit-chip catalog-fit-chip--${transient ? 'warn' : refused ? 'bad' : 'good'}${
          item.fit_confidence === 'approx' ? ' catalog-fit-chip--estimate' : ''
        }`}
        title={
          item.fit_confidence === 'exact'
            ? "Sized from the model's real dimensions (KV cache computed exactly)"
            : 'Estimate — upgrades to exact once the model header has been read'
        }
      >
        <FitIcon bad={refused} />
        {item.fit_confidence === 'approx' ? '~ ' : ''}
        {label}
      </span>
      {/* Memory pressure is the one refusal a user can act on, and acting on it
          needs a live re-probe: the listing's verdicts come from a startup
          snapshot, so reloading the page cannot pick up freed memory. */}
      {transient && onCheckFit ? (
        <button
          type="button"
          className="catalog-fit-check"
          onClick={onCheckFit}
          disabled={checking}
          aria-busy={checking || undefined}
          title="Re-read this machine's free memory and this model's size, right now"
        >
          {checking ? 'Re-checking…' : 'Re-check'}
        </button>
      ) : null}
      {Array.isArray(item.task_tags) && item.task_tags.length ? (
        <span className="catalog-fit-tags">
          <span className="catalog-fit-tags-label">best for</span>
          {item.task_tags.map((tag) => (
            <span key={tag} className="catalog-fit-tag">
              {tag}
            </span>
          ))}
        </span>
      ) : null}
      {detail ? <span className="catalog-fit-detail">{detail}</span> : null}
    </div>
  )
}

function CatalogRow({
  item,
  capabilities,
  installed,
  activeDownload,
  apiBase,
  installAvailable,
  installBlockedReason,
  onInstallStarted,
  onDownloadAcknowledged,
  onAcquired,
  canceled,
  onDownloadRetry,
  acquisitionLocked,
  onAcquisitionPending,
  onAcquisitionSettled,
  onStartModel,
  onModelStarted,
  onOperationBusy,
  onCheckFit,
  checkingFit,
  /* Rendered inside a Hugging Face model card, which already owns the title,
     provenance and quantization picker. Suppresses the row's own head so the two
     do not repeat each other. */
  compact = false,
}) {
  // phase: idle | confirm | starting | waiting | checking | loading | failed | done
  const [phase, setPhase] = useState('idle')
  const [message, setMessage] = useState('')
  const [isError, setIsError] = useState(false)
  const [failedStage, setFailedStage] = useState('')
  const [settlementTick, setSettlementTick] = useState(0)
  const sawDownloadRef = useRef(false)
  const startedAtRef = useRef(0)
  const settledAtRef = useRef(0)
  const acquisitionModeRef = useRef('download')
  const acquisitionItemRef = useRef(item)
  const settlementInFlightRef = useRef(false)
  const lane = predictedLane(item, capabilities)
  const decoration = item.group === 'experimental' ? null : CURATED_DECORATION.get(item.catalog_id)
  // ANY load-refusing verdict must stop the auto-start chain, not just `wont_fit`:
  // the load-time guard refuses both, so chaining into it would end in a 422.
  const refusedByFit = isRefusingFit(item.fit)
  const downloadAndStart = lane === 'supported' && !refusedByFit
  const smokeAfterDownload = item.group !== 'experimental'
    && !refusedByFit
    && !downloadAndStart
    && item.oracle_qualified
  const acquisitionMode = downloadAndStart ? 'start' : smokeAfterDownload ? 'smoke' : 'download'
  const downloading = activeDownload?.status === 'downloading'
  const rejoinableDownload = downloading || activeDownload?.status === 'completed'
  const operationBusy = phase === 'checking' || phase === 'loading'

  useEffect(() => {
    onOperationBusy?.(item.catalog_id, operationBusy)
    return () => {
      if (operationBusy) onOperationBusy?.(item.catalog_id, false)
    }
  }, [item.catalog_id, onOperationBusy, operationBusy])

  useEffect(() => {
    if (phase !== 'idle' || !rejoinableDownload || acquisitionLocked) return
    if (onAcquisitionPending?.(item) === false) return
    sawDownloadRef.current = true
    startedAtRef.current = Date.now()
    settledAtRef.current = 0
    acquisitionModeRef.current = activeDownload?.continuation_mode || acquisitionMode
    acquisitionItemRef.current = item
    settlementInFlightRef.current = false
    setMessage('Rejoined the active download.')
    setIsError(false)
    setPhase('waiting')
  }, [acquisitionLocked, acquisitionMode, activeDownload?.continuation_mode, item, onAcquisitionPending, phase, rejoinableDownload])

  const finishLanded = useCallback(async () => {
    if (!beginCatalogSettlement(settlementInFlightRef)) return
    setIsError(false)
    setFailedStage('')
    onAcquired?.()
    const result = await completeCatalogAcquisition({
      item: acquisitionItemRef.current,
      mode: acquisitionModeRef.current,
      apiBase,
      loadModelForChat: onStartModel,
      onStage: setPhase,
    })
    setMessage(result.message)
    if (!result.ok) {
      setFailedStage(result.stage)
      setIsError(true)
      setPhase('failed')
      onAcquisitionSettled?.(item.catalog_id)
      return
    }
    await onAcquired?.()
    try {
      const ack = await fetch(`${apiBase}/api/models/catalog/ack`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: item.catalog_id }),
      })
      if (!ack.ok) throw new Error(`acknowledgement failed (HTTP ${ack.status})`)
    } catch (error) {
      setFailedStage(result.started ? 'loading' : 'checking')
      setIsError(true)
      setMessage(`The model is ready, but Camelid could not finalize the download state: ${String(error?.message || error)}`)
      setPhase('failed')
      onAcquisitionSettled?.(item.catalog_id)
      return
    }
    await onDownloadAcknowledged?.()
    setPhase('done')
    onAcquisitionSettled?.(item.catalog_id)
    if (result.started) onModelStarted?.()
  }, [apiBase, item, onAcquired, onAcquisitionSettled, onDownloadAcknowledged, onModelStarted, onStartModel])

  const retryAcquisition = () => {
    if (onAcquisitionPending?.(item) === false) {
      setMessage('Wait for the current model acquisition to finish, then retry.')
      return
    }
    settlementInFlightRef.current = false
    finishLanded()
  }

  // The row watches the SHARED downloads poll + local scan instead of polling
  // itself: downloading -> (gone + on disk) = landed; (gone + not on disk after
  // having been seen) = failed or canceled.
  useEffect(() => {
    if (phase !== 'waiting') return undefined
    let refreshing = false
    const refreshSettlement = async () => {
      if (refreshing) return
      refreshing = true
      await onAcquired?.()
      setSettlementTick((value) => value + 1)
      refreshing = false
    }
    refreshSettlement()
    const timer = setInterval(refreshSettlement, 1000)
    return () => clearInterval(timer)
  }, [phase, onAcquired])

  useEffect(() => {
    if (phase !== 'waiting') return
    if (canceled) {
      settlementInFlightRef.current = false
      setPhase('idle')
      setIsError(true)
      setMessage('Download canceled. It can be retried.')
      onAcquisitionSettled?.(item.catalog_id)
      return
    }
    const settlement = catalogDownloadSettlement({
      downloading,
      installed,
      sawDownload: sawDownloadRef.current,
      settledAt: settledAtRef.current,
      startedAt: startedAtRef.current,
    })
    sawDownloadRef.current = settlement.sawDownload
    settledAtRef.current = settlement.settledAt
    if (settlement.action === 'landed') {
      finishLanded()
      return
    }
    if (settlement.action === 'failed') {
      setPhase('idle')
      setIsError(true)
      setMessage('Download did not complete (canceled or failed). It can be retried.')
      onAcquisitionSettled?.(item.catalog_id)
    }
  }, [phase, downloading, installed, canceled, settlementTick, finishLanded, item.catalog_id, onAcquisitionSettled])

  const confirmDownload = async () => {
    setPhase('starting')
    setMessage('')
    setIsError(false)
    sawDownloadRef.current = false
    startedAtRef.current = Date.now()
    settledAtRef.current = 0
    acquisitionModeRef.current = acquisitionMode
    acquisitionItemRef.current = item
    settlementInFlightRef.current = false
    onDownloadRetry?.(item.catalog_id)
    try {
      const res = await fetch(`${apiBase}/api/models/catalog/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          catalog_id: item.catalog_id,
          repo_id: item.repo_id,
          filename: item.filename,
          size_bytes: item.size_bytes,
          continuation_mode: acquisitionMode,
        }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        if (res.status !== 409 || body?.error?.code !== 'download_already_running') {
          throw new Error(body?.error?.message || `download failed (HTTP ${res.status})`)
        }
      }
      setPhase('waiting')
      onInstallStarted?.()
    } catch (err) {
      setPhase('idle')
      setIsError(true)
      setMessage(String(err?.message || err))
      onAcquisitionSettled?.(item.catalog_id)
    }
  }

  const openConfirmation = () => {
    if (onAcquisitionPending?.(item) === false) return
    setPhase('confirm')
  }

  const activeStage = phase === 'checking' ? 1 : phase === 'loading' ? 2 : 0
  const showProgress = ['starting', 'waiting', 'checking', 'loading'].includes(phase)

  return (
    <article
      className={`catalog-row${lane === 'not_anchored' ? ' catalog-row--advisory' : ''}${
        compact ? ' catalog-row--compact' : ''
      }`}
    >
      {compact ? null : (
        <div className="catalog-row-head">
          <div className="catalog-row-id">
            <span className="catalog-row-name">
              {item.name}
              {decoration?.recommended ? <span className="catalog-row-recommended">Recommended</span> : null}
            </span>
            <span className="catalog-row-meta">
              {item.repo_id} · {item.filename} · {prettySize(item.size_bytes)}
              {item.architecture ? ` · ${item.architecture}` : ''}
            </span>
          </div>
          {laneChip(lane)}
        </div>
      )}
      <FitAdvisory item={item} onCheckFit={onCheckFit} checking={checkingFit} />
      {decoration?.blurb ? <p className="catalog-row-blurb">{decoration.blurb}</p> : null}

      {showProgress ? (
        <div className="catalog-start" role="status" aria-live="polite">
          {downloadAndStart ? (
            <ol className="catalog-start-steps" aria-label="Download and start progress">
              {['Download', 'Check', 'Load'].map((label, index) => (
                <li key={label} className={index < activeStage ? 'is-done' : index === activeStage ? 'is-active' : ''}>
                  <span>{index < activeStage ? '✓' : index + 1}</span>
                  {label}
                </li>
              ))}
            </ol>
          ) : smokeAfterDownload ? (
            <ol className="catalog-start-steps catalog-start-steps--two" aria-label="Download and check progress">
              {['Download', 'Check'].map((label, index) => (
                <li key={label} className={index < activeStage ? 'is-done' : index === activeStage ? 'is-active' : ''}>
                  <span>{index < activeStage ? '✓' : index + 1}</span>
                  {label}
                </li>
              ))}
            </ol>
          ) : null}
          <p className="catalog-row-faint">
            {phase === 'checking'
              ? 'Download complete — checking the model…'
              : phase === 'loading'
                ? 'Check passed — loading the model for Chat…'
                : downloading
                  ? 'Downloading — live progress is shown above.'
                  : 'Starting download…'}
          </p>
        </div>
      ) : phase === 'failed' ? (
        <div className="catalog-start-failure" role="alert">
          <p className="catalog-row-error">{message}</p>
          <p className="catalog-row-faint">The file is still on disk. Camelid has not opened Chat.</p>
          <button type="button" className="catalog-row-action" onClick={retryAcquisition}>
            {failedStage === 'checking' ? 'Retry check' : 'Retry start'}
          </button>
        </div>
      ) : phase === 'done' ? (
        <p className={isError ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p>
      ) : installed ? (
        <p className="catalog-row-faint">Already on disk — shown in its section above.</p>
      ) : phase === 'idle' ? (
        <>
          {item.group !== 'experimental' && lane === 'not_anchored' ? (
            <p className="catalog-row-faint">
              Its {item.architecture}/{item.quant} combo is not yet in the runnable lane — still
              downloadable; it lands in Experimental and loads through the experimental chat path.
            </p>
          ) : null}
          {message ? <p className={isError ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p> : null}
          {installAvailable ? (
            <button
              type="button"
              className="catalog-row-action"
              onClick={openConfirmation}
              disabled={acquisitionLocked}
              title={acquisitionLocked ? 'Wait for the current model acquisition to finish' : undefined}
            >
              {downloadAndStart ? 'Download and start…' : 'Download…'}
            </button>
          ) : (
            <>
              <button type="button" className="catalog-row-action" disabled>
                Download unavailable
              </button>
              <p className="catalog-row-faint">{installBlockedReason}</p>
            </>
          )}
        </>
      ) : phase === 'confirm' ? (
        <div className="catalog-confirm">
          <p>
            Download <strong>{item.filename}</strong> from <code>{item.repo_id}</code> (
            {prettySize(item.size_bytes)})? This pulls from HuggingFace into your local models folder.
            {downloadAndStart ? ' Camelid will check it, load it, and open Chat after the download.' : ''}
          </p>
          <div className="catalog-confirm-actions">
            <button type="button" className="catalog-row-action" onClick={confirmDownload}>
              Confirm download
            </button>
            <button
              type="button"
              className="catalog-row-cancel"
              onClick={() => {
                setPhase('idle')
                onAcquisitionSettled?.(item.catalog_id)
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}
    </article>
  )
}

/* One live Hugging Face repo, with its quantizations behind a picker.

   A repo is one model; its `.gguf` files are size/quality variants of the same
   weights. Selecting a variant swaps which exact file the row below will download —
   the confirmation still names that file, so grouping never blurs what is fetched. */
function HfModelCard({ group, renderRow, onCheckFit, isCheckingFit }) {
  /* The selection is the chosen FILE, not its position. "Load more" can append
     quantizations to a repo already on screen, and the list is re-sorted by
     quality on every render — holding an index would silently slide the user's
     choice onto a different file (and a different download) underneath them.
     `null` means "never chosen", which is what lets the default track incoming
     fit verdicts instead of freezing at first paint. */
  const [selectedId, setSelectedId] = useState(null)
  const fallback = group.files[Math.max(0, defaultFileIndex(group.files))]
  const file = group.files.find((candidate) => candidate.catalog_id === selectedId) || fallback

  if (!file) return null

  return (
    <article className="hf-model-card">
      <div className="hf-model-head">
        <div className="hf-model-id">
          <h4 className="hf-model-title">{group.title}</h4>
          <p className="hf-model-meta">
            {group.owner ? <span className="hf-model-owner">{group.owner}</span> : null}
            {group.architecture ? (
              <span className="hf-model-arch" title={`Guessed from the filename. ${HF_GUESS_EXPLANATION}`}>
                {group.architecture} (guessed)
              </span>
            ) : null}
            <span>
              {group.files.length} quantization{group.files.length === 1 ? '' : 's'}
            </span>
            {group.archSupport === 'not_implemented' ? (
              <span className="hf-model-unsupported">Camelid does not implement this architecture</span>
            ) : null}
          </p>
        </div>
        <EvidenceChip state="unsupported" asText>Experimental · unverified</EvidenceChip>
      </div>

      <div className="hf-quant-picker">
        <label className="hf-quant-label" htmlFor={`quant-${group.repoId}`}>
          Quantization
        </label>
        <select
          id={`quant-${group.repoId}`}
          className="hf-quant-select"
          value={file.catalog_id}
          onChange={(event) => setSelectedId(event.target.value)}
        >
          {group.files.map((candidate) => (
            <option key={candidate.catalog_id} value={candidate.catalog_id}>
              {candidate.quant || 'unlabelled'} · {prettySize(candidate.size_bytes)}
              {quantAdvice(candidate.quant).note ? ` · ${quantAdvice(candidate.quant).note}` : ''}
            </option>
          ))}
        </select>
      </div>

      {renderRow(file, {
        compact: true,
        onCheckFit: () => onCheckFit(file),
        checkingFit: isCheckingFit(file),
      })}
    </article>
  )
}

/* Placeholder rows while a live Hugging Face search is in flight. A search costs
   real network round-trips; leaving the previous results on screen with no
   indicator is what made it read as a hang rather than as work in progress. */
function SearchSkeleton() {
  return (
    <div className="catalog-skeleton" role="status" aria-live="polite" aria-busy="true">
      <p className="lane-empty">Searching Hugging Face…</p>
      {[0, 1, 2].map((row) => (
        <div key={row} className="catalog-skeleton-row" aria-hidden="true">
          <span className="catalog-skeleton-bar catalog-skeleton-bar--title" />
          <span className="catalog-skeleton-bar catalog-skeleton-bar--meta" />
        </div>
      ))}
    </div>
  )
}

function CatalogGroup({ title, marker, count, emptyText, children }) {
  return (
    <section className="catalog-group">
      <div className="catalog-group-head">
        <h3>{title}</h3>
        {marker}
      </div>
      <div className="catalog-list">
        {count === 0 ? <p className="lane-empty">{emptyText}</p> : children}
      </div>
    </section>
  )
}

export function CatalogLaneBrowse({
  apiBase = '',
  capabilities,
  localFilenames = new Set(),
  downloads = [],
  installAvailable = true,
  installBlockedReason = '',
  onInstallStarted,
  onDownloadAcknowledged,
  onAcquired,
  canceledCatalogIds = new Set(),
  onDownloadRetry,
  onStartModel,
  onModelStarted,
  onOperationBusy,
}) {
  const base = (apiBase || '').replace(/\/$/, '')
  const [items, setItems] = useState(null)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [nextCursor, setNextCursor] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')
  const [pendingCatalogId, setPendingCatalogId] = useState('')
  const [pendingItem, setPendingItem] = useState(null)
  const [checkingFitIds, setCheckingFitIds] = useState(() => new Set())
  const pendingCatalogIdRef = useRef('')
  const requestSequenceRef = useRef(0)
  const inFlightRef = useRef(null)

  // Debounce the query so each keystroke doesn't fire a live Hugging Face search.
  // A search is one Hub round-trip plus one per repo, so it costs a noticeable
  // fraction of a second even warm; a shorter debounce only queues work that the
  // next keystroke throws away.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query.trim()), 500)
    return () => clearTimeout(t)
  }, [query])

  const load = useCallback(async () => {
    const sequence = ++requestSequenceRef.current
    // Abort the previous search instead of letting it run to completion and be
    // discarded: it holds a Hub connection and a server-side fetch worker.
    inFlightRef.current?.abort()
    const controller = new AbortController()
    inFlightRef.current = controller
    setError('')
    setLoading(true)
    try {
      const params = debouncedQuery ? `?query=${encodeURIComponent(debouncedQuery)}` : ''
      const res = await fetch(`${base}/api/models/catalog${params}`, { signal: controller.signal })
      if (!res.ok) throw new Error(`catalog HTTP ${res.status}`)
      const body = await res.json()
      if (sequence !== requestSequenceRef.current) return
      setItems(body.items || [])
      setNextCursor(body.next_cursor || null)
    } catch (err) {
      if (err?.name === 'AbortError') return
      if (sequence !== requestSequenceRef.current) return
      setError(String(err?.message || err))
    } finally {
      if (sequence === requestSequenceRef.current) setLoading(false)
    }
  }, [base, debouncedQuery])

  useEffect(() => {
    load()
  }, [load])

  // Unmounting mid-search must not leave the request (and its server-side work)
  // running.
  useEffect(() => () => inFlightRef.current?.abort(), [])

  const reserveAcquisition = useCallback((item) => {
    const catalogId = item.catalog_id
    const reservation = reserveCatalogAcquisition(pendingCatalogIdRef.current, catalogId)
    if (!reservation.accepted) return false
    pendingCatalogIdRef.current = reservation.catalogId
    setPendingCatalogId(catalogId)
    setPendingItem(item)
    return true
  }, [])

  const settleAcquisition = useCallback((catalogId) => {
    if (pendingCatalogIdRef.current !== catalogId) return
    pendingCatalogIdRef.current = ''
    setPendingCatalogId('')
    setPendingItem(null)
  }, [])

  // Append the next page of experimental (Hugging Face) results.
  const loadMore = useCallback(async () => {
    if (!nextCursor || !debouncedQuery) return
    const sequence = requestSequenceRef.current
    setLoadingMore(true)
    try {
      const params = `?query=${encodeURIComponent(debouncedQuery)}&cursor=${encodeURIComponent(nextCursor)}`
      const res = await fetch(`${base}/api/models/catalog${params}`)
      if (!res.ok) throw new Error(`catalog HTTP ${res.status}`)
      const body = await res.json()
      if (sequence !== requestSequenceRef.current) return
      const more = (body.items || []).filter((it) => it.group === 'experimental')
      setItems((prev) => {
        const seen = new Set((prev || []).map((it) => it.catalog_id))
        return [...(prev || []), ...more.filter((it) => !seen.has(it.catalog_id))]
      })
      setNextCursor(body.next_cursor || null)
    } catch (err) {
      if (sequence !== requestSequenceRef.current) return
      setError(String(err?.message || err))
    } finally {
      setLoadingMore(false)
    }
  }, [base, debouncedQuery, nextCursor])

  /* Resolve one model's real GGUF dimensions on demand and fold the verdict back
     into its row. Explicit and per-row on purpose: the background warm is capped
     precisely to avoid a header-fetch storm, so the fix for "most rows say unknown"
     is a user-driven check, not a bigger fan-out. */
  const checkFit = useCallback(async (item) => {
    setCheckingFitIds((prev) => new Set(prev).add(item.catalog_id))
    try {
      const res = await fetch(`${base}/api/models/catalog/fit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_id: item.repo_id,
          filename: item.filename,
          size_bytes: item.size_bytes,
        }),
      })
      if (!res.ok) throw new Error(`fit check HTTP ${res.status}`)
      const body = await res.json()
      setItems((prev) =>
        (prev || []).map((row) =>
          row.catalog_id === item.catalog_id
            ? {
                ...row,
                fit: body.fit,
                fit_confidence: body.fit_confidence,
                // Whether the question is settled. Without it an unresolvable model
                // keeps offering a check that can never change anything.
                fit_checked: body.checked,
              }
            : row,
        ),
      )
    } catch (err) {
      setError(String(err?.message || err))
    } finally {
      setCheckingFitIds((prev) => {
        const next = new Set(prev)
        next.delete(item.catalog_id)
        return next
      })
    }
  }, [base])

  const isCheckingFit = useCallback((item) => checkingFitIds.has(item.catalog_id), [checkingFitIds])

  const renderRow = useCallback((item, extra) => (
    <CatalogRow
      key={item.catalog_id}
      item={item}
      capabilities={capabilities}
      installed={localFilenames.has(item.filename)}
      activeDownload={downloads.find((download) => download.id === item.catalog_id)}
      apiBase={base}
      installAvailable={installAvailable}
      installBlockedReason={installBlockedReason}
      onInstallStarted={onInstallStarted}
      onDownloadAcknowledged={onDownloadAcknowledged}
      onAcquired={onAcquired}
      canceled={canceledCatalogIds.has(item.catalog_id)}
      onDownloadRetry={onDownloadRetry}
      acquisitionLocked={Boolean(pendingCatalogId && pendingCatalogId !== item.catalog_id)}
      onAcquisitionPending={reserveAcquisition}
      onAcquisitionSettled={settleAcquisition}
      onStartModel={onStartModel}
      onModelStarted={onModelStarted}
      onOperationBusy={onOperationBusy}
      // Curated rows get the re-check path too, and it matters more for them:
      // the landing view now FOLDS AWAY rows the catalog thinks cannot load, so a
      // stale "not enough memory" would hide a model that has since become
      // runnable, with no way to bring it back.
      onCheckFit={fitIsRecheckable(item.fit) ? () => checkFit(item) : undefined}
      checkingFit={isCheckingFit(item)}
      {...extra}
    />
  ), [
    base, canceledCatalogIds, capabilities, checkFit, downloads, installAvailable,
    installBlockedReason, isCheckingFit, localFilenames, onAcquired, onDownloadAcknowledged,
    onDownloadRetry, onInstallStarted, onModelStarted, onOperationBusy, onStartModel,
    pendingCatalogId, reserveAcquisition, settleAcquisition,
  ])

  // A row whose acquisition is in flight must keep rendering even if a newer
  // search no longer returns it, or the user loses sight of their own download.
  const visibleItems = useMemo(() => {
    const rows = items || []
    return pendingItem && !rows.some((item) => item.catalog_id === pendingItem.catalog_id)
      ? [pendingItem, ...rows]
      : rows
  }, [items, pendingItem])
  const curated = useMemo(
    () => visibleItems.filter((it) => it.group !== 'experimental'),
    [visibleItems],
  )
  const experimental = useMemo(
    () => visibleItems.filter((it) => it.group === 'experimental'),
    [visibleItems],
  )
  const searching = debouncedQuery.length >= 2

  const hfGroups = useMemo(() => groupHfFilesByRepo(experimental), [experimental])
  const { loadable, unimplemented } = useMemo(() => partitionByArchSupport(hfGroups), [hfGroups])
  // Landing state only: with no query, lead with what this machine can actually
  // run rather than a wall of rows the user cannot use.
  const { runnable: curatedRunnable, blocked: curatedBlocked } = useMemo(
    () => partitionCuratedByFit(curated),
    [curated],
  )

  const renderHfCard = (group) => (
    <HfModelCard
      key={group.repoId}
      group={group}
      renderRow={renderRow}
      onCheckFit={checkFit}
      isCheckingFit={isCheckingFit}
    />
  )

  return (
    <div className="catalog-lane-browse">
      <div className="local-lane-head">
        <h2>Get models</h2>
      </div>
      <p className="local-lane-intro">
        Curated picks are pinned and known-good. Searching also browses live Hugging Face GGUFs as an
        experimental group — those are unverified and carry no parity claim. Downloads are explicit
        and confirmed; progress appears in Downloads above, and the model joins its derived section
        when the file lands.
      </p>
      <input
        className="catalog-search"
        aria-label="Search model catalog"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={Boolean(pendingCatalogId)}
        placeholder="Search curated picks and live Hugging Face GGUFs (name, repo, filename)"
      />
      {error ? (
        <p className="lane-error">
          {items === null ? `Catalog unavailable: ${error}` : error}
        </p>
      ) : null}
      {items === null && !error ? <p className="lane-empty">Loading catalog…</p> : null}

      {items === null && !error ? null : searching ? (
        <CatalogGroup
          title="Curated"
          marker={null}
          count={curated.length}
          emptyText="No curated entries match."
        >
          {curated.map((item) => renderRow(item))}
        </CatalogGroup>
      ) : (
        <>
          <CatalogGroup
            title="Curated — runs on this machine"
            marker={null}
            count={curatedRunnable.length}
            emptyText="No curated model fits the memory free right now. Close some applications, then use Re-check on one of the rows below."
          >
            {curatedRunnable.map((item) => renderRow(item))}
          </CatalogGroup>
          {curatedBlocked.length ? (
            <details className="catalog-collapsed">
              <summary>
                {curatedBlocked.length} more curated model{curatedBlocked.length === 1 ? '' : 's'} this
                machine cannot load right now
              </summary>
              <div className="catalog-list">{curatedBlocked.map((item) => renderRow(item))}</div>
            </details>
          ) : null}
        </>
      )}

      {searching && items !== null ? (
        <>
          <CatalogGroup
            title="Experimental (Hugging Face)"
            marker={
              <span className="catalog-experimental-marker">
                <EvidenceChip state="unsupported" asText>Experimental — unverified, no parity claim</EvidenceChip>
              </span>
            }
            count={loading ? 1 : loadable.length}
            emptyText={
              // Saying "nothing matched" when results DID match and were merely
              // folded into the section below would be false, and it hides the one
              // place the user should look next.
              unimplemented.length
                ? `Every match (${unimplemented.length}) uses an architecture Camelid does not implement — see below.`
                : 'No live Hugging Face GGUFs match (or the Hub is unreachable).'
            }
          >
            {loading ? <SearchSkeleton /> : loadable.map(renderHfCard)}
          </CatalogGroup>
          {!loading && unimplemented.length ? (
            <details className="catalog-collapsed">
              <summary>
                {unimplemented.length} result{unimplemented.length === 1 ? '' : 's'} whose architecture
                Camelid does not implement
              </summary>
              <p className="catalog-row-faint">{HF_GUESS_EXPLANATION}</p>
              <div className="catalog-list">{unimplemented.map(renderHfCard)}</div>
            </details>
          ) : null}
          {nextCursor && !loading ? (
            <button
              type="button"
              className="catalog-row-action"
              onClick={loadMore}
              disabled={loadingMore || Boolean(pendingCatalogId)}
            >
              {loadingMore ? 'Loading…' : 'Load more from Hugging Face'}
            </button>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
