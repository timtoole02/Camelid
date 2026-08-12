import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { beginCatalogSettlement, catalogDownloadSettlement, completeCatalogAcquisition } from '../../lib/catalogActivation'
import {
  catalogBundleInstalled,
  catalogCompanionArtifacts,
  missingCatalogArtifacts,
} from '../../lib/catalogCompanions'
import {
  catalogQualification,
  defaultFileIndex,
  fitDetail,
  fitIsRecheckable,
  fitIsSettled,
  fitLabel,
  ghostMoeFits,
  groupHfFilesByRepo,
  isRefusingFit,
  partitionByArchSupport,
  partitionCuratedByFit,
  predictedLane,
  quantAdvice,
} from '../../lib/catalogBrowse'
import { SUPPORTED_MODELS } from '../../lib/supportedModels'
import { formatBytes } from '../../lib/formatters'
import { isEmbeddingOnlyModel } from '../../lib/modelCapabilities.js'
import { Button } from '../ui/Button'
import { EvidenceChip } from '../ui/EvidenceChip'
import { IconCheck } from '../ui/icons'
import { Notice } from '../ui/Notice'
import { ModelFamilyGroups } from './ModelFamilyGroups'

/* Zone 5 — Get models. Curated picks first, then live Hugging Face GGUF search
   (>= 2 chars). Each row shows which lane it WOULD land in (derived: supported
   contract match, oracle-qualified runnable, or not-yet-anchored). Download is
   user-initiated and explicitly confirmed (filename + HF repo + size); no
   background/auto pulls. Live progress renders in the global Downloads zone —
   rows here only reflect their own acquisition state, read from the shared
   downloads poll + the live /api/models/local scan (never localStorage). After a
   supported or runnable curated download lands, Camelid starts it through the
   normal guarded load path; unanchored Hub guesses remain download-only.

   Hugging Face results are grouped into one card per repo with a quantization
   picker. The Hub returns files, and a single repo routinely ships 20-27 quants of
   the same weights, so rendering them flat turned a 15-repo search into a 250-row
   wall. Grouping is presentation only: a download is still one exact file,
   confirmed by name. */

/* Curated download suggestions (blurbs, "Recommended") may DECORATE catalog rows,
   never place them: lane membership and outcome chips stay derived. */
const CURATED_DECORATION = new Map(SUPPORTED_MODELS.map((item) => [item.catalog_id, item]))

/* Why a live Hugging Face row can never imply support — said once per section and
   in a tooltip on the guessed value, rather than as a paragraph on all 150 rows. */
const HF_GUESS_EXPLANATION =
  'Architecture and quantization are read from the filename, not the model. The real lane is only known after the file loads.'

function qualificationChip(qualification) {
  if (qualification.kind === 'supported') {
    return <EvidenceChip state="supported" asText>Supported</EvidenceChip>
  }
  if (qualification.kind === 'verified' || qualification.kind === 'runnable') {
    return <EvidenceChip state="runnable" asText>{qualification.label}</EvidenceChip>
  }
  if (qualification.kind === 'variance') {
    return <EvidenceChip state="evidence" asText>{qualification.label}</EvidenceChip>
  }
  if (qualification.kind === 'limited' || qualification.kind === 'validating') {
    return <EvidenceChip state="evidence" asText>{qualification.label}</EvidenceChip>
  }
  return <EvidenceChip state="unsupported" asText>{qualification.label}</EvidenceChip>
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
    /* A settled question with no verdict is described ONCE above the list (see
       UndeterminedFitNotice) instead of repeating the same disclaimer on every card. */
    if (fitIsSettled(item)) return null
    if (!onCheckFit) return null
    return (
      <div className="catalog-fit-row">
        <span className="catalog-fit-chip catalog-fit-chip--unknown">
          <FitIcon />
          Fit unknown until checked
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={onCheckFit}
          loading={checking}
          disabled={checking}
        >
          {checking ? 'Checking…' : item.fit_checked === false ? 'Try again' : 'Check if it fits'}
        </Button>
        {item.fit_checked === false ? (
          <span className="catalog-fit-detail">
            Could not reach Hugging Face for this model’s details. Retrying may work.
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
        <Button
          variant="ghost"
          size="sm"
          onClick={onCheckFit}
          loading={checking}
          disabled={checking}
          title="Re-read this machine's free memory and this model's size, right now"
        >
          {checking ? 'Re-checking…' : 'Re-check'}
        </Button>
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
      {/* The estimate caveat must be visible, not title-only: tooltips never reach
          keyboard or touch users, who are exactly who the ~ prefix confuses. */}
      {item.fit_confidence === 'approx' ? (
        <span className="catalog-fit-detail">
          Estimated from file size — the verdict becomes exact once the model header has been read.
        </span>
      ) : null}
    </div>
  )
}

/* Some models' size simply can't be verified on this machine. Said ONCE, above the
   list, instead of stamping the same chip and sentence onto every affected card. */
function UndeterminedFitNotice({ items }) {
  const anyUndetermined = items.some((item) => !fitLabel(item.fit) && fitIsSettled(item))
  if (!anyUndetermined) return null
  return (
    <div className="catalog-fit-row">
      <span className="catalog-fit-chip catalog-fit-chip--unknown">
        <FitIcon />
        Fit can’t be checked for some models
      </span>
      {/* The reason lives in visible text, not a title tooltip — tooltips never
          reach keyboard or touch users. */}
      <span className="catalog-fit-detail">
        Their dimensions can’t be read, or this machine’s free memory can’t be measured. Their
        downloads aren’t blocked — Camelid just can’t confirm in advance that they fit this machine.
      </span>
    </div>
  )
}

function GhostMoeOption({ item, selected, prepared, disabled, onChange }) {
  const ghost = item?.ghost_moe
  if (!ghost?.available) return null
  const fits = ghostMoeFits(item)
  return (
    <div className={`catalog-ghost-option${fits ? ' catalog-ghost-option--ready' : ''}`}>
      <label className="catalog-ghost-toggle">
        <input
          type="checkbox"
          checked={prepared || selected}
          disabled={prepared || disabled || !fits}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>
          <strong>{prepared ? 'Ghost MoE enabled' : 'Use Ghost MoE'}</strong>
          {ghost.recommended && !prepared ? <em>Recommended for this GPU</em> : null}
        </span>
      </label>
      <p>
        Keeps the common core and hot experts on CUDA, then pages routed experts from disk.
        The prepared install uses about {formatBytes(ghost.installed_bytes)}; preparation temporarily
        needs up to {formatBytes(ghost.peak_disk_bytes)} total.
      </p>
      {!ghost.host_eligible ? (
        <p className="catalog-ghost-warning">Ghost MoE requires Windows and a compatible CUDA GPU.</p>
      ) : ghost.fit === 'insufficient_free_memory' ? (
        <p className="catalog-ghost-warning">
          This GPU has enough total VRAM. Camelid will release its current model before starting Ghost MoE;
          other GPU applications may also need to be closed.
        </p>
      ) : !fits ? (
        <p className="catalog-ghost-warning">The Ghost common resident set does not fit the currently available GPU memory.</p>
      ) : null}
    </div>
  )
}

function CatalogRow({
  item,
  capabilities,
  installed,
  ghostMoePrepared = false,
  missingArtifacts = [],
  activeDownload,
  apiBase,
  installAvailable,
  installBlockedReason,
  onInstallStarted,
  onDownloadAcknowledged,
  onAcquired,
  canceled,
  onDownloadRetry,
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
  // Ghost-MoE changes the on-disk representation, so keep it an explicit user
  // opt-in. `recommended` decorates the option but must never tick it on behalf
  // of the user; prepared installs remain checked and locked by design.
  const [ghostMoeSelected, setGhostMoeSelected] = useState(() => Boolean(ghostMoePrepared))
  const [settlementTick, setSettlementTick] = useState(0)
  const sawDownloadRef = useRef(false)
  const startedAtRef = useRef(0)
  const settledAtRef = useRef(0)
  const acquisitionModeRef = useRef('download')
  const acquisitionItemRef = useRef(item)
  const settlementInFlightRef = useRef(false)
  const qualification = catalogQualification(item, capabilities)
  const lane = predictedLane(item, capabilities)
  const embeddingOnly = isEmbeddingOnlyModel(item)
  const decoration = item.group === 'experimental' ? null : CURATED_DECORATION.get(item.catalog_id)
  // ANY load-refusing verdict must stop the auto-start chain, not just `wont_fit`:
  // the load-time guard refuses both, so chaining into it would end in a 422.
  const refusedByFit = isRefusingFit(item.fit)
  const ghostMoeUsable = ghostMoeFits(item)
  const useGhostMoe = ghostMoePrepared || (ghostMoeSelected && ghostMoeUsable)
  const effectiveFitRefusal = refusedByFit && !useGhostMoe
  // A compatible curated row is already implemented and oracle-anchored. The
  // authoritative inspect/load path below performs the real header and runtime
  // safety checks, so forcing a second, minute-long smoke admission before the
  // user can start it adds latency without adding a load-time guard. Live Hub
  // guesses remain download-only because predictedLane deliberately returns
  // `not_anchored` for them.
  const downloadAndStart = (lane === 'supported' || lane === 'compatible') && !effectiveFitRefusal
  const acquisitionMode = downloadAndStart ? 'start' : 'download'
  const downloading = activeDownload?.status === 'downloading' || activeDownload?.status === 'preparing'
  const preparingGhost = activeDownload?.status === 'preparing'
  const downloadFailed = activeDownload?.status === 'failed'
  const rejoinableDownload = downloading || activeDownload?.status === 'completed'
  const operationBusy = phase === 'checking' || phase === 'loading'
  const companions = catalogCompanionArtifacts(item)
  const primaryInstalled = !missingArtifacts.some((artifact) => artifact.role === 'model')
  const onlyCompanionsMissing = primaryInstalled && missingArtifacts.length > 0
  const missingBytes = missingArtifacts.reduce(
    (total, artifact) => total + Number(artifact.size_bytes || 0),
    0,
  )

  useEffect(() => {
    onOperationBusy?.(item.catalog_id, operationBusy)
    return () => {
      if (operationBusy) onOperationBusy?.(item.catalog_id, false)
    }
  }, [item.catalog_id, onOperationBusy, operationBusy])

  useEffect(() => {
    if (phase !== 'idle' || !rejoinableDownload) return
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
  }, [acquisitionMode, activeDownload?.continuation_mode, item, onAcquisitionPending, phase, rejoinableDownload])

  const finishLanded = useCallback(async () => {
    if (!beginCatalogSettlement(settlementInFlightRef)) return
    setIsError(false)
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
      setMessage('Wait for the current download to finish, then retry.')
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
      failed: downloadFailed,
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
  }, [phase, downloading, downloadFailed, installed, canceled, settlementTick, finishLanded, item.catalog_id, onAcquisitionSettled])

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
          ghost_moe: useGhostMoe,
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
              {item.repo_id} · {item.filename} · {formatBytes(item.size_bytes)}
              {item.architecture ? ` · ${item.architecture}` : ''}
            </span>
          </div>
          {qualificationChip(qualification)}
        </div>
      )}
      <FitAdvisory item={item} onCheckFit={onCheckFit} checking={checkingFit} />
      {qualification.detail ? <p className="catalog-row-faint">{qualification.detail}</p> : null}
      {decoration?.blurb ? <p className="catalog-row-blurb">{decoration.blurb}</p> : null}
      <GhostMoeOption
        item={item}
        selected={ghostMoeSelected}
        prepared={ghostMoePrepared}
        disabled={!['idle', 'confirm'].includes(phase)}
        onChange={setGhostMoeSelected}
      />

      {showProgress ? (
        <div className="catalog-start" role="status" aria-live="polite">
          {downloadAndStart ? (
            <ol className="catalog-start-steps" aria-label="Download and start progress">
              {['Download', 'Prepare', 'Start'].map((label, index) => (
                <li key={label} className={index < activeStage ? 'is-done' : index === activeStage ? 'is-active' : ''}>
                  <span>{index < activeStage ? <IconCheck size={12} /> : index + 1}</span>
                  {label}
                </li>
              ))}
            </ol>
          ) : null}
          <p className="catalog-row-faint">
            {phase === 'checking'
              ? 'Download complete — preparing the model…'
              : phase === 'loading'
                ? embeddingOnly
                  ? 'Starting the embedding sidecar…'
                  : 'Starting the model for Chat…'
                : preparingGhost
                  ? 'Preparing Ghost MoE — repacking routed experts and reclaiming the full download…'
                : downloading
                  ? activeDownload?.current_artifact_role === 'vision_projector'
                    ? 'Downloading the vision projector — live progress is shown above.'
                    : 'Downloading the model — live progress is shown above.'
                  : 'Starting download…'}
          </p>
        </div>
      ) : phase === 'failed' ? (
        <div className="catalog-start-failure" role="alert">
          <p className="catalog-row-error">{message}</p>
          <p className="catalog-row-faint">
            Anything already downloaded stays on disk. Chat opens only once everything needed has finished downloading.
          </p>
          <Button variant="outline" size="sm" onClick={retryAcquisition}>
            Retry start
          </Button>
        </div>
      ) : phase === 'done' ? (
        <p className={isError ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p>
      ) : installed && (!useGhostMoe || ghostMoePrepared) ? (
        <p className="catalog-row-faint">
          Already on disk{ghostMoePrepared ? ' with Ghost MoE enabled' : ''} — shown in its section above.
        </p>
      ) : phase === 'idle' ? (
        <>
          {onlyCompanionsMissing ? (
            <p className="catalog-row-faint">
              The model is on disk, but its required vision projector is missing. Download it to enable image prompts.
            </p>
          ) : null}
          {item.group !== 'experimental' && lane === 'not_anchored' ? (
            <p className="catalog-row-faint">
              This model can still be downloaded — it will appear under Other local models with
              the same evidence status.
            </p>
          ) : null}
          {message ? <p className={isError ? 'catalog-row-error' : 'catalog-row-faint'}>{message}</p> : null}
          {installAvailable ? (
            <Button
              variant="tonal"
              size="sm"
              onClick={openConfirmation}
            >
              {onlyCompanionsMissing
                ? downloadAndStart
                  ? 'Download vision projector and start…'
                  : 'Download vision projector…'
                : installed && useGhostMoe
                  ? 'Enable Ghost MoE and start…'
                : downloadAndStart
                  ? 'Download and start…'
                  : 'Download…'}
            </Button>
          ) : (
            <>
              <Button variant="tonal" size="sm" disabled>
                Download unavailable
              </Button>
              <p className="catalog-row-faint">{installBlockedReason}</p>
            </>
          )}
        </>
      ) : phase === 'confirm' ? (
        <div className="catalog-confirm">
          {onlyCompanionsMissing ? (
            <p>
              Download the required vision projector <strong>{missingArtifacts[0]?.filename}</strong> from{' '}
              <code>{missingArtifacts[0]?.repo_id}</code> ({formatBytes(missingBytes)})? The model is already on disk;
              Camelid will reuse it without downloading it again.
              {downloadAndStart ? ' Camelid will load the model and open Chat after the projector lands.' : ''}
            </p>
          ) : companions.length ? (
            <p>
              Download <strong>{item.filename}</strong> plus its required vision projector{' '}
              <strong>{companions[0].filename}</strong> ({formatBytes(missingBytes)})? Camelid downloads both into your
              local models folder and enables image prompts only after both files land.
              {downloadAndStart ? ' It will then start the model and open Chat.' : ''}
            </p>
          ) : (
            <p>
              {installed ? (
                <>
                  Prepare the existing <strong>{item.filename}</strong> for Ghost MoE? Camelid will reuse the local
                  GGUF without downloading it again.
                </>
              ) : (
                <>
                  Download <strong>{item.filename}</strong> from <code>{item.repo_id}</code> (
                  {formatBytes(item.size_bytes)})? This pulls from HuggingFace into your local models folder.
                </>
              )}
              {useGhostMoe
                ? ' Camelid will repack routed experts for Ghost MoE, reclaim the temporary full GGUF, start the prepared model, and open Chat.'
                : downloadAndStart
                  ? ' Camelid will start it and open Chat after the download.'
                  : ''}
            </p>
          )}
          <div className="catalog-confirm-actions">
            <Button variant="tonal" size="sm" onClick={confirmDownload}>
              {installed ? 'Confirm preparation' : 'Confirm download'}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setPhase('idle')
                onAcquisitionSettled?.(item.catalog_id)
              }}
            >
              Cancel
            </Button>
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
              <span className="hf-model-unsupported">Camelid can&rsquo;t run this model type</span>
            ) : null}
          </p>
        </div>
        <EvidenceChip state="unsupported" asText>Unverified</EvidenceChip>
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
              {candidate.quant || 'unlabelled'} · {formatBytes(candidate.size_bytes)}
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

function CatalogFamilyGroups({ items, renderItem, openFamilies = false }) {
  return (
    <ModelFamilyGroups
      items={items}
      renderItem={renderItem}
      className="model-family-group--catalog"
      initiallyOpen={() => openFamilies}
    />
  )
}

export function CatalogLaneBrowse({
  apiBase = '',
  capabilities,
  localFilenames = new Set(),
  ghostMoePreparedFilenames = new Set(),
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
  externalQuery,
  onCuratedMatchCount,
}) {
  const base = (apiBase || '').replace(/\/$/, '')
  const [items, setItems] = useState(null)
  const [query, setQuery] = useState('')
  /* The Models page owns one search for the whole page, so when it drives this
     component the box here would be a second, competing input — it is hidden and
     this query simply follows the page's. Empty is synced too: clearing up there
     must clear the catalog, not strand an old term down here. */
  const pageControlled = externalQuery !== undefined
  useEffect(() => {
    if (!pageControlled) return
    setQuery(String(externalQuery || ''))
  }, [pageControlled, externalQuery])
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [nextCursor, setNextCursor] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState('')
  const [pendingItems, setPendingItems] = useState([])
  const [checkingFitIds, setCheckingFitIds] = useState(() => new Set())
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
    // Each row owns its own acquisition state and the backend rejects only real
    // conflicts (same catalog id or overlapping destination files). Keep every
    // pending row visible, but never serialize unrelated model downloads here.
    setPendingItems((current) => [
      item,
      ...current.filter((pending) => pending.catalog_id !== item.catalog_id),
    ])
    return true
  }, [])

  const settleAcquisition = useCallback((catalogId) => {
    setPendingItems((current) => current.filter((item) => item.catalog_id !== catalogId))
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
                ghost_moe: body.ghost_moe ?? row.ghost_moe,
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
      installed={catalogBundleInstalled(item, localFilenames)}
      ghostMoePrepared={ghostMoePreparedFilenames.has(item.filename)}
      missingArtifacts={missingCatalogArtifacts(item, localFilenames)}
      activeDownload={downloads.find((download) => download.id === item.catalog_id)}
      apiBase={base}
      installAvailable={installAvailable}
      installBlockedReason={installBlockedReason}
      onInstallStarted={onInstallStarted}
      onDownloadAcknowledged={onDownloadAcknowledged}
      onAcquired={onAcquired}
      canceled={canceledCatalogIds.has(item.catalog_id)}
      onDownloadRetry={onDownloadRetry}
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
    ghostMoePreparedFilenames, installBlockedReason, isCheckingFit, localFilenames, onAcquired, onDownloadAcknowledged,
    onDownloadRetry, onInstallStarted, onModelStarted, onOperationBusy, onStartModel,
    reserveAcquisition, settleAcquisition,
  ])

  // Every row whose acquisition is in flight must keep rendering even if a newer
  // search no longer returns it, or the user loses sight of parallel downloads.
  const visibleItems = useMemo(() => {
    const rows = items || []
    const missingPending = pendingItems.filter(
      (pending) => !rows.some((item) => item.catalog_id === pending.catalog_id),
    )
    return [...missingPending, ...rows]
  }, [items, pendingItems])
  const curated = useMemo(
    () => visibleItems.filter((it) => it.group !== 'experimental'),
    [visibleItems],
  )
  const experimental = useMemo(
    () => visibleItems.filter((it) => it.group === 'experimental'),
    [visibleItems],
  )
  const searching = debouncedQuery.length >= 2

  /* The page's result line needs to know whether scrolling down is worth it, so
     report how many curated rows the current term matches. Curated only: live
     Hugging Face results arrive asynchronously, and a count that changed after
     the fact would be worse than no count. */
  useEffect(() => {
    if (!onCuratedMatchCount) return
    onCuratedMatchCount(searching ? curated.length : null)
  }, [onCuratedMatchCount, searching, curated.length])

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
        Curated picks are pinned exact files with a visible qualification result. Verified rows
        passed load, deterministic output comparison, and guarded app/API checks; rows with a hold
        show the exact reason. Live Hugging Face search results are unverified. Every download asks
        for confirmation first; progress appears in Downloads above, and finished downloads join
        the sections above.
      </p>
      {!pageControlled && (
        <input
          className="catalog-search"
          aria-label="Search model catalog"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search curated picks and live Hugging Face GGUFs (name, repo, filename)"
        />
      )}
      <UndeterminedFitNotice items={visibleItems} />
      {error ? (
        <Notice
          notice={items === null ? `Catalog unavailable: ${error}` : error}
          tone="error"
          onDismiss={() => setError('')}
        />
      ) : null}
      {items === null && !error ? <p className="lane-empty">Loading catalog…</p> : null}

      {items === null && !error ? null : searching ? (
        <CatalogGroup
          title="Curated"
          marker={null}
          count={curated.length}
          emptyText="No curated entries match."
        >
          <CatalogFamilyGroups
            items={curated}
            renderItem={(item) => renderRow(item)}
            openFamilies={searching}
          />
        </CatalogGroup>
      ) : (
        <>
          {/* The list holds everything this machine can run, including rows whose
              memory is merely occupied right now — each carries its own amber chip
              and Re-check button, so the state is visible per row instead of
              hiding the model. Only models too big for the machine fold away. */}
          <CatalogGroup
            title="Curated picks"
            marker={null}
            count={curatedRunnable.length}
            emptyText="No curated model matches."
          >
            <CatalogFamilyGroups
              items={curatedRunnable}
              renderItem={(item) => renderRow(item)}
            />
          </CatalogGroup>
          {curatedBlocked.length ? (
            <details className="catalog-collapsed">
              <summary>
                {curatedBlocked.length} model{curatedBlocked.length === 1 ? '' : 's'} too big for this
                machine
              </summary>
              <div className="catalog-list">
                <CatalogFamilyGroups
                  items={curatedBlocked}
                  renderItem={(item) => renderRow(item)}
                />
              </div>
            </details>
          ) : null}
        </>
      )}

      {searching && items !== null ? (
        <>
          <CatalogGroup
            title="Hugging Face (unverified)"
            marker={
              <span className="catalog-experimental-marker">
                <EvidenceChip state="unsupported" asText>Unverified</EvidenceChip>
              </span>
            }
            count={loading ? 1 : loadable.length}
            emptyText={
              // Saying "nothing matched" when results DID match and were merely
              // folded into the section below would be false, and it hides the one
              // place the user should look next.
              unimplemented.length
                ? `Every match (${unimplemented.length}) is a model type Camelid can't run — see below.`
                : 'No live Hugging Face GGUFs match (or the Hub is unreachable).'
            }
          >
            {loading ? <SearchSkeleton /> : (
              <CatalogFamilyGroups
                items={loadable}
                renderItem={renderHfCard}
                openFamilies={searching}
              />
            )}
          </CatalogGroup>
          {!loading && unimplemented.length ? (
            <details className="catalog-collapsed">
              <summary>
                {unimplemented.length} result{unimplemented.length === 1 ? '' : 's'} of a model type
                Camelid can&rsquo;t run
              </summary>
              <p className="catalog-row-faint">{HF_GUESS_EXPLANATION}</p>
              <div className="catalog-list">
                <CatalogFamilyGroups
                  items={unimplemented}
                  renderItem={renderHfCard}
                  openFamilies={searching}
                />
              </div>
            </details>
          ) : null}
          {nextCursor && !loading ? (
            <Button
              variant="outline"
              size="sm"
              onClick={loadMore}
              loading={loadingMore}
              disabled={loadingMore}
            >
              Load more from Hugging Face
            </Button>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
