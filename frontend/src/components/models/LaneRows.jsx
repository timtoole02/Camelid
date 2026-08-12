import { Button } from '../ui/Button'
import { EvidenceChip } from '../ui/EvidenceChip'
import { IconPlay, IconStop, IconTrash } from '../ui/icons'
import { ParityReceiptCard } from '../chat/render/ParityReceipt'
import { quantAdvice } from '../../lib/catalogBrowse'
import { formatBytes } from '../../lib/formatters'
import { isEmbeddingOnlyModel, isGenerationCapableModel } from '../../lib/modelCapabilities.js'

/* Lane row components for the Models page — moved verbatim from
   LocalLaneSections when the page was consolidated into zones. Copper is
   reserved for supported; runnable is amber and never copper; the
   not-yet-runnable state is shown, never hidden. */

export function metaLine(entry) {
  /* Context windows are powers of two: divide by 1024 so 32768 reads as the
     conventional 32K (and 131072 as 128K), never 33K. */
  const ctx = entry.context_length
    ? `${entry.context_length >= 1024 ? `${Math.round(entry.context_length / 1024)}K` : entry.context_length} ctx`
    : null
  /* The quant token alone is jargon; attach the human-language note from the
     shared quant table. tokenizer_kind stays in the ModelInspector, where it
     belongs — it carries no decision value on this surface. */
  const advice = quantAdvice(entry.quantization).note
  const quant = entry.quantization
    ? advice
      ? `${entry.quantization} (${advice})`
      : entry.quantization
    : null
  return [
    entry.architecture,
    quant,
    entry.size_bytes ? formatBytes(entry.size_bytes) : null,
    ctx,
  ]
    .filter(Boolean)
    .join(' · ')
}

/* What the MODEL is GOOD AT — its strengths/use-cases, by family. Independent of any
   system, hardware, or lane: this describes the model, not where it runs. */
export function describeModel(entry) {
  if (isEmbeddingOnlyModel(entry)) return 'Creates vector embeddings for search, retrieval, and reranking. It is not a Chat model.'
  if (!isGenerationCapableModel(entry)) return 'A companion model asset used by another runtime; it is not a standalone Chat model.'
  const name = (entry.filename || '').toLowerCase()
  if (name.includes('mistral')) return 'Good at reasoning, coding, and following detailed instructions.'
  if (name.includes('tinyllama')) return 'A tiny model for quick, simple chat and experiments.'
  switch (entry.architecture) {
    case 'qwen3':
      return 'Good at multilingual chat, reasoning, coding, and math.'
    case 'gemma':
    case 'gemma3':
    case 'gemma4':
      return 'Good at multilingual chat and general reasoning.'
    case 'phi3':
      return 'Good at reasoning, math, and coding in a compact model.'
    case 'llama':
      return 'Good at everyday chat, summarizing, and writing.'
    default:
      return entry.chat_capable
        ? 'Good at general chat and instruction following.'
        : 'Text generation.'
  }
}

export function Section({ title, subtitle, count, children }) {
  return (
    <section className="lane-section">
      <header className="lane-section-head">
        <h2>
          {title} {count !== undefined && <span className="lane-section-count">{count}</span>}
        </h2>
        {subtitle ? <p className="lane-section-sub">{subtitle}</p> : null}
      </header>
      <div className="lane-section-body">{children}</div>
    </section>
  )
}

function DeleteModelButton({ entry, busy, blockedReason, onDelete }) {
  if (!entry.delete_token) return null
  return (
    <Button
      variant="ghost"
      size="sm"
      className="cxv-danger"
      icon={<IconTrash size={17} />}
      onClick={() => onDelete(entry)}
      disabled={busy || Boolean(blockedReason)}
      aria-label={`Delete ${entry.filename} from disk`}
      aria-describedby={blockedReason ? 'model-delete-guard' : undefined}
      title={blockedReason || 'Delete from disk'}
    >
      Delete
    </Button>
  )
}

function RuntimeModelControl({ entry, resident, busy, deleteBusy, onUse, onUnload }) {
  if (!isEmbeddingOnlyModel(entry) && !isGenerationCapableModel(entry)) return null
  if (resident) {
    if (!isEmbeddingOnlyModel(entry) || !onUnload) return null
    return (
      <Button
        variant="outline"
        size="sm"
        icon={<IconStop size={16} />}
        onClick={() => onUnload(entry.filename)}
        loading={busy}
        disabled={busy || deleteBusy}
        aria-label={`Unload ${entry.filename}`}
      >
        Unload
      </Button>
    )
  }
  return (
    <Button
      variant="tonal"
      size="sm"
      icon={<IconPlay size={16} />}
      onClick={onUse}
      loading={busy}
      disabled={busy || deleteBusy}
      aria-label={`Load ${entry.filename}`}
      title="Load this model into Camelid"
    >
      Load
    </Button>
  )
}

function DefaultModelControl({ entry, isDefault, busy, saving, onMakeDefault }) {
  if (isDefault) {
    return (
      <span className="lane-row-default" title="Camelid loads this model when the app starts">
        ★ Starts automatically
      </span>
    )
  }
  if (isEmbeddingOnlyModel(entry)) {
    return (
      <span className="lane-row-default" title="Embedding models load as sidecars and do not replace the startup Chat model">
        Embedding sidecar
      </span>
    )
  }
  if (!isGenerationCapableModel(entry)) {
    return <span className="lane-row-default">Companion asset</span>
  }
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => onMakeDefault(entry.filename)}
      loading={saving}
      disabled={busy}
      title="Load this model automatically when Camelid starts"
    >
      Make default
    </Button>
  )
}

export function SupportedRow({
  entry,
  active,
  resident = active,
  busy,
  deleteBusy,
  defaultBusy,
  isDefault,
  blockedReason,
  onUse,
  onUnload,
  onDelete,
  onMakeDefault,
}) {
  return (
    <article
      className={`lane-row lane-row--supported${resident ? ' lane-row--active' : ''}`}
      aria-label={`Supported model ${entry.filename}`}
    >
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="supported" asText>Supported</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      {resident ? (
        <p className="lane-row-loaded">
          ● {isEmbeddingOnlyModel(entry) ? 'Ready for embeddings — not available in Chat.' : active ? 'Loaded — this is the active chat model.' : 'Loaded in Camelid.'}
        </p>
      ) : null}
      <div className="lane-row-actions">
        <RuntimeModelControl entry={entry} resident={resident} busy={busy} deleteBusy={deleteBusy} onUse={onUse} onUnload={onUnload} />
        <DefaultModelControl
          entry={entry}
          isDefault={isDefault}
          busy={defaultBusy || busy || deleteBusy}
          saving={defaultBusy}
          onMakeDefault={onMakeDefault}
        />
        {!resident ? (
          <DeleteModelButton entry={entry} busy={busy || deleteBusy || defaultBusy} blockedReason={blockedReason} onDelete={onDelete} />
        ) : null}
      </div>
    </article>
  )
}

export function CompatibleRow({
  entry,
  receipt,
  exactRowVerified = false,
  numericalVariance = false,
  active = false,
  resident = active,
  busy,
  deleteBusy,
  defaultBusy,
  isDefault,
  blockedReason,
  onUse,
  onUnload,
  onDelete,
  onMakeDefault,
}) {
  return (
    <article className={`lane-row lane-row--runnable${resident ? ' lane-row--active' : ''}`} aria-label={`Compatible model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="runnable" asText>{exactRowVerified ? 'Verified' : numericalVariance ? 'Runnable' : 'Local test passed'}</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      {receipt ? (
        <ParityReceiptCard receipt={receipt} />
      ) : null}
      <p className="lane-row-faint">
        {exactRowVerified
          ? 'This exact row passed load, deterministic output comparison, and guarded app/API checks. Extended-context support is still pending.'
          : numericalVariance
            ? 'This exact model loads and generates normally. Some deterministic token IDs differ from the pinned reference, so it remains outside Verified and Supported.'
          : isEmbeddingOnlyModel(entry)
            ? 'This model passed a quick local embedding test, but its vectors are not verified for broad use.'
            : <>This model passed a quick local run test. Reference-output verification is still pending.</>}
      </p>
      {resident ? (
        <p className="lane-row-loaded">
          ● {isEmbeddingOnlyModel(entry) ? 'Ready for embeddings — not available in Chat.' : active ? 'Loaded — this is the active chat model.' : 'Loaded in Camelid.'}
        </p>
      ) : null}
      <div className="lane-row-actions">
        <RuntimeModelControl entry={entry} resident={resident} busy={busy} deleteBusy={deleteBusy} onUse={onUse} onUnload={onUnload} />
        <DefaultModelControl
          entry={entry}
          isDefault={isDefault}
          busy={defaultBusy || busy || deleteBusy}
          saving={defaultBusy}
          onMakeDefault={onMakeDefault}
        />
        {!resident ? <DeleteModelButton entry={entry} busy={busy || deleteBusy || defaultBusy} blockedReason={blockedReason} onDelete={onDelete} /> : null}
      </div>
    </article>
  )
}

export function EligibleRow({ entry, busy, deleteBusy, blockedReason, onRun, onDelete }) {
  return (
    <article className="lane-row lane-row--runnable" aria-label={`Ready-to-test model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="runnable" asText>Ready to test</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      <div className="lane-row-actions">
        <Button
          variant="tonal"
          size="sm"
          onClick={onRun}
          loading={busy}
          disabled={busy || deleteBusy}
          title="Run a quick local test of this model"
        >
          Run quick test
        </Button>
        <DeleteModelButton entry={entry} busy={busy || deleteBusy} blockedReason={blockedReason} onDelete={onDelete} />
      </div>
    </article>
  )
}

export function NotAnchoredRow({
  entry,
  active = false,
  resident = active,
  busy,
  deleteBusy,
  defaultBusy,
  isDefault,
  blockedReason,
  onUse,
  onUnload,
  onDelete,
  onMakeDefault,
}) {
  return (
    <article className={`lane-row lane-row--blocked${resident ? ' lane-row--active' : ''}`} aria-label={`Unverified model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="unsupported" asText>Unverified</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      <p className="lane-row-note">
        {isEmbeddingOnlyModel(entry)
          ? 'This embedding runtime loads and runs, but its vectors have not been verified.'
          : 'This model loads and runs, but its output isn’t verified for correctness.'}
        {' '}Use it with that limitation in mind.
      </p>
      {resident ? (
        <p className="lane-row-loaded">
          ● {isEmbeddingOnlyModel(entry) ? 'Ready for embeddings — not available in Chat.' : active ? 'Loaded — this is the active chat model.' : 'Loaded in Camelid.'}
        </p>
      ) : null}
      <div className="lane-row-actions">
        <RuntimeModelControl entry={entry} resident={resident} busy={busy} deleteBusy={deleteBusy} onUse={onUse} onUnload={onUnload} />
        <DefaultModelControl
          entry={entry}
          isDefault={isDefault}
          busy={defaultBusy || busy || deleteBusy}
          saving={defaultBusy}
          onMakeDefault={onMakeDefault}
        />
        {!resident ? <DeleteModelButton entry={entry} busy={busy || deleteBusy || defaultBusy} blockedReason={blockedReason} onDelete={onDelete} /> : null}
      </div>
    </article>
  )
}
