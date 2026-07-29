import { EvidenceChip } from '../ui/EvidenceChip'
import { IconTrash } from '../ui/icons'
import { ParityReceiptCard } from '../chat/render/ParityReceipt'

/* Lane row components for the Models page — moved verbatim from
   LocalLaneSections when the page was consolidated into zones. Copper is
   reserved for supported; runnable is amber and never copper; the
   not-yet-runnable state is shown, never hidden. */

const GB = 1024 * 1024 * 1024

export function prettySize(bytes) {
  if (!bytes) return ''
  if (bytes >= GB) return `${(bytes / GB).toFixed(bytes >= 10 * GB ? 0 : 1)} GB`
  return `${Math.round(bytes / (1024 * 1024))} MB`
}

export function metaLine(entry) {
  const ctx = entry.context_length
    ? `${entry.context_length >= 1000 ? `${Math.round(entry.context_length / 1000)}K` : entry.context_length} ctx`
    : null
  return [entry.architecture, entry.quantization, entry.tokenizer_kind, prettySize(entry.size_bytes), ctx]
    .filter(Boolean)
    .join(' · ')
}

/* What the MODEL is GOOD AT — its strengths/use-cases, by family. Independent of any
   system, hardware, or lane: this describes the model, not where it runs. */
export function describeModel(entry) {
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
        <h3>
          {title} {count !== undefined && <span className="lane-section-count">{count}</span>}
        </h3>
        {subtitle ? <p className="lane-section-sub">{subtitle}</p> : null}
      </header>
      <div className="lane-section-body">{children}</div>
    </section>
  )
}

function DeleteModelButton({ entry, busy, blockedReason, onDelete }) {
  if (!entry.delete_token) return null
  return (
    <button
      type="button"
      className="lane-row-delete"
      onClick={() => onDelete(entry)}
      disabled={busy || Boolean(blockedReason)}
      aria-label={`Delete ${entry.filename} from disk`}
      aria-describedby={blockedReason ? 'model-delete-guard' : undefined}
      title={blockedReason || 'Delete from disk'}
    >
      <IconTrash size={18} />
    </button>
  )
}

function DefaultModelControl({ entry, isDefault, busy, onMakeDefault }) {
  if (isDefault) {
    return (
      <span className="lane-row-default" title="Camelid loads this model when the app starts">
        ★ Starts automatically
      </span>
    )
  }
  return (
    <button
      type="button"
      className="lane-row-default-action"
      onClick={() => onMakeDefault(entry.filename)}
      disabled={busy}
      title="Load this model automatically the next time Camelid starts"
    >
      {busy ? 'Saving…' : 'Make default'}
    </button>
  )
}

export function SupportedRow({
  entry,
  active,
  busy,
  deleteBusy,
  defaultBusy,
  isDefault,
  blockedReason,
  onUse,
  onDelete,
  onMakeDefault,
}) {
  return (
    <article
      className={`lane-row lane-row--supported${active ? ' lane-row--active' : ''}`}
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
      {active ? <p className="lane-row-loaded">● Loaded — this is the active chat model.</p> : null}
      <div className="lane-row-actions">
        {!active ? (
          <button type="button" className="lane-row-action" onClick={onUse} disabled={busy || deleteBusy}>
            {busy ? 'Loading…' : 'Use for chat'}
          </button>
        ) : null}
        <DefaultModelControl
          entry={entry}
          isDefault={isDefault}
          busy={defaultBusy || busy || deleteBusy}
          onMakeDefault={onMakeDefault}
        />
        {!active ? (
          <DeleteModelButton entry={entry} busy={busy || deleteBusy || defaultBusy} blockedReason={blockedReason} onDelete={onDelete} />
        ) : null}
      </div>
    </article>
  )
}

export function CompatibleRow({
  entry,
  receipt,
  busy,
  deleteBusy,
  defaultBusy,
  isDefault,
  blockedReason,
  onUse,
  onDelete,
  onMakeDefault,
}) {
  return (
    <article className="lane-row lane-row--runnable" aria-label={`Compatible model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="runnable" asText>Runnable</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      {receipt ? (
        <ParityReceiptCard receipt={receipt} />
      ) : (
        <p className="lane-row-faint">Loading runnable receipt…</p>
      )}
      <p className="lane-row-faint">
        The receipt above attests one deterministic CLI run on the generic f32 lane
        (<code>camelid runnable-smoke</code>) — it is not a parity contract. Loading it for
        chat serves it the same way any implemented-but-unanchored model is served, so its
        chat output carries no parity claim and is not cross-validated against the reference.
      </p>
      <div className="lane-row-actions">
        <button type="button" className="lane-row-action" onClick={onUse} disabled={busy || deleteBusy}>
          {busy ? 'Loading…' : 'Use for chat (experimental)'}
        </button>
        <DefaultModelControl
          entry={entry}
          isDefault={isDefault}
          busy={defaultBusy || busy || deleteBusy}
          onMakeDefault={onMakeDefault}
        />
        <DeleteModelButton entry={entry} busy={busy || deleteBusy || defaultBusy} blockedReason={blockedReason} onDelete={onDelete} />
      </div>
    </article>
  )
}

export function EligibleRow({ entry, busy, deleteBusy, blockedReason, onRun, onDelete }) {
  return (
    <article className="lane-row lane-row--runnable" aria-label={`Smoke-eligible model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="runnable" asText>Oracle-qualified</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      <div className="lane-row-actions">
        <button type="button" className="lane-row-action" onClick={onRun} disabled={busy || deleteBusy}>
          {busy ? 'Running smoke-admission…' : 'Run smoke-admission'}
        </button>
        <DeleteModelButton entry={entry} busy={busy || deleteBusy} blockedReason={blockedReason} onDelete={onDelete} />
      </div>
    </article>
  )
}

export function NotAnchoredRow({
  entry,
  busy,
  deleteBusy,
  defaultBusy,
  isDefault,
  blockedReason,
  onUse,
  onDelete,
  onMakeDefault,
}) {
  return (
    <article className="lane-row lane-row--blocked" aria-label={`Experimental model ${entry.filename}`}>
      <div className="lane-row-head">
        <div className="lane-row-id">
          <span className="lane-row-name">{entry.filename}</span>
          <span className="lane-row-meta">{metaLine(entry)}</span>
        </div>
        <EvidenceChip state="unsupported" asText>Experimental — unverified</EvidenceChip>
      </div>
      <p className="lane-row-note">{describeModel(entry)}</p>
      <p className="lane-row-note">
        Implemented but not parity-anchored: it loads and runs (GPU-resident when it
        fits), but its output is not cross-validated against the reference. For
        experimentation only.
      </p>
      <div className="lane-row-actions">
        <button type="button" className="lane-row-action" onClick={onUse} disabled={busy || deleteBusy}>
          {busy ? 'Loading…' : 'Use for chat (experimental)'}
        </button>
        <DefaultModelControl
          entry={entry}
          isDefault={isDefault}
          busy={defaultBusy || busy || deleteBusy}
          onMakeDefault={onMakeDefault}
        />
        <DeleteModelButton entry={entry} busy={busy || deleteBusy || defaultBusy} blockedReason={blockedReason} onDelete={onDelete} />
      </div>
    </article>
  )
}
