import { EvidenceChip } from '../ui/EvidenceChip'
import { Button } from '../ui/Button'

/* Fail-closed blocker surface for a GGUF whose architecture Camelid does not
   implement (or whose metadata is invalid). Shows the EXACT typed reason verbatim
   and states that chat is disabled — never a "try anyway" that would route to a
   different inference path. When the backend's reason names a dedicated lane
   (e.g. DiffusionGemma's `camelid diffusion-gemma-chat`), that command is pulled
   out as a copyable redirect. `blocker` is `{ code, message }`.

   The capacity refusals are the one family that DOES get an override, and it is
   not the exception it looks like: they are the verdict of a memory ESTIMATE about
   this machine, not a statement that no correct code path exists. Forcing re-runs
   the identical load on the identical path — the only thing skipped is the
   estimate — so nothing here can produce output from an unverified runtime. The
   override has to live in the UI because the engine's escape hatch is an
   environment variable and the desktop sidecar is spawned args-only; telling a
   GUI user to set one is advice they cannot act on. */

const DEDICATED_LANE_COMMAND = /(camelid\s+diffusion-gemma-chat[^\n.]*)/i

export function blockerNoteFor(blocker) {
  if (blocker?.code === 'model_io_error') {
    return 'Camelid could not open this model from the configured storage location, so chat stays disabled until the path is corrected.'
  }
  if (blocker?.code === 'model_too_large_for_host') {
    return 'This model’s estimated footprint is larger than this machine can hold, so Camelid stopped before the load rather than risking an unstable one. A smaller model or a smaller quantization is the reliable fix.'
  }
  if (blocker?.code === 'host_memory_exhausted') {
    return 'This allocation is larger than this machine can supply even after evicting everything evictable — the shortfall is wired memory the system will not release. Closing applications does not recover it; a smaller model or a smaller quantization does.'
  }
  if (blocker?.code === 'host_memory_unavailable') {
    return 'This machine is big enough — the memory is in use right now. Closing some applications frees it, and the load then succeeds unchanged.'
  }
  if (blocker?.code === 'model_requires_unload') {
    return 'Another model is resident and holding the memory this one needs. Unload it from the active model bar, then load this one.'
  }
  if (blocker?.code === 'unsupported_model_architecture') {
    return 'This architecture is not implemented, so chat stays disabled — Camelid fails closed rather than emit plausible-but-wrong tokens on a different code path.'
  }
  return 'This model did not pass Camelid’s runtime admission checks, so chat stays disabled rather than using an unverified fallback path.'
}

/* Refusals that came from the memory estimate rather than from a missing
   implementation. Only these may be overridden. */
const FORCEABLE_REFUSALS = new Set([
  'host_memory_exhausted',
  'host_memory_unavailable',
  'model_too_large_for_host',
])

export function blockerIsForceable(blocker) {
  return FORCEABLE_REFUSALS.has(blocker?.code)
}

/* What overriding actually costs, per code. Vague reassurance here would be the
   same failure as the environment-variable advice it replaces. */
export function forceRiskFor(blocker) {
  if (blocker?.code === 'host_memory_exhausted') {
    return 'Loading anyway asks this machine for memory it does not currently have. The load may fail part-way, and the attempt itself can push the whole system into heavy swapping until it does.'
  }
  if (blocker?.code === 'model_too_large_for_host') {
    return 'Loading anyway is likely to fail outright, and this machine may become unresponsive while it tries. Do this only if you have reason to believe the footprint estimate is wrong for your hardware.'
  }
  return 'Loading anyway competes for the memory another application is holding. If that application does not release it, the load fails; nothing else on this machine is changed by trying.'
}

export function UnsupportedBlocker({ blocker, className = '', onForceLoad = null, forceBusy = false }) {
  if (!blocker?.message) return null
  const redirect = blocker.message.match(DEDICATED_LANE_COMMAND)?.[1]?.trim() || null
  const forceable = Boolean(onForceLoad) && blockerIsForceable(blocker)

  return (
    /* views.css colors this surface's edge with --color-danger, a token
       tokens.css never defines (its fallback hex was rendering instead). Bind it
       to the real error token here so the edge tracks the theme. */
    <div
      className={`unsupported-blocker ${className}`.trim()}
      role="alert"
      style={{ '--color-danger': 'var(--color-error)' }}
    >
      <div className="unsupported-blocker__head">
        <EvidenceChip state="unsupported" asText size="sm">Fail-closed</EvidenceChip>
        {blocker.code ? <code className="unsupported-blocker__code">{blocker.code}</code> : null}
      </div>
      <p className="unsupported-blocker__message">{blocker.message}</p>
      <p className="unsupported-blocker__note">{blockerNoteFor(blocker)}</p>
      {redirect ? (
        <p className="unsupported-blocker__redirect">
          Dedicated lane: <code>{redirect}</code>
        </p>
      ) : null}
      {forceable ? (
        <div className="unsupported-blocker__override">
          <p className="unsupported-blocker__risk">{forceRiskFor(blocker)}</p>
          <Button
            variant="outline"
            size="sm"
            loading={forceBusy}
            disabled={forceBusy}
            onClick={onForceLoad}
          >
            Load anyway
          </Button>
        </div>
      ) : null}
    </div>
  )
}

export default UnsupportedBlocker
