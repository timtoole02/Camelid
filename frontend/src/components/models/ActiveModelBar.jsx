import { Button } from '../ui/Button'
import { StatusDot } from '../ui/StatusDot'
import { EvidenceChip } from '../ui/EvidenceChip'
import { IconStop } from '../ui/icons'
import { laneOf } from '../../lib/modelLanes'
import { isEmbeddingOnlyModel } from '../../lib/modelCapabilities.js'

/* Zone 1 — what is loaded right now, with the one Unload action. The lane chip is
   derived for the loaded file exactly like the section rows are; runtime readiness
   comes from /health via the dashboard runtime object. */

function activeLaneChip(lane) {
  if (lane === 'supported') return <EvidenceChip state="supported" asText size="sm">Supported</EvidenceChip>
  if (lane === 'compatible') return <EvidenceChip state="runnable" asText size="sm">Runnable</EvidenceChip>
  if (lane === 'eligible') return <EvidenceChip state="runnable" asText size="sm">Ready to test</EvidenceChip>
  return <EvidenceChip state="unsupported" asText size="sm">Unverified</EvidenceChip>
}

export function ActiveModelBar({
  runtime,
  activeFilename,
  activeEntry,
  capabilities,
  busy,
  unloading,
  verification,
  verificationBusy,
  onVerify,
  onUnload,
}) {
  const online = runtime?.status === 'online'
  const generationReady = Boolean(runtime?.generation_ready)
  const loaded = Boolean(activeFilename)
  const embeddingOnly = loaded && isEmbeddingOnlyModel(activeEntry || { filename: activeFilename }, runtime)
  const readyForTask = loaded && (generationReady || embeddingOnly)
  return (
    <section className="models-active-bar" aria-label="Active model">
      <div className="models-active-bar__id">
        <StatusDot
          tone={online ? (readyForTask ? 'ready' : 'warn') : 'offline'}
          pulse={readyForTask}
          label=""
        />
        <div className="models-active-bar__name">
          <strong>{loaded ? activeFilename : 'No model loaded'}</strong>
          <span>
            {!online
              ? 'Runtime offline'
              : loaded
                ? embeddingOnly
                  ? 'Ready for embeddings · not a Chat model'
                  : generationReady
                  ? 'Generation-ready'
                  : 'Loaded, but not generation-ready yet'
                : 'Load a model below to unlock chat.'}
          </span>
        </div>
      </div>
      <div className="models-active-bar__actions">
        {loaded && activeEntry ? activeLaneChip(laneOf(activeEntry, capabilities)) : null}
        {loaded && !embeddingOnly && verification?.report?.outcome === 'verified' ? (
          <EvidenceChip
            state="evidence"
            size="sm"
            source={{
              rowId: verification.report.profile_id,
              detail: 'One pinned deterministic request matched for this exact GGUF. This is not a broad support claim.',
            }}
          >
            Verified on this machine
          </EvidenceChip>
        ) : null}
        {loaded && !embeddingOnly && verification?.report?.outcome === 'not_verified' ? (
          <EvidenceChip state="error" asText size="sm">Not verified</EvidenceChip>
        ) : null}
        {loaded && !embeddingOnly && verification?.eligible ? (
          <Button variant="outline" size="sm" onClick={onVerify} loading={verificationBusy} disabled={busy}>
            {verification?.report ? 'Verify again' : 'Verify'}
          </Button>
        ) : null}
        {loaded ? (
          <Button
            variant="outline"
            size="sm"
            icon={<IconStop size={15} />}
            onClick={onUnload}
            loading={unloading}
            disabled={busy}
          >
            Unload
          </Button>
        ) : null}
      </div>
    </section>
  )
}

export default ActiveModelBar
