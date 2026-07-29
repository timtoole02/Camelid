/* First-run activation: deciding whether this is a fresh install, and which single
   model to offer.

   Both answers are DERIVED from live server state on every render — never from a
   localStorage "has onboarded" flag. A stored flag is wrong in both directions (it
   survives a wiped models folder, and a new browser profile resurrects the card for
   an established install), and this is precisely the surface where a stale profile
   has hidden a first-load bug before.

   The recommendation is derived too: it is whatever the catalog and the support
   contract say right now, so promoting or demoting a row cannot leave a hard-coded
   model id behind in the onboarding path. */

import { isCompatibilitySupportedForModel } from './capabilities.js'
import { isRefusingFit } from './catalogBrowse.js'
import { hasLocalModelPath } from './modelState.js'

/* A fresh install: the engine is answering, it is holding no model, and there is no
   GGUF on disk to hold.

   All three matter. Offline is the backend banner's job, not onboarding's. A loaded
   model means the user is already through the funnel. And a machine that HAS models
   but has none loaded is not a fresh install — it is someone who unloaded, or whose
   only local model failed to load; sending them at a download would be wrong, and
   the Models page already owns that case.

   `models` is the dashboard's merged model list, which is reconciled against the
   live `/api/models/local` disk scan, so a record whose file is gone cannot keep the
   card hidden. */
export function isFirstRunHost({ runtime, models = [] } = {}) {
  if (runtime?.status !== 'online') return false
  if (runtime?.loaded_now || runtime?.active_model_id) return false
  return !models.some((model) => hasLocalModelPath(model))
}

/* Ascending by download size, with the catalog id as a deterministic tie-break so
   two equally sized rows cannot reorder between renders. */
function bySmallestDownload(a, b) {
  if (a.size_bytes !== b.size_bytes) return a.size_bytes - b.size_bytes
  return String(a.catalog_id).localeCompare(String(b.catalog_id))
}

/* First run promises a chat token, so encoder-only rows are not candidates even
   when they have their own supported exact-row contract. The Nomic sidecar is much
   smaller than every generative model; without this task-lane guard it would win the
   size sort, load successfully, leave Chat with no active model, and strand the
   onboarding flow it was supposed to complete. Architecture is retained as a
   defensive signal for older catalog payloads that predate task tags. */
function isGenerativeCatalogItem(item) {
  const taskTags = Array.isArray(item?.task_tags)
    ? item.task_tags.map((tag) => String(tag || '').toLowerCase())
    : []
  return item?.architecture !== 'nomic-bert' && !taskTags.includes('embeddings')
}

/* The one model a fresh install is offered.

   Rules, in order, and each is a refusal rather than a fallback:

   1. Only generative `supported_*` contract rows. The first thing a new user runs
      must be a row Camelid has cross-validated for Chat — never an encoder sidecar,
      experimental row, or merely-runnable row. The test suite pins this, because
      "just take the smallest supported row" would offer the much smaller Nomic
      embedding model and still leave Chat without a model.
   2. Only rows this host can actually load. A row the load-time fit guard would
      refuse with a 422 must never be offered; that turns one click into a dead end.
   3. Smallest first. The cost of the first token is the download, so the shortest
      honest path wins.

   Returns a tagged result rather than `null`, because the three empty cases need
   three different sentences:
     - `recommended`     — `item` is the offer.
     - `no_fitting_row`  — supported rows exist, none fits here. `smallest` is the
                           closest one, so the UI can say what it would have offered.
     - `no_supported_row`— the catalog carries no supported row at all (an unreachable
                           backend, or a contract that advertises none). */
export function recommendFirstRunModel(items = [], capabilities = null) {
  const supported = (items || []).filter(
    (item) => item?.group === 'curated'
      && isGenerativeCatalogItem(item)
      && isCompatibilitySupportedForModel(capabilities, null, item),
  )
  if (!supported.length) return { kind: 'no_supported_row', item: null, smallest: null }

  const ordered = [...supported].sort(bySmallestDownload)
  const fitting = ordered.filter((item) => !isRefusingFit(item.fit))
  if (!fitting.length) {
    return { kind: 'no_fitting_row', item: null, smallest: ordered[0] }
  }
  return { kind: 'recommended', item: fitting[0], smallest: ordered[0] }
}

/* Whether a failed activation is worth a retry button.

   A host that is too small stays too small, so offering "Try again" there is a
   button that cannot work. Memory pressure and every untyped/transport failure are
   retryable. */
const PERMANENT_REFUSAL_CODES = new Set(['model_too_large_for_host', 'unsupported_model_architecture'])

export function firstRunFailureIsRetryable(code = '') {
  return !PERMANENT_REFUSAL_CODES.has(String(code || ''))
}

/* What "Try again" must actually DO.

   Activation runs only after the artifact is observed on disk, so a failure in
   inspect/load/readiness leaves a complete, valid GGUF sitting there. Retrying by
   re-installing would re-download the whole file for nothing — and worse: the
   install endpoint drops the completed download record and starts a fresh `curl`,
   whose final `rename` onto the GGUF that already exists can fail (it is now the
   loaded/held file), at which point the backend deletes the freshly downloaded
   `.part` and reports failure. A retry that ends with LESS than it started with.

   So the retry target is decided by whether the artifact landed, not by which
   phase failed. Mirrors `CatalogLaneBrowse`'s `retryAcquisition`. */
export function firstRunRetryAction({ artifactInstalled = false } = {}) {
  return artifactInstalled ? 'activate' : 'download'
}

/* What a cancel attempt actually achieved.

   Cancelling is a request, not a fact. The backend answers three different ways and
   only one of them means "stopped": 200 removed a running download; 409
   `download_already_completed` means it finished first and KEPT its file (cancel is
   not delete); 404 means no such download — which during `starting` can simply mean
   the install has not registered yet. Reporting "Nothing was installed" on any of
   those is how the card ends up lying about a 610 MB file that is sitting on disk.

   So the outcome is decided by re-reading reality, and the observations are
   deliberately tri-state: `null` means "could not look", which must never collapse
   into "no". Refusing to claim anything we did not observe is what keeps a failed
   probe from producing a false all-clear.

     artifactInstalled === true -> `activate`: the file is there; finish the job.
     stillDownloading  === true -> `resume`:   cancel did not take; keep watching.
     either observation unknown -> `resume`:   we did not look, so we do not claim.
     both observed false        -> `canceled`: nothing running, nothing installed. */
export function firstRunCancelOutcome({
  confirmed = false,
  stillDownloading = null,
  artifactInstalled = null,
} = {}) {
  if (artifactInstalled === true) {
    return { action: 'activate', message: '' }
  }
  if (stillDownloading === true) {
    return {
      action: 'resume',
      message: 'The download did not stop — it is still running. Leaving it in progress.',
    }
  }
  if (artifactInstalled === null || stillDownloading === null) {
    return {
      action: 'resume',
      message: 'Camelid could not confirm whether the download stopped, so it is still being watched.',
    }
  }
  return {
    action: 'canceled',
    message: confirmed
      ? 'Download canceled. Nothing was installed — you can start again.'
      : 'The download is no longer running and nothing was installed — you can start again.',
  }
}
