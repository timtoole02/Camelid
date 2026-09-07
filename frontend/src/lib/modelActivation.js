import { isEmbeddingOnlyModel } from './modelCapabilities.js'

/* The exact HTTP sequence that puts a local GGUF into its runtime, defined once.

   Two surfaces run it — the Models page's "Use" action and the first-run activation
   card — and they must not drift, because the ORDER is the contract:

     1. header-only inspect, so an architecture Camelid cannot run is refused before
        anything reads a multi-GB file;
     2. the authoritative load (generative models replace the active chat model;
        inspected embedding-only models register as sidecars without replacing it);
     3. a lane-specific readiness check: active identity plus generation readiness
        for chat, or a real bounded embedding for the encoder sidecar.

   A second hand-written copy is how one caller quietly loses step 1 or step 3. Every
   dependency is injected, so the whole sequence is testable with no backend. */

/* Stage names are the vocabulary `completeCatalogAcquisition` and the catalog rows
   already render ("Check", "Load"); they are part of this module's contract. */
const CHECKING = 'checking'
const LOADING = 'loading'

/* Inspect metadata decides whether loading a model should replace active Chat.
   New backends expose explicit task facts; `nomic-bert` remains an older-server
   fallback. An explicit generation capability always wins over stale browser
   metadata or filename-derived display hints. */
export function inspectDeclaresEmbeddingSidecar(inspect) {
  if (inspect?.generation_capable === true) return false
  if (inspect?.embedding_only === true) return true
  if (inspect?.embedding_capable === true && inspect?.generation_capable === false) return true
  if (inspect?.embedding_capable === false && inspect?.generation_capable === false) return false
  return inspect?.architecture === 'nomic-bert'
}

export function modelFilenameFromPath(value) {
  return String(value || '').split(/[\\/]/).pop() || ''
}

/* Load `filename` (a bare name inside the engine's configured models directory).

   Generative models return `{ ok: true }`; a supported embedding sidecar returns
   `{ ok: true, embedding: true }` after its sidecar readiness probe. A successful
   load that the engine wants to qualify also carries `warnings` — the advisory
   `[{ code, severity, message }]` array from the load response, passed through
   untouched and ABSENT when the engine sent nothing. Failures use
   `{ ok: false, stage, message, code, blocker }`. `code` is the backend's stable
   `error.code` when it sent one — the caller needs it to tell a permanent refusal
   (`model_too_large_for_host`, `host_memory_exhausted`) from something worth
   retrying, and to avoid rendering a raw HTTP error. `blocker` is the fail-closed
   `{ code, message }` shape the Models page renders verbatim.

   `force` skips the engine's pre-load fit preflight for this one request. It is the
   only override a GUI user has: the desktop sidecar is spawned args-only, so the
   equivalent environment variable is unreachable from the app.

   `readActiveFilename` exists so a caller that already owns a `/api/models/current`
   poll can answer the identity check from it: that response carries the full model
   record (megabytes), so a second unconditional fetch is not free. */
export async function loadLocalModelForChat({
  apiBase = '',
  filename,
  path: requestedPath = '',
  modelId = filename,
  fetchImpl = globalThis.fetch,
  onStage = () => {},
  readActiveFilename = null,
  model = null,
  force = false,
} = {}) {
  const base = String(apiBase || '').replace(/\/$/, '')
  /* A models-relative path, NOT the engine's absolute models_dir joined with '/'.
     On Windows the reported models_dir can be a `\\?\` verbatim path, and Win32 does
     not separator-normalize inside one, so the concatenation produced an invalid
     name (os error 123). The backend resolves this relative path itself. */
  const path = requestedPath || `models/${filename}`
  const requestModelId = modelId || filename
  let stage = CHECKING

  try {
    onStage(CHECKING)
    const inspectRes = await fetchImpl(`${base}/api/models/inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    })
    const inspect = await inspectRes.json().catch(() => ({}))
    if (!inspectRes.ok) {
      const message = inspect?.error?.message || `model inspection failed (HTTP ${inspectRes.status})`
      const code = inspect?.error?.code || ''
      return { ok: false, stage: CHECKING, message, code, blocker: code ? { code, message } : null }
    }
    if (inspect?.blocker) {
      return {
        ok: false,
        stage: CHECKING,
        message: inspect.blocker.message,
        code: inspect.blocker.code || '',
        blocker: inspect.blocker,
      }
    }
    const inspectedModel = {
      ...(model || {}),
      filename,
      architecture: inspect?.architecture || model?.architecture,
      embedding_capable: typeof inspect?.embedding_capable === 'boolean'
        ? inspect.embedding_capable
        : model?.embedding_capable,
      generation_capable: typeof inspect?.generation_capable === 'boolean'
        ? inspect.generation_capable
        : model?.generation_capable,
    }
    const inspectHasTaskFacts = typeof inspect?.embedding_only === 'boolean'
      || typeof inspect?.embedding_capable === 'boolean'
      || typeof inspect?.generation_capable === 'boolean'
    const embeddingModel = inspectHasTaskFacts
      ? inspectDeclaresEmbeddingSidecar(inspect)
      : isEmbeddingOnlyModel(inspectedModel)

    // Only an inspected, implemented model reaches the authoritative load.
    stage = LOADING
    onStage(LOADING)
    const loadRes = await fetchImpl(`${base}/api/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: requestModelId,
        path,
        replace: !embeddingModel,
        set_active: !embeddingModel,
        // Omitted rather than sent as false: the default request stays the shape
        // every existing server already accepts.
        ...(force ? { force: true } : {}),
      }),
    })
    if (!loadRes.ok) {
      const body = await loadRes.json().catch(() => ({}))
      const code = body?.error?.code || ''
      const message = body?.error?.message || `load failed (HTTP ${loadRes.status})`
      // A typed fail-closed load error is a blocker the caller renders verbatim;
      // `invalid_model` stays a plain error because it carries no next step.
      return {
        ok: false,
        stage: LOADING,
        message,
        code,
        blocker: code && code !== 'invalid_model' ? { code, message } : null,
      }
    }

    /* A 200 can still carry advice. The fit preflight is advisory, so a host that
       is merely busy loads and reports it here instead of refusing. An absent
       field is silence, not an empty list — nothing is synthesized from it. */
    const loaded = await loadRes.json().catch(() => ({}))
    const warnings = Array.isArray(loaded?.warnings) && loaded.warnings.length ? loaded.warnings : null

    if (embeddingModel) {
      const probeRes = await fetchImpl(`${base}/v1/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: filename,
          input: 'Camelid embedding readiness probe',
        }),
      })
      const probe = await probeRes.json().catch(() => ({}))
      const embedding = probe?.data?.[0]?.embedding
      const expectedDimensions = Number(
        inspect?.native_embedding_dimensions ?? inspect?.embedding_dimensions,
      )
      const nativeDimensionsMatch = !Number.isFinite(expectedDimensions)
        || expectedDimensions <= 0
        || embedding?.length === expectedDimensions
      const embeddingReady = Array.isArray(embedding)
        && embedding.length > 0
        && embedding.every((value) => Number.isFinite(value))
        && nativeDimensionsMatch
      if (!probeRes.ok || !embeddingReady) {
        return {
          ok: false,
          stage: LOADING,
          message: probe?.error?.message
            || `Camelid registered ${filename}, but its embedding runtime did not pass readiness.`,
          code: probe?.error?.code || '',
          blocker: null,
        }
      }
      return {
        ok: true,
        id: requestModelId,
        embedding: true,
        embedding_capable: true,
        generation_capable: false,
        ...(warnings ? { warnings } : {}),
      }
    }

    const activeFilename = readActiveFilename
      ? await readActiveFilename()
      : await readCurrentFilename(base, fetchImpl)
    if (activeFilename !== filename) {
      return {
        ok: false,
        stage: LOADING,
        message: `Camelid loaded the request but did not confirm ${filename} as the active model.`,
        code: '',
        blocker: null,
      }
    }

    const healthRes = await fetchImpl(`${base}/v1/health`)
    const health = await healthRes.json().catch(() => ({}))
    if (!healthRes.ok) {
      return {
        ok: false,
        stage: LOADING,
        message: health?.error?.message || `readiness check failed (HTTP ${healthRes.status})`,
        code: '',
        blocker: null,
      }
    }
    if (!health.loaded_now || !health.generation_ready || health.active_model_id !== requestModelId) {
      return {
        ok: false,
        stage: LOADING,
        message: `Camelid loaded ${filename}, but it is not generation-ready yet.`,
        code: '',
        blocker: null,
      }
    }
    return { ok: true, id: requestModelId, ...(warnings ? { warnings } : {}) }
  } catch (error) {
    return { ok: false, stage, message: String(error?.message || error), code: '', blocker: null }
  }
}

/* Unload one exact resident model. Passing the id matters because Camelid can
   keep a non-active embedding sidecar resident alongside the active Chat model;
   the no-body form only unloads the active model. */
export async function unloadLocalModel({
  apiBase = '',
  modelId,
  fetchImpl = globalThis.fetch,
} = {}) {
  const base = String(apiBase || '').replace(/\/$/, '')
  if (!modelId) {
    return { ok: false, message: 'Choose a loaded model to unload.', code: 'model_not_selected' }
  }

  try {
    const res = await fetchImpl(`${base}/api/models/unload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: modelId }),
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      return {
        ok: false,
        message: body?.error?.message || `unload failed (HTTP ${res.status})`,
        code: body?.error?.code || '',
      }
    }
    return { ok: true }
  } catch (error) {
    return { ok: false, message: String(error?.message || error), code: '' }
  }
}

async function readCurrentFilename(base, fetchImpl) {
  const res = await fetchImpl(`${base}/api/models/current`)
  if (!res.ok) return ''
  const body = await res.json().catch(() => ({}))
  return modelFilenameFromPath(body?.path)
}

/* Build the generation engine before the user's first prompt pays for it.

   The server does exactly this at startup when it boots WITH a model (see
   `warmup_generation_blocking` in src/api/mod.rs): one tiny self-request through the
   real chat path, because the resident engine (kernel compile, weight upload, first
   prefill) is built lazily on the first generation. A model loaded later — which is
   every model the first-run flow installs — never got that treatment, so the cold
   build landed on the user's first message and read as "the model is slow".

   Deliberately best-effort and never awaited for correctness: a failure here only
   restores the old lazy build. */
export async function warmGenerationPath({
  apiBase = '',
  modelId,
  fetchImpl = globalThis.fetch,
  signal = undefined,
} = {}) {
  const base = String(apiBase || '').replace(/\/$/, '')
  try {
    const res = await fetchImpl(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 1,
        temperature: 0,
        stream: false,
      }),
      signal,
    })
    // Drain the body so the connection is not left half-read.
    await res.json().catch(() => ({}))
    return res.ok
  } catch {
    return false
  }
}
