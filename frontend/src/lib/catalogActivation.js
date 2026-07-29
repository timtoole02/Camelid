function failureMessage(body, fallback) {
  return body?.error?.message || body?.message || fallback
}

export function catalogDownloadSettlement({
  downloading,
  installed,
  sawDownload,
  settledAt = 0,
  startedAt,
  now = Date.now(),
  localScanGraceMs = 30000,
}) {
  if (downloading) return { action: 'wait', sawDownload: true, settledAt: 0 }
  if (installed) return { action: 'landed', sawDownload, settledAt: 0 }
  if (sawDownload && settledAt === 0) {
    return { action: 'wait', sawDownload, settledAt: now }
  }
  const localScanGraceElapsed = settledAt > 0 && now - settledAt >= localScanGraceMs
  const neverAppeared = !sawDownload && now - startedAt > 20000
  return {
    action: localScanGraceElapsed || neverAppeared ? 'failed' : 'wait',
    sawDownload,
    settledAt,
  }
}

export function beginCatalogSettlement(inFlightRef) {
  if (inFlightRef.current) return false
  inFlightRef.current = true
  return true
}

export function reserveCatalogAcquisition(currentCatalogId, requestedCatalogId) {
  if (currentCatalogId && currentCatalogId !== requestedCatalogId) {
    return { accepted: false, catalogId: currentCatalogId }
  }
  return { accepted: true, catalogId: requestedCatalogId }
}

export async function completeCatalogAcquisition({
  item,
  mode = 'download',
  apiBase = '',
  fetchImpl = globalThis.fetch,
  loadModelForChat,
  onStage = () => {},
}) {
  if (mode === 'download') {
    return {
      ok: true,
      started: false,
      stage: 'downloaded',
      message: 'Downloaded — the file is on disk and shown in its section above.',
    }
  }

  if (mode === 'smoke') {
    onStage('checking')
    let response
    let body
    try {
      response = await fetchImpl(`${apiBase}/api/models/runnable-smoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: item.filename }),
      })
      body = await response.json().catch(() => ({}))
    } catch (error) {
      return {
        ok: false,
        started: false,
        stage: 'checking',
        message: `Downloaded, but the model check could not run: ${String(error?.message || error)}`,
      }
    }
    if (!response.ok || body?.passed !== true) {
      return {
        ok: false,
        started: false,
        stage: 'checking',
        message: failureMessage(body, 'Downloaded, but the model did not pass its bounded check.'),
      }
    }
    return {
      ok: true,
      started: false,
      stage: 'checked',
      message: 'Downloaded and smoke-admitted — see it above in Experimental.',
    }
  }

  if (typeof loadModelForChat !== 'function') {
    return {
      ok: false,
      started: false,
      stage: 'loading',
      message: 'Downloaded, but automatic start is unavailable. Use the local model row above.',
    }
  }

  try {
    const loaded = await loadModelForChat(item.filename, { onStage })
    if (!loaded?.ok) {
      return {
        ok: false,
        started: false,
        stage: loaded?.stage || 'loading',
        // The loader's typed `error.code` (e.g. `model_too_large_for_host`) travels
        // with the message so a caller can distinguish a permanent refusal from a
        // retryable one instead of parsing prose.
        code: loaded?.code || '',
        message: loaded?.message || 'Downloaded and checked, but Camelid could not load the model.',
      }
    }
    if (loaded.embedding) {
      return {
        ok: true,
        started: false,
        embedding: true,
        stage: 'ready',
        message: 'Embedding model ready — the current Chat model was left active.',
      }
    }
  } catch (error) {
    return {
      ok: false,
      started: false,
      stage: 'loading',
      message: `Downloaded and checked, but Camelid could not load the model: ${String(error?.message || error)}`,
    }
  }

  return {
    ok: true,
    started: true,
    stage: 'ready',
    message: 'Ready — opening Chat.',
  }
}
