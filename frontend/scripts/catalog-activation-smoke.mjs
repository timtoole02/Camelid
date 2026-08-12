import assert from 'node:assert/strict'
import { beginCatalogSettlement, catalogDownloadSettlement, completeCatalogAcquisition } from '../src/lib/catalogActivation.js'

function response({ ok = true, body = {} } = {}) {
  return { ok, json: async () => body }
}

const curated = {
  group: 'supported',
  filename: 'Llama-3.2-1B-Instruct-Q8_0.gguf',
  oracle_qualified: true,
}

{
  const ref = { current: false }
  assert.equal(beginCatalogSettlement(ref), true)
  assert.equal(ref.current, true)
  assert.equal(beginCatalogSettlement(ref), false, 'settlement must reject re-entry')
  ref.current = false
  assert.equal(beginCatalogSettlement(ref), true, 'explicit retry may re-enter')
}

{
  const downloading = catalogDownloadSettlement({
    downloading: true,
    installed: false,
    sawDownload: false,
    startedAt: 100,
    now: 200,
  })
  assert.deepEqual(downloading, { action: 'wait', sawDownload: true, settledAt: 0 })

  const awaitingScan = catalogDownloadSettlement({
    downloading: false,
    installed: false,
    sawDownload: true,
    settledAt: 0,
    startedAt: 100,
    now: 1000,
  })
  assert.deepEqual(awaitingScan, { action: 'wait', sawDownload: true, settledAt: 1000 })

  const landed = catalogDownloadSettlement({
    downloading: false,
    installed: true,
    sawDownload: true,
    settledAt: 1000,
    startedAt: 100,
    now: 2000,
  })
  assert.equal(landed.action, 'landed', 'a completed local scan must win during the grace window')

  const missing = catalogDownloadSettlement({
    downloading: false,
    installed: false,
    sawDownload: true,
    settledAt: 1000,
    startedAt: 100,
    now: 31000,
  })
  assert.equal(missing.action, 'failed', 'a missing file must fail after the local scan grace window')
}

{
  const stages = []
  const requests = []
  const loads = []
  const result = await completeCatalogAcquisition({
    item: curated,
    mode: 'start',
    apiBase: 'http://camelid.test',
    fetchImpl: async (url, options) => {
      requests.push({ url, options })
      return response({ body: { passed: true } })
    },
    loadModelForChat: async (filename, { onStage }) => {
      onStage('checking')
      onStage('loading')
      loads.push(filename)
      return { ok: true }
    },
    onStage: (stage) => stages.push(stage),
  })
  assert.deepEqual(stages, ['checking', 'loading'])
  assert.equal(requests.length, 0, 'supported serving rows use inspect/load, not runnable smoke admission')
  assert.deepEqual(loads, [curated.filename])
  assert.equal(result.started, true)
  assert.equal(result.stage, 'ready')
}

{
  const result = await completeCatalogAcquisition({
    item: {
      group: 'curated',
      filename: 'nomic-embed-text-v1.5.Q8_0.gguf',
      oracle_qualified: true,
    },
    mode: 'start',
    loadModelForChat: async () => ({ ok: true, embedding: true }),
  })
  assert.equal(result.ok, true)
  assert.equal(result.started, false, 'an encoder sidecar must not navigate to Chat')
  assert.equal(result.embedding, true)
  assert.match(result.message, /current Chat model was left active/)
}

{
  let loadCalled = false
  const result = await completeCatalogAcquisition({
    item: { ...curated, group: 'curated' },
    mode: 'smoke',
    fetchImpl: async () => response({
      ok: false,
      body: { error: { message: 'coherence check failed' } },
    }),
    loadModelForChat: async () => {
      loadCalled = true
      return { ok: true }
    },
  })
  assert.equal(result.ok, false)
  assert.equal(result.stage, 'checking')
  assert.equal(result.message, 'coherence check failed')
  assert.equal(loadCalled, false, 'a failed model check must never attempt a load')
}

{
  const result = await completeCatalogAcquisition({
    item: curated,
    mode: 'start',
    loadModelForChat: async () => ({ ok: false, stage: 'loading', message: 'model_too_large_for_host' }),
  })
  assert.equal(result.ok, false)
  assert.equal(result.stage, 'loading')
  assert.equal(result.message, 'model_too_large_for_host')
}

{
  const result = await completeCatalogAcquisition({
    item: curated,
    mode: 'start',
    loadModelForChat: async () => ({ ok: false, stage: 'checking', message: 'inspect unavailable' }),
  })
  assert.equal(result.stage, 'checking')
  assert.equal(result.message, 'inspect unavailable')
}

{
  let sideEffect = false
  const result = await completeCatalogAcquisition({
    item: { ...curated, oracle_qualified: false },
    mode: 'download',
    fetchImpl: async () => {
      sideEffect = true
      return response()
    },
    loadModelForChat: async () => {
      sideEffect = true
      return { ok: true }
    },
  })
  assert.equal(sideEffect, false, 'download-only rows must not claim a smoke check or serving path')
  assert.equal(result.started, false)
}

{
  let sideEffect = false
  const result = await completeCatalogAcquisition({
    item: { group: 'experimental', filename: 'arbitrary.gguf', oracle_qualified: true },
    mode: 'download',
    fetchImpl: async () => {
      sideEffect = true
      return response()
    },
    loadModelForChat: async () => {
      sideEffect = true
      return { ok: true }
    },
  })
  assert.equal(sideEffect, false, 'arbitrary Hugging Face rows must remain download-only')
  assert.equal(result.started, false)
  assert.equal(result.stage, 'downloaded')
}

console.log('catalog activation smoke: all checks passed')
